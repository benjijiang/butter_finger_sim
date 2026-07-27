"""Wake-first voice conversation lifecycle and safe arm cleanup."""
from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from butter_finger.actions import ActionRunner
from butter_finger.arm import ArmBackend, JointLimitError, UnknownJointError
from butter_finger.config import ActionConfig, ArmConfig, load_action_config, load_arm_config
from butter_finger.voice.audio import AudioDeviceError, AudioIO
from butter_finger.voice.config import VoiceConfig
from butter_finger.voice.coordinator import TurnCoordinator, TurnState
from butter_finger.voice.emotions import validate_conversational_actions
from butter_finger.voice.realtime import (
    FunctionCall,
    RealtimeError,
    RealtimeTransport,
    UsageReport,
    append_audio_event,
    build_session_update,
    decode_audio_delta,
    emotion_retry_event,
    extract_function_calls,
    extract_response_transcript,
    function_output_event,
    spoken_response_event,
)
from butter_finger.voice.wake import WakeWordDetector


def _stream_to_stdout(text: str) -> None:
    print(text, end="", flush=True)


class DryRunArm(ArmBackend):
    """Validated ArmBackend that logs commands and never emits PWM."""

    def __init__(
        self,
        config: ArmConfig | None = None,
        *,
        output: Callable[[str], None] = print,
    ) -> None:
        self._config = config if config is not None else load_arm_config()
        self._positions = self._config.home_pose
        self._output = output
        self.events: list[tuple[Any, ...]] = []

    def validate_targets(self, targets_rad: Mapping[str, float]) -> None:
        for joint, position in targets_rad.items():
            if joint not in self._config.sim_limits:
                raise UnknownJointError(f"Unknown joint {joint!r}")
            if not self._config.sim_limits[joint].contains(float(position)):
                raise JointLimitError(
                    f"Target {position} for {joint!r} is outside its limits"
                )

    def move_joint(
        self,
        joint: str,
        position_rad: float,
        *,
        duration_s: float | None = None,
    ) -> None:
        self.move_joints({joint: position_rad}, duration_s=duration_s)

    def move_joints(
        self,
        targets_rad: Mapping[str, float],
        *,
        duration_s: float | None = None,
    ) -> None:
        self.validate_targets(targets_rad)
        targets = dict(targets_rad)
        self._positions.update(targets)
        self.events.append(("move", targets, duration_s))
        self._output(
            f"[dry-run arm] move {targets} over {duration_s or 0:.2f}s"
        )

    def get_joint_positions(self) -> dict[str, float]:
        return dict(self._positions)

    def go_home(self) -> None:
        self._positions = self._config.home_pose
        self.events.append(("home",))
        self._output("[dry-run arm] home")

    def disconnect(self) -> None:
        self.events.append(("disconnect",))


class VoiceChatRuntime:
    """Own audio, cloud conversation, and arm for one exclusive process."""

    def __init__(
        self,
        *,
        config: VoiceConfig,
        audio: AudioIO,
        wake_detector: WakeWordDetector,
        arm: ArmBackend,
        transport_factory: Callable[[], RealtimeTransport],
        action_config: ActionConfig | None = None,
        output: Callable[[str], None] = print,
        stream_output: Callable[[str], None] = _stream_to_stdout,
        monotonic: Callable[[], float] = time.monotonic,
        audio_is_open: bool = False,
    ) -> None:
        self.config = config
        self.audio = audio
        self.wake_detector = wake_detector
        self.arm = arm
        self._transport_factory = transport_factory
        self._action_config = (
            action_config if action_config is not None else load_action_config()
        )
        validate_conversational_actions(self._action_config.action_names)
        self._output = output
        self._stream_output = stream_output
        self._monotonic = monotonic
        self._audio_is_open = audio_is_open
        self._stop = threading.Event()
        self._transport: RealtimeTransport | None = None
        self._coordinator: TurnCoordinator | None = None
        self._arm_started = False
        self._shutdown = False
        self._transcript_stream_started = False

    def request_stop(self) -> None:
        self._stop.set()

    def _startup(self) -> None:
        if not self._audio_is_open:
            self.audio.open()
            self._audio_is_open = True
        self.arm.go_home()
        self._arm_started = True
        idle = load_arm_config().poses[self.config.idle_pose]
        self.arm.validate_targets(idle)
        self.arm.move_joints(
            idle,
            duration_s=self.config.idle_move_duration_s,
        )

    def _wait_for_wake(self) -> bool:
        self._output(
            f"Waiting for wake phrase: {self.config.wake_phrase!r}"
        )
        self.wake_detector.reset()
        while not self._stop.is_set():
            pcm = self.audio.read(self.config.wake_sample_rate_hz)
            if self.wake_detector.process(pcm):
                self._output("Wake phrase detected. Opening Realtime session...")
                return True
        return False

    def _handle_decision(
        self,
        transport: RealtimeTransport,
        call: FunctionCall,
    ) -> None:
        coordinator = self._coordinator
        if coordinator is None:
            return
        decision = coordinator.handle_function_call(call)
        if decision.duplicate:
            return
        transport.send_event(
            function_output_event(
                decision.call_id,
                accepted=decision.accepted,
                action=decision.action,
                error=decision.error,
            )
        )
        if decision.accepted:
            transport.send_event(spoken_response_event())
        elif decision.should_retry:
            transport.send_event(emotion_retry_event())
        else:
            raise RealtimeError(
                "The model did not provide one safe emotional action after "
                f"retry ({decision.error})"
            )

    def _finish_turn(self) -> None:
        coordinator = self._coordinator
        if coordinator is None or coordinator.state != TurnState.SPEAKING:
            return
        self.audio.drain()
        completed = coordinator.complete_spoken_response()
        if self._transcript_stream_started:
            self._stream_output("\n")
        else:
            self._output("Assistant: [no transcript]")
        self._transcript_stream_started = False
        self._output(f"Emotion: {completed.action}")
        self._output(f"API usage: {completed.usage.compact()}")

    def _conversation(self) -> None:
        transport = self._transport_factory()
        coordinator = TurnCoordinator(
            ActionRunner(self.arm, self._action_config)
        )
        self._transport = transport
        self._coordinator = coordinator
        pending_calls: dict[str, FunctionCall] = {}
        last_user_speech = self._monotonic()
        try:
            transport.connect()
            transport.send_event(build_session_update(self.config))
            self._output("Voice chat active. Speak naturally.")
            while not self._stop.is_set():
                state = coordinator.state
                if state == TurnState.LISTENING:
                    pcm = self.audio.read(self.config.api_sample_rate_hz)
                    transport.send_event(append_audio_event(pcm))

                event = transport.receive_event(timeout_s=0.001)
                if event is None:
                    if (
                        self._monotonic() - last_user_speech
                        >= self.config.inactivity_timeout_s
                    ):
                        self._output(
                            "No user speech for "
                            f"{self.config.inactivity_timeout_s:g}s; "
                            "returning to wake-word mode."
                        )
                        return
                    continue

                event_type = event.get("type")
                if event_type == "error":
                    error = event.get("error", {})
                    raise RealtimeError(f"Realtime server error: {error}")
                if event_type == "input_audio_buffer.speech_started":
                    if coordinator.on_speech_started():
                        last_user_speech = self._monotonic()
                        self._transcript_stream_started = False
                    continue
                if event_type == "input_audio_buffer.speech_stopped":
                    coordinator.on_speech_stopped()
                    continue
                if event_type == "response.function_call_arguments.done":
                    for call in extract_function_calls(event):
                        pending_calls[call.call_id] = call
                    continue
                if event_type == "response.output_audio.delta":
                    if coordinator.state == TurnState.SPEAKING:
                        self.audio.write(
                            decode_audio_delta(event),
                            self.config.api_sample_rate_hz,
                        )
                    continue
                if event_type == "response.output_audio_transcript.delta":
                    delta = event.get("delta")
                    if isinstance(delta, str):
                        if not self._transcript_stream_started:
                            self._stream_output("Assistant: ")
                            self._transcript_stream_started = True
                        self._stream_output(delta)
                        coordinator.append_transcript(delta)
                    continue
                if event_type == "response.output_audio_transcript.done":
                    transcript = event.get("transcript")
                    if (
                        isinstance(transcript, str)
                        and transcript
                        and not self._transcript_stream_started
                    ):
                        self._stream_output("Assistant: ")
                        self._stream_output(transcript)
                        self._transcript_stream_started = True
                        coordinator.append_transcript(transcript)
                    continue
                if event_type != "response.done":
                    continue

                coordinator.add_usage(UsageReport.from_response_done(event))
                if (
                    coordinator.state == TurnState.SPEAKING
                    and not self._transcript_stream_started
                ):
                    transcript = extract_response_transcript(event)
                    if transcript:
                        self._stream_output("Assistant: ")
                        self._stream_output(transcript)
                        self._transcript_stream_started = True
                        coordinator.append_transcript(transcript)
                calls_by_id = dict(pending_calls)
                for call in extract_function_calls(event):
                    calls_by_id[call.call_id] = call
                calls = tuple(calls_by_id.values())
                pending_calls.clear()
                if len(calls) > 1:
                    for call in calls:
                        transport.send_event(
                            function_output_event(
                                call.call_id,
                                accepted=False,
                                error="exactly one emotion call is allowed",
                            )
                        )
                    raise RealtimeError(
                        "The model returned more than one emotion call; "
                        "no gesture was started"
                    )
                for call in calls:
                    self._handle_decision(transport, call)
                if not calls and coordinator.state == TurnState.SPEAKING:
                    self._finish_turn()
                elif not calls and coordinator.state == TurnState.WAITING_EMOTION:
                    raise RealtimeError(
                        "The model response did not contain an emotion call"
                    )
        finally:
            try:
                if self._transcript_stream_started:
                    self._stream_output("\n")
                    self._transcript_stream_started = False
                try:
                    coordinator.abandon_response()
                except Exception as exc:
                    self._output(f"Gesture ended with an error: {exc}")
                try:
                    coordinator.stop()
                except Exception as exc:
                    self._output(f"Gesture worker cleanup error: {exc}")
            finally:
                transport.close()
                self._transport = None
                self._coordinator = None

    def _recover_audio(self, exc: BaseException) -> bool:
        self._output(f"Audio unavailable ({exc}); waiting for Bluetooth...")
        while not self._stop.is_set():
            if self._stop.wait(self.config.reconnect_delay_s):
                return False
            try:
                self.audio.reconnect()
                self.wake_detector.reset()
                self._output("Bluetooth audio reconnected.")
                return True
            except Exception as reconnect_error:
                self._output(f"Bluetooth still unavailable: {reconnect_error}")
        return False

    def run_forever(self) -> None:
        self._startup()
        while not self._stop.is_set():
            try:
                if not self._wait_for_wake():
                    break
                self._conversation()
            except AudioDeviceError as exc:
                if not self._recover_audio(exc):
                    break
            except RealtimeError as exc:
                self._output(
                    f"{exc}. Returning to local wake-word mode; room audio "
                    "is not uploaded while disconnected."
                )
                self.wake_detector.reset()

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self.request_stop()
        coordinator = self._coordinator
        if coordinator is not None:
            try:
                coordinator.abandon_response()
            except Exception as exc:
                self._output(f"Gesture ended with an error: {exc}")
        transport = self._transport
        if transport is not None:
            transport.close()
        try:
            if self._arm_started:
                try:
                    self.arm.go_home()
                except Exception as exc:
                    self._output(f"Could not return arm home: {exc}")
        finally:
            try:
                self.wake_detector.close()
            finally:
                try:
                    self.audio.close()
                finally:
                    self.arm.disconnect()
