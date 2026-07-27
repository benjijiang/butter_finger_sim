#!/usr/bin/env python3
"""Pi-only wake-word voice chat with emotional arm gestures.

Examples:

    python examples/voice_chat.py --list-audio-devices
    python examples/voice_chat.py --device-name "My Speaker" --wake-test
    python examples/voice_chat.py --device-name "My Speaker" --dry-run
    python examples/voice_chat.py --device-name "My Speaker" --confirm-hardware
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from butter_finger.arm import ArmBackend
from butter_finger.voice.audio import (
    AudioDeviceError,
    AudioIO,
    SoundDeviceDuplexAudio,
    format_audio_devices,
    query_audio_devices,
)
from butter_finger.voice.config import ALLOWED_MODELS, VoiceConfig, load_voice_config
from butter_finger.voice.realtime import (
    RealtimeTransport,
    WebSocketRealtimeTransport,
)
from butter_finger.voice.runtime import DryRunArm, VoiceChatRuntime
from butter_finger.voice.wake import (
    PocketSphinxWakeWordDetector,
    WakeWordDetector,
    WakeWordError,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device-name",
        help="case-insensitive Bluetooth audio device-name substring",
    )
    parser.add_argument("--input-device-id", type=int)
    parser.add_argument("--output-device-id", type=int)
    parser.add_argument(
        "--list-audio-devices",
        action="store_true",
        help="list audio endpoint IDs and exit without opening the arm",
    )
    parser.add_argument(
        "--audio-loopback",
        metavar="SECONDS",
        type=float,
        help="record then replay Bluetooth audio; never opens the arm",
    )
    parser.add_argument(
        "--wake-test",
        action="store_true",
        help="test local wake detection only; never opens the arm or cloud",
    )
    parser.add_argument(
        "--wake-threshold",
        type=float,
        help="override the PocketSphinx keyphrase threshold",
    )
    parser.add_argument(
        "--model",
        choices=ALLOWED_MODELS,
        help="Realtime model (use the mini model for lower cost)",
    )
    parser.add_argument(
        "--inactivity-timeout",
        type=float,
        help="seconds without user speech before returning to wake mode",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run voice chat with a validated logging arm; never sends PWM",
    )
    parser.add_argument(
        "--confirm-hardware",
        action="store_true",
        help="required acknowledgement before opening RaspberryPiArm",
    )
    return parser


def _overridden_config(args: argparse.Namespace) -> VoiceConfig:
    config = load_voice_config()
    updates: dict[str, Any] = {}
    for argument, field in (
        ("device_name", "device_name"),
        ("input_device_id", "input_device_id"),
        ("output_device_id", "output_device_id"),
        ("wake_threshold", "wake_threshold"),
        ("model", "model"),
        ("inactivity_timeout", "inactivity_timeout_s"),
    ):
        value = getattr(args, argument)
        if value is not None:
            updates[field] = value
    return replace(config, **updates)


def _create_audio(config: VoiceConfig) -> AudioIO:
    return SoundDeviceDuplexAudio(
        device_name=config.device_name,
        input_device_id=config.input_device_id,
        output_device_id=config.output_device_id,
        chunk_ms=config.audio_chunk_ms,
    )


def _create_wake(config: VoiceConfig) -> WakeWordDetector:
    return PocketSphinxWakeWordDetector(
        phrase=config.wake_phrase,
        threshold=config.wake_threshold,
        sample_rate_hz=config.wake_sample_rate_hz,
    )


def _create_arm(dry_run: bool) -> ArmBackend:
    if dry_run:
        return DryRunArm()
    # Importing the concrete backend is safe here because hardware confirmation
    # and duplex audio validation have already succeeded.
    from butter_finger.backends.raspberry_pi_arm import RaspberryPiArm

    return RaspberryPiArm()


def _loopback(audio: AudioIO, config: VoiceConfig, seconds: float) -> None:
    chunks = max(1, math.ceil(seconds * 1000 / config.audio_chunk_ms))
    print(f"Recording {seconds:g}s from the selected input endpoint...")
    captured = [
        audio.read(config.api_sample_rate_hz)
        for _ in range(chunks)
    ]
    print("Playing the capture through the selected output endpoint...")
    for chunk in captured:
        audio.write(chunk, config.api_sample_rate_hz)
    audio.drain()
    print("Loopback complete.")


def _wake_test(
    audio: AudioIO,
    detector: WakeWordDetector,
    config: VoiceConfig,
) -> None:
    print(
        f"Local wake test active at threshold {config.wake_threshold:g}. "
        f"Say {config.wake_phrase!r}; press Ctrl-C to stop."
    )
    detector.reset()
    while True:
        if detector.process(audio.read(config.wake_sample_rate_hz)):
            print(f"DETECTED: {config.wake_phrase}")


def main(
    argv: Sequence[str] | None = None,
    *,
    audio_factory: Callable[[VoiceConfig], AudioIO] | None = None,
    wake_factory: Callable[[VoiceConfig], WakeWordDetector] | None = None,
    arm_factory: Callable[[bool], ArmBackend] | None = None,
    transport_factory: Callable[[str, VoiceConfig], RealtimeTransport] | None = None,
) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.input_device_id is not None and args.input_device_id < 0:
        parser.error("--input-device-id must be non-negative")
    if args.output_device_id is not None and args.output_device_id < 0:
        parser.error("--output-device-id must be non-negative")
    for value, label in (
        (args.audio_loopback, "--audio-loopback"),
        (args.wake_threshold, "--wake-threshold"),
        (args.inactivity_timeout, "--inactivity-timeout"),
    ):
        if value is not None and (
            not math.isfinite(value) or value <= 0
        ):
            parser.error(f"{label} must be a finite number greater than zero")

    if args.list_audio_devices:
        try:
            print(format_audio_devices(query_audio_devices()))
        except AudioDeviceError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        return 0

    config = _overridden_config(args)
    make_audio = audio_factory if audio_factory is not None else _create_audio
    make_wake = wake_factory if wake_factory is not None else _create_wake
    audio: AudioIO | None = None
    detector: WakeWordDetector | None = None

    try:
        audio = make_audio(config)
        # Duplex validation happens before any physical-arm object is opened.
        audio.open()
        detector = make_wake(config)
    except (AudioDeviceError, WakeWordError, ValueError) as exc:
        if audio is not None:
            audio.close()
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.audio_loopback is not None or args.wake_test:
        try:
            if args.audio_loopback is not None:
                _loopback(audio, config, args.audio_loopback)
            else:
                _wake_test(audio, detector, config)
        except KeyboardInterrupt:
            print("\nDiagnostic stopped.")
        finally:
            detector.close()
            audio.close()
        return 0

    if not args.dry_run and not args.confirm_hardware:
        detector.close()
        audio.close()
        parser.error(
            "real voice chat requires --confirm-hardware; use --dry-run "
            "to guarantee that no PWM is sent"
        )

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        detector.close()
        audio.close()
        print(
            "ERROR: OPENAI_API_KEY is not set. Voice chat reads the key only "
            "from the environment.",
            file=sys.stderr,
        )
        return 2

    make_arm = arm_factory if arm_factory is not None else _create_arm
    try:
        arm = make_arm(args.dry_run)
    except Exception as exc:
        detector.close()
        audio.close()
        print(f"ERROR: could not open arm: {exc}", file=sys.stderr)
        return 1

    if transport_factory is None:
        make_transport = lambda key, cfg: WebSocketRealtimeTransport(
            api_key=key,
            model=cfg.model,
        )
    else:
        make_transport = transport_factory

    runtime = VoiceChatRuntime(
        config=config,
        audio=audio,
        wake_detector=detector,
        arm=arm,
        transport_factory=lambda: make_transport(api_key, config),
        audio_is_open=True,
    )
    try:
        runtime.run_forever()
    except KeyboardInterrupt:
        print("\nStopping voice chat safely...")
        return_code = 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return_code = 1
    else:
        return_code = 0
    finally:
        runtime.shutdown()
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
