"""Bluetooth-capable duplex PCM audio with explicit endpoint validation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class AudioDeviceError(RuntimeError):
    """A usable bidirectional input/output device is unavailable."""


class AudioIO(Protocol):
    """Runtime audio contract used by the dependency-free coordinator."""

    def open(self) -> None: ...

    def read(self, sample_rate_hz: int) -> bytes: ...

    def write(self, pcm16: bytes, sample_rate_hz: int) -> None: ...

    def drain(self) -> None: ...

    def reconnect(self) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate_hz: int

    @property
    def has_input(self) -> bool:
        return self.max_input_channels > 0

    @property
    def has_output(self) -> bool:
        return self.max_output_channels > 0


@dataclass(frozen=True)
class DuplexSelection:
    input: AudioDevice
    output: AudioDevice


def _device_from_mapping(index: int, raw: Any) -> AudioDevice:
    try:
        return AudioDevice(
            index=index,
            name=str(raw["name"]),
            max_input_channels=int(raw["max_input_channels"]),
            max_output_channels=int(raw["max_output_channels"]),
            default_sample_rate_hz=int(round(float(raw["default_samplerate"]))),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AudioDeviceError(
            f"Audio backend returned malformed device entry {index}"
        ) from exc


def audio_devices_from_query(raw_devices: Any) -> tuple[AudioDevice, ...]:
    """Normalize sounddevice's query result for display and testing."""
    return tuple(
        _device_from_mapping(index, raw)
        for index, raw in enumerate(raw_devices)
    )


def select_duplex_device(
    devices: tuple[AudioDevice, ...],
    *,
    name_substring: str | None = None,
    input_device_id: int | None = None,
    output_device_id: int | None = None,
    default_input_id: int | None = None,
    default_output_id: int | None = None,
) -> DuplexSelection:
    """Select and validate capture/playback endpoints before arm startup."""
    by_id = {device.index: device for device in devices}

    def explicit(device_id: int | None, direction: str) -> AudioDevice | None:
        if device_id is None:
            return None
        try:
            device = by_id[device_id]
        except KeyError as exc:
            raise AudioDeviceError(
                f"{direction} audio device ID {device_id} does not exist"
            ) from exc
        capable = device.has_input if direction == "input" else device.has_output
        if not capable:
            raise AudioDeviceError(
                f"Audio device {device_id} ({device.name!r}) has no "
                f"{direction} endpoint"
            )
        return device

    selected_input = explicit(input_device_id, "input")
    selected_output = explicit(output_device_id, "output")

    candidates = devices
    if name_substring and (
        selected_input is None or selected_output is None
    ):
        needle = name_substring.casefold()
        candidates = tuple(
            device for device in devices if needle in device.name.casefold()
        )
        if not candidates:
            raise AudioDeviceError(
                f"No audio device name contains {name_substring!r}. "
                "Use --list-audio-devices to inspect endpoint IDs."
            )

    if selected_input is None:
        selected_input = next(
            (device for device in candidates if device.has_input),
            None,
        )
    if selected_output is None:
        if selected_input is not None:
            selected_output = next(
                (
                    device
                    for device in candidates
                    if device.has_output and device.name == selected_input.name
                ),
                None,
            )
        if selected_output is None:
            selected_output = next(
                (device for device in candidates if device.has_output),
                None,
            )

    if name_substring is None:
        if input_device_id is None and default_input_id in by_id:
            default_in = by_id[default_input_id]
            if default_in.has_input:
                selected_input = default_in
        if output_device_id is None and default_output_id in by_id:
            default_out = by_id[default_output_id]
            if default_out.has_output:
                selected_output = default_out

    if selected_output is not None and selected_input is None:
        raise AudioDeviceError(
            "The selected Bluetooth device exposes playback only (usually "
            "A2DP). Enable its bidirectional HFP/HSP headset or speakerphone "
            "profile, then verify that an input endpoint appears."
        )
    if selected_input is None or selected_output is None:
        raise AudioDeviceError(
            "A capture endpoint and a playback endpoint are both required. "
            "Use --list-audio-devices and select a Bluetooth HFP/HSP profile."
        )
    return DuplexSelection(input=selected_input, output=selected_output)


def resample_pcm16(
    pcm16: bytes,
    source_rate_hz: int,
    target_rate_hz: int,
) -> bytes:
    """Resample mono little-endian PCM16 using linear interpolation."""
    if source_rate_hz <= 0 or target_rate_hz <= 0:
        raise ValueError("sample rates must be positive")
    if len(pcm16) % 2:
        raise ValueError("PCM16 byte length must be even")
    if source_rate_hz == target_rate_hz or not pcm16:
        return pcm16
    source = np.frombuffer(pcm16, dtype="<i2").astype(np.float64)
    if len(source) == 1:
        target_length = max(1, round(target_rate_hz / source_rate_hz))
        return np.repeat(source, target_length).astype("<i2").tobytes()
    target_length = max(
        1,
        round(len(source) * target_rate_hz / source_rate_hz),
    )
    source_positions = np.arange(len(source), dtype=np.float64)
    target_positions = np.linspace(
        0.0,
        len(source) - 1,
        target_length,
        dtype=np.float64,
    )
    target = np.interp(target_positions, source_positions, source)
    return np.clip(np.rint(target), -32768, 32767).astype("<i2").tobytes()


def query_audio_devices() -> tuple[AudioDevice, ...]:
    """List devices, importing sounddevice only when this is called."""
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise AudioDeviceError(
            "sounddevice is not installed; install the 'voice' optional "
            "dependencies"
        ) from exc
    return audio_devices_from_query(sd.query_devices())


def format_audio_devices(devices: tuple[AudioDevice, ...]) -> str:
    lines = ["ID  IN  OUT  RATE    NAME"]
    for device in devices:
        lines.append(
            f"{device.index:<3} {device.max_input_channels:<3} "
            f"{device.max_output_channels:<4} "
            f"{device.default_sample_rate_hz:<7} {device.name}"
        )
    return "\n".join(lines)


class SoundDeviceDuplexAudio:
    """Blocking mono PCM16 duplex streams suitable for a Pi worker loop."""

    def __init__(
        self,
        *,
        device_name: str | None,
        input_device_id: int | None,
        output_device_id: int | None,
        chunk_ms: int,
    ) -> None:
        if chunk_ms <= 0:
            raise ValueError("chunk_ms must be positive")
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise AudioDeviceError(
                "sounddevice is not installed; run "
                "pip install -e '.[voice]'"
            ) from exc
        self._sd = sd
        self._device_name = device_name
        self._input_device_id = input_device_id
        self._output_device_id = output_device_id
        self._chunk_ms = chunk_ms
        self._selection: DuplexSelection | None = None
        self._input_stream: Any = None
        self._output_stream: Any = None

    @property
    def selection(self) -> DuplexSelection:
        if self._selection is None:
            raise AudioDeviceError("audio streams are not open")
        return self._selection

    def _defaults(self) -> tuple[int | None, int | None]:
        default = getattr(self._sd, "default", None)
        pair = getattr(default, "device", (None, None))
        try:
            input_id, output_id = pair
        except (TypeError, ValueError):
            return None, None
        return (
            int(input_id) if input_id is not None and input_id >= 0 else None,
            int(output_id) if output_id is not None and output_id >= 0 else None,
        )

    def open(self) -> None:
        if self._input_stream is not None or self._output_stream is not None:
            return
        devices = audio_devices_from_query(self._sd.query_devices())
        default_input, default_output = self._defaults()
        selection = select_duplex_device(
            devices,
            name_substring=self._device_name,
            input_device_id=self._input_device_id,
            output_device_id=self._output_device_id,
            default_input_id=default_input,
            default_output_id=default_output,
        )
        try:
            self._sd.check_input_settings(
                device=selection.input.index,
                channels=1,
                dtype="int16",
                samplerate=selection.input.default_sample_rate_hz,
            )
            self._sd.check_output_settings(
                device=selection.output.index,
                channels=1,
                dtype="int16",
                samplerate=selection.output.default_sample_rate_hz,
            )
            input_frames = max(
                1,
                round(
                    selection.input.default_sample_rate_hz
                    * self._chunk_ms
                    / 1000
                ),
            )
            self._input_stream = self._sd.RawInputStream(
                device=selection.input.index,
                samplerate=selection.input.default_sample_rate_hz,
                blocksize=input_frames,
                channels=1,
                dtype="int16",
            )
            self._output_stream = self._sd.RawOutputStream(
                device=selection.output.index,
                samplerate=selection.output.default_sample_rate_hz,
                channels=1,
                dtype="int16",
            )
            self._input_stream.start()
            self._output_stream.start()
        except Exception as exc:
            self.close()
            raise AudioDeviceError(
                "Could not open both Bluetooth audio endpoints. Select the "
                "HFP/HSP speakerphone profile instead of playback-only A2DP."
            ) from exc
        self._selection = selection

    def read(self, sample_rate_hz: int) -> bytes:
        stream = self._input_stream
        if stream is None:
            raise AudioDeviceError("audio input stream is not open")
        source_rate = self.selection.input.default_sample_rate_hz
        frames = max(1, round(source_rate * self._chunk_ms / 1000))
        try:
            data, _overflowed = stream.read(frames)
        except Exception as exc:
            raise AudioDeviceError("Bluetooth input disconnected") from exc
        return resample_pcm16(bytes(data), source_rate, sample_rate_hz)

    def write(self, pcm16: bytes, sample_rate_hz: int) -> None:
        stream = self._output_stream
        if stream is None:
            raise AudioDeviceError("audio output stream is not open")
        target_rate = self.selection.output.default_sample_rate_hz
        data = resample_pcm16(pcm16, sample_rate_hz, target_rate)
        try:
            stream.write(data)
        except Exception as exc:
            raise AudioDeviceError("Bluetooth output disconnected") from exc

    def drain(self) -> None:
        """Wait for queued playback, then resume the output stream."""
        stream = self._output_stream
        if stream is None:
            return
        try:
            stream.stop()
            stream.start()
        except Exception as exc:
            raise AudioDeviceError("Bluetooth output disconnected") from exc

    def reconnect(self) -> None:
        self.close()
        self.open()

    def close(self) -> None:
        for name in ("_input_stream", "_output_stream"):
            stream = getattr(self, name)
            setattr(self, name, None)
            if stream is None:
                continue
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        self._selection = None
