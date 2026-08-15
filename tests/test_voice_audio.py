"""Bluetooth endpoint selection and PCM resampling tests."""
from __future__ import annotations

import numpy as np
import pytest

from butter_finger.voice.audio import (
    AudioDevice,
    AudioDeviceError,
    resample_pcm16,
    select_duplex_device,
)


def device(
    index: int,
    name: str,
    *,
    inputs: int = 0,
    outputs: int = 0,
    rate: int = 16000,
) -> AudioDevice:
    return AudioDevice(index, name, inputs, outputs, rate)


def test_name_selector_accepts_separate_bluetooth_hfp_endpoints() -> None:
    devices = (
        device(1, "Desk Speaker A2DP", outputs=2, rate=48000),
        device(2, "Desk Speaker HFP", inputs=1, rate=16000),
        device(3, "Desk Speaker HFP", outputs=1, rate=16000),
    )

    selected = select_duplex_device(
        devices,
        name_substring="desk speaker",
    )

    assert selected.input.index == 2
    assert selected.output.index == 3


def test_playback_only_bluetooth_device_explains_hfp_requirement() -> None:
    devices = (device(4, "Pocket Box A2DP", outputs=2, rate=48000),)

    with pytest.raises(AudioDeviceError, match="A2DP.*HFP/HSP"):
        select_duplex_device(devices, name_substring="Pocket Box")


def test_explicit_endpoint_must_support_requested_direction() -> None:
    devices = (
        device(1, "Output", outputs=2),
        device(2, "Input", inputs=1),
    )
    with pytest.raises(AudioDeviceError, match="no input endpoint"):
        select_duplex_device(
            devices,
            input_device_id=1,
            output_device_id=1,
        )


def test_explicit_ids_override_a_name_substring() -> None:
    devices = (
        device(1, "Chosen Mic", inputs=1),
        device(2, "Chosen Speaker", outputs=1),
    )

    selected = select_duplex_device(
        devices,
        name_substring="does not match",
        input_device_id=1,
        output_device_id=2,
    )

    assert selected.input.index == 1
    assert selected.output.index == 2


def test_default_input_and_output_ids_are_used() -> None:
    devices = (
        device(1, "Mic", inputs=1),
        device(2, "Speaker", outputs=2, rate=48000),
    )

    selected = select_duplex_device(
        devices,
        default_input_id=1,
        default_output_id=2,
    )

    assert selected.input.index == 1
    assert selected.output.index == 2


def test_pcm16_resampling_changes_rate_and_preserves_endpoints() -> None:
    source = np.array([-12000, -6000, 0, 6000, 12000], dtype="<i2")

    upsampled = np.frombuffer(
        resample_pcm16(source.tobytes(), 16000, 24000),
        dtype="<i2",
    )

    assert len(upsampled) == round(len(source) * 24000 / 16000)
    assert upsampled[0] == source[0]
    assert upsampled[-1] == source[-1]


def test_pcm16_resampling_rejects_partial_sample() -> None:
    with pytest.raises(ValueError, match="even"):
        resample_pcm16(b"\x00", 16000, 24000)
