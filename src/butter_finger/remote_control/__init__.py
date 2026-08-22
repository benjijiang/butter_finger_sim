"""Stage 3 network protocol and Mac/Pi remote-control adapters.

Importing this package does not import MediaPipe, OpenCV, the Hiwonder SDK,
or a hardware backend.
"""
from __future__ import annotations

from butter_finger.remote_control.config import (
    REMOTE_TELEOPERATION_CONFIG_PATH,
    RemoteTeleoperationConfig,
    compute_profile_sha256,
    load_remote_teleoperation_config,
)
from butter_finger.remote_control.client import (
    RemoteClientStatus,
    RemoteControlClient,
    RemoteControlError,
    Stage3ControlStatus,
    Stage3Coordinator,
)
from butter_finger.remote_control.protocol import (
    NDJSONDecoder,
    PROTOCOL_VERSION,
    ProtocolError,
    decode_line,
    encode_message,
    validate_message,
)

__all__ = [
    "NDJSONDecoder",
    "PROTOCOL_VERSION",
    "ProtocolError",
    "REMOTE_TELEOPERATION_CONFIG_PATH",
    "RemoteTeleoperationConfig",
    "RemoteClientStatus",
    "RemoteControlClient",
    "RemoteControlError",
    "Stage3ControlStatus",
    "Stage3Coordinator",
    "compute_profile_sha256",
    "decode_line",
    "encode_message",
    "load_remote_teleoperation_config",
    "validate_message",
]
