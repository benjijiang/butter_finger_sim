"""Backend-neutral scheduling for validated, timed arm actions.

The bundled actions currently contain development-only simulation radians.
The runner depends only on ArmBackend so a future calibrated RaspberryPiArm
can use the same scheduling path, but these actions must not be selected for
real hardware until their targets have been measured and approved.
"""
from __future__ import annotations

from butter_finger.arm import ArmBackend
from butter_finger.config import ActionConfig, ArmAction, load_action_config


class UnknownActionError(ValueError):
    """An action name is not present in the loaded action configuration."""


class ActionRunner:
    """Run configured action steps sequentially through an ArmBackend."""

    def __init__(
        self,
        arm: ArmBackend,
        config: ActionConfig | None = None,
    ) -> None:
        self.arm = arm
        self.config = config if config is not None else load_action_config()

    @property
    def action_names(self) -> tuple[str, ...]:
        return self.config.action_names

    def get_action(self, name: str) -> ArmAction:
        try:
            return self.config.actions[name]
        except KeyError as exc:
            raise UnknownActionError(
                f"Unknown action {name!r}; expected one of {list(self.action_names)}"
            ) from exc

    def run(self, name: str) -> None:
        """Run every step in ``name`` synchronously, in configured order."""
        action = self.get_action(name)
        for step in action.steps:
            self.arm.move_joints(
                step.targets_rad,
                duration_s=step.duration_s,
            )
