"""Sub-module containing the scenario configuration."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from loguru import logger as log

from metasim.utils.configclass import configclass
from metasim.utils.hf_util import FileDownloader
from metasim.utils.setup_util import get_robot, get_scene, get_task

from .checkers import BaseChecker, EmptyChecker
from .lights import BaseLightCfg, CylinderLightCfg, DistantLightCfg
from .objects import BaseObjCfg
from .randomization import RandomizationCfg
from .render import RenderCfg
from .robots.base_robot_cfg import BaseRobotCfg
from .scenes.base_scene_cfg import SceneCfg
from .sensors import BaseCameraCfg, BaseSensorCfg, PinholeCameraCfg
from .tasks.base_task_cfg import BaseTaskCfg


@configclass
class ScenarioCfg:
    """Scenario configuration."""

    task: BaseTaskCfg | None = None  # This item should be removed?
    """None means no task specified"""
    robot: BaseRobotCfg = MISSING
    scene: SceneCfg | None = None
    """None means no scene"""
    lights: list[BaseLightCfg] = [DistantLightCfg()]
    objects: list[BaseObjCfg] = []
    cameras: list[BaseCameraCfg] = [PinholeCameraCfg()]
    sensors: list[BaseSensorCfg] = []
    checker: BaseChecker = EmptyChecker()
    render: RenderCfg = RenderCfg()
    random: RandomizationCfg = RandomizationCfg()

    ## Handlers
    sim: Literal["isaaclab", "isaacgym", "pyrep", "pybullet", "sapien", "mujoco"] = "isaaclab"
    renderer: Literal["isaaclab", "isaacgym", "pyrep", "pybullet", "sapien", "mujoco"] | None = None

    ## Others
    num_envs: int = 1
    decimation: int = 1
    episode_length: int = 10000000  # never timeout
    try_add_table: bool = True
    object_states: bool = False
    split: Literal["train", "val", "test", "all"] = "all"
    headless: bool = False

    def __post_init__(self):
        """Post-initialization configuration."""
        ### Align configurations
        if (self.random.scene or self.scene is not None) and self.try_add_table:
            log.warning("try_add_table is set to False because scene randomization is enabled or a scene is specified")
            self.try_add_table = False

        ### Randomization level configuration
        if self.random.level >= 1:
            self.lights = [
                CylinderLightCfg(pos=(-2.0, -2.0, 4.0), intensity=5e4),
                CylinderLightCfg(pos=(-2.0, 2.0, 4.0), intensity=5e4),
                CylinderLightCfg(pos=(2.0, -2.0, 4.0), intensity=5e4),
                CylinderLightCfg(pos=(2.0, 2.0, 4.0), intensity=5e4),
            ]

        ### Parse task and robot
        if isinstance(self.task, str):
            self.task = get_task(self.task)
        if isinstance(self.robot, str):
            self.robot = get_robot(self.robot)
        if isinstance(self.scene, str):
            self.scene = get_scene(self.scene)

        ### Check and download all the paths
        FileDownloader(self).do_it()
