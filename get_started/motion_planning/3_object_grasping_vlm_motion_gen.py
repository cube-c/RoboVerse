"""This script is used to grasp an object from a point cloud."""

from __future__ import annotations

"""This script is used to test the static scene."""


from typing import Literal

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import copy
import os

# import getpass
# user = getpass.getuser()  # Safe way to get current username
# os.environ["XDG_RUNTIME_DIR"] = f"/tmp/{user}-runtime"
# os.makedirs(os.environ["XDG_RUNTIME_DIR"], exist_ok=True)
import numpy as np
import rootutils
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

import open3d as o3d
import rootutils
from loguru import logger as log
from rich.logging import RichHandler

from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.state import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenPlanConfig
from metasim.sim import IdentityEnvWrapper

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])
import re

from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as R
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from get_started.motion_planning.util_gsnet import GSNet
from get_started.utils import ObsSaver, get_pcd_from_rgbd
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import SimType
from metasim.utils import configclass
from metasim.utils.camera_util import get_cam_params
from metasim.utils.demo_util import get_traj
from metasim.utils.kinematics_utils import get_curobo_models_with_pcd
from metasim.utils.setup_util import get_sim_env_class, get_task


class VLMPointExtractor:
    def __init__(self, ckpt_path="Qwen/Qwen2.5-VL-7B-Instruct"):
        self.ckpt_path = ckpt_path
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            ckpt_path, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(ckpt_path)

    def extract_points(self, img, object_name):
        prompt = f"{object_name}\nOutput its coordinates in XML format <points x y>object</points>."
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=text,
            images=img,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        point = self._extract_points_from_text(output_text, img.width, img.height)
        log.info(f"Qwen2.5-VL point: {point}")
        return point

    def _extract_points_from_text(self, text, image_w, image_h):
        all_points = []
        for match in re.finditer(r"Click\(([0-9]+\.[0-9]), ?([0-9]+\.[0-9])\)", text):
            try:
                point = [float(match.group(i)) for i in range(1, 3)]
            except ValueError:
                pass
            else:
                point = np.array(point)
                if np.max(point) > 100:
                    # Treat as an invalid output
                    continue
                point /= 100.0
                point = point * np.array([image_w, image_h])
                all_points.append(point)

        for match in re.finditer(r"\(([0-9]+\.[0-9]),? ?([0-9]+\.[0-9])\)", text):
            try:
                point = [float(match.group(i)) for i in range(1, 3)]
            except ValueError:
                pass
            else:
                point = np.array(point)
                if np.max(point) > 100:
                    # Treat as an invalid output
                    continue
                point /= 100.0
                point = point * np.array([image_w, image_h])
                all_points.append(point)
        for match in re.finditer(r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"', text):
            try:
                point = [float(match.group(i)) for i in range(1, 3)]
            except ValueError:
                pass
            else:
                point = np.array(point)
                # if np.max(point) > 100:
                #     # Treat as an invalid output
                #     continue
                # point /= 100.0
                # point = point * np.array([image_w, image_h])
                all_points.append(point)
        for match in re.finditer(r"(?:\d+|p)\s*=\s*([0-9]{3})\s*,\s*([0-9]{3})", text):
            try:
                point = [int(match.group(i)) / 10.0 for i in range(1, 3)]
            except ValueError:
                pass
            else:
                point = np.array(point)
                if np.max(point) > 100:
                    # Treat as an invalid output
                    continue
                point /= 100.0
                point = point * np.array([image_w, image_h])
                all_points.append(point)
        return all_points


class GraspPoseFinder:
    def __init__(self):
        self.gsnet = GSNet()

    def find(self, pcd: o3d.geometry.PointCloud):
        points = np.array(pcd.points)
        colors = np.array(pcd.colors)
        log.info(f"Point cloud shape: {points.shape}")
        log.info(
            f"Point cloud bounds: X[{points[:, 0].min():.3f}, {points[:, 0].max():.3f}], "
            f"Y[{points[:, 1].min():.3f}, {points[:, 1].max():.3f}], "
            f"Z[{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]"
        )

        # do not consider the arm points
        # point_mask = points[:,2] <= 0.1
        point_mask = np.ones(points.shape[0], dtype=bool)
        point_mask = np.logical_and(
            points[:, 2] <= 0.2,
            np.logical_and(points[:, 0] <= 1.0, np.logical_and(points[:, 1] >= -1.0, points[:, 1] <= 1.0)),
        )

        # point_mask = np.logical_and(points[:,2] <= 0.1 , points[:,0] >= 0.1)
        points_masked = points[point_mask]
        points_masked[:, 2] = -points_masked[:, 2]
        points_masked[:, 1] = -points_masked[:, 1]
        colors_masked = colors[point_mask]
        log.info(
            f"New Point cloud bounds: X[{points_masked[:, 0].min():.3f}, {points_masked[:, 0].max():.3f}], "
            f"Y[{points_masked[:, 1].min():.3f}, {points_masked[:, 1].max():.3f}], "
            f"Z[{points_masked[:, 2].min():.3f}, {points_masked[:, 2].max():.3f}]"
        )
        pcd.points = o3d.utility.Vector3dVector(points_masked)
        pcd.colors = o3d.utility.Vector3dVector(colors_masked)

        grasps = self.gsnet.inference(np.array(pcd.points))

        # log best grasp candidate
        log.info(f"Total grasp candidates: {len(grasps)}")
        log.info(f"Best grasp candidate: {grasps[0].translation}")
        log.info(f"Best grasp candidate rotation: {grasps[0].rotation_matrix}")

        return grasps

    def visualize(self, pcd: o3d.geometry.PointCloud, grasps, image_only=False, save_dir="", filename=""):
        pcd_clone = copy.deepcopy(pcd)
        self.gsnet.visualize(pcd_clone, grasps, image_only=image_only, save_dir=save_dir, filename=filename)


def get_environment(scenario, sim="isaaclab", robot_offset=1.15):
    """Initialize the simulation environment."""
    log.info(f"Using simulator: {sim}")
    env_class = get_sim_env_class(SimType(sim))
    env = env_class(scenario)

    robot = scenario.robots[0]
    init_states, _, _ = get_traj(task, robot, env.handler)
    init_state = init_states[0]

    log.info(
        f"robot position: {init_state['robots'][robot.name]['pos']} | rotation: {init_state['robots'][robot.name]['rot']}"
    )
    # log all objects position
    for obj_name, obj_pos in init_state["objects"].items():
        log.info(f"object {obj_name} position: {obj_pos['pos']} | rotation: {obj_pos['rot']}")

    # translate robot and all objects to make robot at the origin
    for obj_name, obj_pos in init_state["objects"].items():
        init_state["objects"][obj_name]["pos"] -= init_state["robots"][robot.name]["pos"]

    # translate robot to (robot_offset, 0.0, 0.0) and rotate 180 degrees around z axis
    init_state["robots"][robot.name]["pos"] = torch.tensor([robot_offset, 0.0, 0.0])
    init_state["robots"][robot.name]["rot"] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    # translate/rotate all objects relative to robot
    q_z_180 = torch.tensor([0.0, 1.0, 0.0, 0.0])
    q_x_180 = torch.tensor([1.0, 0.0, 0.0, 0.0])

    for obj_name, obj_pos in init_state["objects"].items():
        init_state["objects"][obj_name]["pos"][0] = robot_offset - init_state["objects"][obj_name]["pos"][0] + 0.05
        init_state["objects"][obj_name]["pos"][1] = -init_state["objects"][obj_name]["pos"][1]
        if obj_name == "ketchup" or obj_name == "salad_dressing":
            init_state["objects"][obj_name]["rot"] = torch.tensor(
                (R.from_quat(q_z_180) * R.from_quat(init_state["objects"][obj_name]["rot"])).as_quat()
            )
        if obj_name == "bbq_sauce":
            init_state["objects"][obj_name]["rot"] = torch.tensor(
                (R.from_quat(q_x_180) * R.from_quat(init_state["objects"][obj_name]["rot"])).as_quat()
            )

    log.info(
        f"[After translation] robot position: {init_state['robots'][robot.name]['pos']} | rotation: {init_state['robots'][robot.name]['rot']}"
    )
    for obj_name, obj_pos in init_state["objects"].items():
        log.info(f"[After translation] object {obj_name} position: {obj_pos['pos']} | rotation: {obj_pos['rot']}")

    robot.default_position = torch.tensor([robot_offset, 0.0, 0.0])
    robot.default_orientation = torch.tensor([0.0, 0.0, 0.0, 1.0])

    init_state["robots"][robot.name]["dof_pos"]["panda_finger_joint1"] = 0.04
    init_state["robots"][robot.name]["dof_pos"]["panda_finger_joint2"] = 0.04

    env.reset(states=[init_state])
    return env


def get_point_cloud_from_camera(img, depth, camera, output_suffix=""):
    """Get the point cloud from the observation."""
    log.info(f"img shape: {img.shape}, depth shape: {depth.shape}")
    max_depth = np.max(depth[0].cpu().numpy())
    scene_depth = depth / max_depth * 255.0  # normalize depth to [0, 1]
    scene_img = Image.fromarray(img[0].cpu().numpy())
    scene_depth = Image.fromarray((scene_depth[0].squeeze(-1).cpu().numpy() / max_depth * 255.0).astype("uint8"))
    scene_img.save(f"get_started/output/motion_planning/3_object_grasping_vlm/img{output_suffix}.png")
    scene_depth.save(f"get_started/output/motion_planning/3_object_grasping_vlm/depth{output_suffix}.png")

    extr, intr = get_cam_params(
        cam_pos=torch.tensor([camera.pos]),
        cam_look_at=torch.tensor([camera.look_at]),
        width=camera.width,
        height=camera.height,
        focal_length=camera.focal_length,
        horizontal_aperture=camera.horizontal_aperture,
    )
    pcd = get_pcd_from_rgbd(depth.cpu()[0], img.cpu()[0], intr[0], extr[0])
    pcd.estimate_normals()
    log.info(f"pcd shape: {np.array(pcd.points).shape}")
    return pcd, intr, extr


def get_point_cloud_from_obs(obs, save_pcd=False):
    # check shapes
    #######################################################################
    # merge pcds
    # before merging, we need to align the two point clouds
    # Align the second viewpoint to the first viewpoint
    # 1. Get the transformation matrix from the second viewpoint to the first viewpoint
    # 2. Apply the transformation matrix to the second viewpoint
    # 3. Merge the two point clouds
    # 4. Save the merged point cloud
    depth = obs.cameras["camera0"].depth
    pcd1, intr, extr = get_point_cloud_from_camera(
        obs.cameras["camera0"].rgb, obs.cameras["camera0"].depth, scenario.cameras[0], output_suffix="1"
    )
    pcd2, _, _ = get_point_cloud_from_camera(
        obs.cameras["camera1"].rgb, obs.cameras["camera1"].depth, scenario.cameras[1], output_suffix="2"
    )

    reg = o3d.pipelines.registration.registration_icp(
        pcd2, pcd1, 0.0025, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    # Convert to numpy arrays
    pts1 = np.asarray(pcd1.points)  # shape (N, 3)
    pts2 = np.asarray(pcd2.points)  # shape (M, 3)

    # Apply transformation to pcd2
    T = reg.transformation  # shape (4, 4)
    pts2_h = np.hstack((pts2, np.ones((pts2.shape[0], 1))))  # shape (M, 4)
    pts2_transformed = (T @ pts2_h.T).T[:, :3]

    pts1 = np.asarray(pcd1.points)
    all_points = np.vstack((pts1, pts2_transformed))

    colors1 = np.asarray(pcd1.colors)
    colors2 = np.asarray(pcd2.colors)
    all_colors = np.vstack((colors1, colors2))

    pcd_merged = o3d.geometry.PointCloud()
    pcd_merged.points = o3d.utility.Vector3dVector(all_points)
    pcd_merged.colors = o3d.utility.Vector3dVector(all_colors)
    o3d.io.write_point_cloud("get_started/output/motion_planning/3_object_grasping_vlm/pcd_merged.ply", pcd_merged)

    return pcd_merged, depth[0], intr[0], extr[0]


def find_closest_grasp(target_point, grasp_group):
    """
    Find the closest grasp candidate to the target 3D point.

    Args:
        target_point: Single 3D point (x, y, z)
        grasp_group: GraspGroup object containing grasp candidates

    Returns:
        Index of the closest grasp candidate
    """
    if len(grasp_group) == 0:
        return None

    min_distance = float("inf")
    closest_grasp_idx = 0

    for i, grasp in enumerate(grasp_group):
        grasp_position = grasp.translation

        # Calculate distance to target point
        distance = np.linalg.norm(grasp_position - target_point)
        if distance < min_distance:
            min_distance = distance
            closest_grasp_idx = i

    return closest_grasp_idx


def get_3d_point_from_pixel(pixel_point, depth, cam_intr_mat, cam_extr_mat):
    """
    Convert 2D pixel coordinates to 3D points using the point cloud.

    Since the point cloud was generated from RGBD using get_pcd_from_rgbd(),
    there's a direct mapping between pixels and 3D points.

    Args:
        pixel_points: List of 2D pixel coordinates [(x, y), ...]
        pcd_array: numpy array of point cloud
        camera_intrinsics: Camera intrinsic matrix (optional)
        camera_extrinsics: Camera extrinsic matrix (optional)

    Returns:
        List of 3D points [(x, y, z), ...]
    """
    x, y = int(pixel_point[0]), int(pixel_point[1])
    z = depth[y, x][0].item()  # depth[y, x] gives depth in meters
    log.info(f"depth: {z}")

    # 2. Unproject (x, y, z) into camera coordinates
    fx = cam_intr_mat[0, 0]
    fy = cam_intr_mat[1, 1]
    cx = cam_intr_mat[0, 2]
    cy = cam_intr_mat[1, 2]
    log.info(f"fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")

    x_cam = (x - cx) * z / fx
    y_cam = (y - cy) * z / fy
    z_cam = z
    log.info(f"x_cam: {x_cam}, y_cam: {y_cam}, z_cam: {z_cam}")

    point_cam = np.array([x_cam, y_cam, z_cam, 1.0])  # homogeneous coordinates

    # 3. Transform to world coordinates using extrinsic matrix
    cam_extr = np.array(cam_extr_mat)  # should be a 4x4 matrix
    cam_extr = np.linalg.inv(cam_extr)
    log.info(f"cam_extr: {cam_extr}")
    point_world = cam_extr @ point_cam

    # 4. Get 3D coordinate in world space
    xyz = point_world[:3]
    log.info(f"xyz: {xyz}")
    return xyz


def find_closest_point_in_pcd(pcd, query_point):
    """
    Find the closest point in the Open3D point cloud to the given 3D point.
    """
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    [_, idx, _] = pcd_tree.search_knn_vector_3d(query_point, 1)
    nearest_point = np.asarray(pcd.points)[idx[0]]
    return nearest_point


def filter_out_robot_from_pcd(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """
    Filtering out a robot region from pcd.
    """
    # Currently this is a hard-coded script, based on initial robot position which is centered at (1.15, 0.0, 1.0) and dimension is (0.3, 0.3, 2.0)
    # TODO : remove robot using segmentation
    # ex) https://github.com/NVlabs/curobo/blob/ebb71702f3f70e767f40fd8e050674af0288abe8/examples/robot_image_segmentation_example.py
    points = np.array(pcd.points)
    colors = np.array(pcd.colors)

    robot_offset = np.array([1.15, 0.0, 1.0])
    robot_dimension = np.array([0.3, 0.3, 2.0])  # x, y, z dimensions
    point_mask = np.logical_or(
        np.abs(points[:, 0] - robot_offset[0]) > robot_dimension[0] / 2,
        np.abs(points[:, 1] - robot_offset[1]) > robot_dimension[1] / 2,
        np.abs(points[:, 2] - robot_offset[2]) > robot_dimension[2] / 2,
    )
    points_filtered = points[point_mask]
    colors_filtered = colors[point_mask]
    pcd_filtered = o3d.geometry.PointCloud()
    pcd_filtered.points = o3d.utility.Vector3dVector(points_filtered)
    pcd_filtered.colors = o3d.utility.Vector3dVector(colors_filtered)
    return pcd_filtered


def gripper_pose(
    obs_saver: ObsSaver,
    env: IdentityEnvWrapper,
    open_gripper=False,
    step=20,
):
    """Get the gripper pose."""
    robot = env.handler.robots[0]
    joint_pos = env.handler.env.scene.articulations[robot.name].data.joint_pos.cuda()
    ee_n_dof = len(robot.gripper_open_q)

    log.info(f"Current robot joint state: {joint_pos}")
    joint_pos[:, -ee_n_dof:] = 0.04 if open_gripper else 0.0

    actions = [
        {"dof_pos_target": dict(zip(robot.actuators.keys(), joint_pos[i_env].tolist()))}
        for i_env in range(scenario.num_envs)
    ]
    log.info(f"Gripper actions: {actions}")
    for _ in range(step):
        obs, _, _, _, _ = env.step(actions)
        obs_saver.add(obs)


def move_to_pose(
    obs_saver: ObsSaver,
    env: IdentityEnvWrapper,
    motion_gen: MotionGen,
    plan_config: MotionGenPlanConfig,
    ee_pos_target: torch.Tensor,
    ee_quat_target: torch.Tensor,
    steps=20,
):
    """Move the robot to the target pose."""
    robot = env.handler.robots[0]
    joint_pos = env.handler.env.scene.articulations[robot.name].data.joint_pos.cuda()
    curobo_n_dof = len(motion_gen.kinematics.joint_names)

    ik_goal = Pose(position=ee_pos_target, quaternion=ee_quat_target)
    log.info(f"Joint position : {joint_pos}")
    log.info(f"Joint names: {motion_gen.kinematics.joint_names}")
    cu_js = JointState.from_position(position=joint_pos, joint_names=list(robot.actuators.keys()))
    cu_js = JointState.get_ordered_joint_state(cu_js, motion_gen.kinematics.joint_names)
    log.info(f"Current robot joint state: {cu_js}")
    result = motion_gen.plan_single(cu_js, ik_goal, plan_config)

    succ = result.success.item()
    if not succ:
        log.debug("Failed to find feasible solution")
        log.debug(f"Reason: {result}")
        return

    cmd_plan = result.get_interpolated_plan().position
    for i in range(cmd_plan.shape[0]):
        joint_pos[:, :curobo_n_dof] = cmd_plan[i : i + 1, :]
        actions = [
            {"dof_pos_target": dict(zip(robot.actuators.keys(), joint_pos[i_env]))}
            for i_env in range(scenario.num_envs)
        ]
        obs, _, _, _, _ = env.step(actions)
        obs_saver.add(obs)


@configclass
class Args:
    """Arguments for the static scene."""

    robot: str = "franka"

    ## Handlers
    sim: Literal["isaaclab", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3", "mujoco"] = "isaaclab"

    ## Others
    num_envs: int = 1
    headless: bool = False
    task_name: str = "LiberoPickChocolatePudding"
    object_name: str = "BBQ sauce"

    def __post_init__(self):
        """Post-initialization configuration."""
        log.info(f"Args: {self}")


args = tyro.cli(Args)


# initialize scenario
task_name = args.task_name
object_name = args.object_name

task = get_task(task_name)

# default
scenario = ScenarioCfg(
    task=task_name,
    cameras=[
        PinholeCameraCfg(name="camera0", width=1024, height=1024, pos=(0.0, 0.0, 0.8), look_at=(0.9, 0.0, 0.0)),
        PinholeCameraCfg(name="camera1", width=1024, height=1024, pos=(0.9, 1.0, 0.6), look_at=(0.9, 0.0, 0.0)),
    ],
    robots=[args.robot],
    try_add_table=False,
    sim=args.sim,
    headless=args.headless,
    num_envs=args.num_envs,
)
robot = scenario.robots[0]
robot_offset = 1.15

env = get_environment(scenario, sim=args.sim)

## Main loop
os.makedirs("get_started/output", exist_ok=True)
obs_saver = ObsSaver(
    video_path=f"get_started/output/motion_planning/3_object_grasping_vlm/{task_name}_{object_name}_{args.sim}.mp4"
)

grasp_finder = GraspPoseFinder()

step = 0
for step in range(1):
    log.debug(f"Step {step}")

    # Warm up the environment by taking some dummy actions with default joint positions
    dummy_action = [{"dof_pos_target": robot.default_joint_positions} for _ in range(scenario.num_envs)]
    for _ in range(120):
        obs, _, _, _, _ = env.step(dummy_action)
    obs_saver.add(obs)

    # Get point cloud from observation
    pcd, depth, cam_intr_mat, cam_extr_mat = get_point_cloud_from_obs(obs)
    pcd = filter_out_robot_from_pcd(pcd)
    motion_gen, plan_config = get_curobo_models_with_pcd(pcd, robot)

    # remove file: /home/lukesong_google_com/RoboVerse/get_started/output/motion_planning/3_object_grasping_vlm/gsnet_visualization.png
    # if os.path.exists("get_started/output/motion_planning/3_object_grasping_vlm/gsnet_visualization_new.png"):
    #   os.remove("get_started/output/motion_planning/3_object_grasping_vlm/gsnet_visualization_new.png")
    gg = grasp_finder.find(pcd)

    grasp_finder.visualize(pcd, gg, image_only=True, save_dir="3_object_grasping_vlm", filename=f"{task_name}")
    grasp_finder.visualize(
        pcd, gg[:1], image_only=True, save_dir="3_object_grasping_vlm", filename=f"gsnet_top_one_{task_name}"
    )

    # # Qwen2.5-VL
    vlm_extractor = VLMPointExtractor()
    img = Image.fromarray(obs.cameras["camera0"].rgb[0].cpu().numpy())
    point = vlm_extractor.extract_points(img, object_name)

    # save point with image
    img_with_point = img.copy()
    draw = ImageDraw.Draw(img_with_point)
    for p in point:
        # Draw a larger circle with red fill and white border for better visibility
        x, y = int(p[0]), int(p[1])
        radius = 8
        # Draw white border
        draw.ellipse(
            [x - radius - 2, y - radius - 2, x + radius + 2, y + radius + 2], fill="white", outline="black", width=2
        )
        # Draw red center
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill="red")
        # Add a small black dot in the center
        draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill="black")
    img_with_point.save(
        f"get_started/output/motion_planning/3_object_grasping_vlm/img_with_point_{task_name}_{object_name}.png"
    )

    # get 3d point of pixel (`point`) in the point cloud
    if len(point) > 0:
        point_3d = [get_3d_point_from_pixel(point[0], depth, cam_intr_mat, cam_extr_mat)]
        # point_3d = [find_closest_point_in_pcd(pcd, point_3d[0])]
        log.info(f"3d point of pixel: {point_3d}")
        # convert
        point_3d[0][1] = -point_3d[0][1]
        point_3d[0][2] = -point_3d[0][2]
        log.info(f"converted 3d point of pixel: {point_3d}")
    else:
        log.warning("No points detected by Qwen2.5-VL")
        point_3d = []

    # find the closest grasp candidate to the 3d point
    # point_3d = [] # debug
    if len(point_3d) > 0:
        closest_grasp = find_closest_grasp(point_3d[0], gg)
        log.info(f"Closest grasp candidate: {closest_grasp}")
    else:
        log.warning("No points detected by Qwen2.5-VL")
        closest_grasp = None
    # Select grasp based on VLM detection or use best score
    if len(point_3d) > 0 and closest_grasp is not None:
        selected_gg = gg[closest_grasp]
        log.info(f"Selected grasp based on VLM detection: grasp #{closest_grasp}")
        grasp_finder.visualize(
            pcd,
            gg[closest_grasp : closest_grasp + 1],
            image_only=True,
            save_dir="3_object_grasping_vlm",
            filename=f"qwen2.5vl_top_one_{task_name}_{object_name}",
        )

    else:
        selected_gg = gg[0]  # Use best scoring grasp
        log.info("Using best scoring grasp (no VLM detection)")

    # Debug: Print original grasp pose from GSNet
    # selected_gg = gg[0]
    log.info(f"Original GSNet grasp position: {selected_gg.translation}")
    log.info(f"Original GSNet grasp rotation:\n{selected_gg.rotation_matrix}")

    position = selected_gg.translation.copy()
    # robot translation

    position[2] = -position[2]
    position[1] = -position[1]

    position[0] = robot_offset - position[0]
    position[1] = -position[1]

    # Add robot position offset since robot is now at (robot_offset, 0, 0)
    log.info(f"After coordinate flip + robot offset - position: {position}")

    rotation_ori = selected_gg.rotation_matrix
    # Since robot is rotated 180° around Z-axis, we need to account for this
    # Robot quaternion (0,0,0,1) means 180° rotation around Z
    # R_180 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    R_180 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    R_world = R_180 @ rotation_ori @ R_180.T
    log.info(f"After coordinate transformation - rotation:\n{R_world}")

    gripper_out = torch.tensor(R_world[:, 0])

    gripper_short = torch.tensor(R_world[:, 2])
    gripper_out = gripper_out / np.linalg.norm(gripper_out)
    gripper_short = gripper_short / np.linalg.norm(gripper_short)

    # 计算 y = z × x
    gripper_long = np.cross(gripper_short, gripper_out)
    gripper_long = gripper_long / np.linalg.norm(gripper_long)

    # 重新计算 z，使得 x、y 完全正交（防止数值误差）
    gripper_long = np.cross(gripper_out, gripper_short)
    gripper_long = gripper_long / np.linalg.norm(gripper_long)
    gripper_long = torch.tensor(gripper_long)
    # rotation = np.dot(rotation, delta_m)

    # breakpoint()
    rotation_transform_for_franka = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
    )
    # import pdb; pdb.set_trace()
    rotation_target = torch.stack(
        [
            gripper_out + 1e-4,
            gripper_long + 1e-4,
            gripper_short + 1e-4,
        ],
        dim=0,
    ).float()
    rotation = rotation_target @ rotation_transform_for_franka

    quat = R.from_matrix(rotation).as_quat()
    quat *= np.array([1.0, -1.0, -1.0, 1.0])

    ee_pos_target = torch.zeros((args.num_envs, 3), device="cuda:0")
    ee_quat_target = torch.zeros((args.num_envs, 4), device="cuda:0")

    # Debug: Print final robot target pose
    log.info(f"Final robot target position: {position}")
    log.info(f"Final robot target quaternion: {quat}")

    # position[2] += 0.1
    ee_pos_target[0] = torch.tensor(position, device="cuda:0")
    ee_quat_target[0] = torch.tensor(quat, device="cuda:0")

    pre_grasp_pos = ee_pos_target.clone()
    grasp_pos = ee_pos_target.clone()
    lift_pos = ee_pos_target.clone()
    # breakpoint()
    gripper_out = gripper_out.to("cuda:0")

    # grasp_pos[:] -= gripper_out * 0.02
    # grasp_pos[:, 2] += 0.01
    # lift_pos[:] -= gripper_out * 0.02
    log.info(f"original pre_grasp_pos: {pre_grasp_pos}")
    # pre_grasp_pos[:, :2] -= gripper_out[:2] * 0.1

    # pre_grasp_pos[:, :2] -= gripper_out[:2] * 0.075
    pre_grasp_pos[:] -= gripper_out * 0.07
    # grasp_pos[:, :2] -= gripper_out[:2] * 0.07
    # lift_pos[:, :2] -= gripper_out[:2] * 0.0
    grasp_pos[:] -= gripper_out * 0.07
    lift_pos[:] -= gripper_out * 0.07

    # TODO: set hyperparameter for heuristic grasp posing
    pre_grasp_pos[:, 2] += 0.20
    lift_pos[:, 2] += 0.2
    # grasp_pos[:, 2] += 0.1
    log.info(f"gripper_out: {gripper_out}")
    log.info(f"pre_grasp_pos: {pre_grasp_pos}")
    log.info(f"grasp_pos: {grasp_pos}")

    # TODO: line 965-971
    gripper_pose(obs_saver, env, open_gripper=True, step=20)
    move_to_pose(obs_saver, env, motion_gen, plan_config, pre_grasp_pos, ee_quat_target, steps=50)
    move_to_pose(obs_saver, env, motion_gen, plan_config, grasp_pos, ee_quat_target, steps=50)
    gripper_pose(obs_saver, env, open_gripper=False, step=40)
    move_to_pose(obs_saver, env, motion_gen, plan_config, lift_pos, ee_quat_target, steps=50)

    step += 1

obs_saver.save()
