"""This script is used to grasp an object from a point cloud."""

from __future__ import annotations

"""This script is used to test the static scene."""


from typing import Literal

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import os
import copy
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
from curobo.types.math import Pose
from loguru import logger as log
from rich.logging import RichHandler

from get_started.utils import convert_to_ply

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])
from scipy.spatial.transform import Rotation as R

from get_started.motion_planning.util_gsnet import GSNet
from get_started.utils import ObsSaver, get_pcd_from_rgbd
from metasim.cfg.objects import RigidObjCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import PhysicStateType, SimType
from metasim.utils import configclass
from metasim.utils.camera_util import get_cam_params
from metasim.utils.kinematics_utils import get_curobo_models
from metasim.utils.setup_util import get_sim_env_class, get_robot, get_task

from metasim.utils.demo_util import get_traj

from PIL import Image, ImageDraw
from transformers import Qwen2_5_VLForConditionalGeneration, AutoModelForCausalLM, AutoProcessor, GenerationConfig
from typing import Optional
import re
from xml.etree import ElementTree


ckpt_path = "Qwen/Qwen2.5-VL-7B-Instruct" # pretrained model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(ckpt_path, torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(ckpt_path)

def extract_points(text, image_w, image_h):
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
    for match in re.finditer(r'(?:\d+|p)\s*=\s*([0-9]{3})\s*,\s*([0-9]{3})', text):
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

def parse_point(pred: str, image_size: Optional[tuple[int, int]] = None):
    """
    Args:
        pred: The prediction string from the model.
        image_size: The size of the image, (width, height). If provided, return in pixels, otherwise return in normalized coordinates.
    Returns:
        The predicted point as a numpy array of shape (2,).
    """
    point_xmls = re.findall(r'<point?.*?</point?>', pred, re.DOTALL)
    if len(point_xmls) == 0:
        print(f"Invalid prediction: {pred}")
        return None
    point_xml = point_xmls[0]
    try:
        point_elem = ElementTree.fromstring(point_xml)

        if point_elem is not None:
            if point_elem.tag == 'point':
                x = float(point_elem.get('x'))
                y = float(point_elem.get('y'))
            elif point_elem.tag == 'points':
                x = float(point_elem.get('x1'))
                y = float(point_elem.get('y1'))
            else:
                print(f"Invalid prediction: {pred}")
                return None
            ret = np.array([x, y])
            if image_size is not None:
                ret = ret / 100 * np.array(image_size)
            return ret
        else:
            print("No point element found in XML")
    except ElementTree.ParseError as e:
        print(f"Failed to parse XML: {e}")
    return None


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
prompt = f"{object_name}\nOutput its coordinates in XML format <points x y>object</points>."

task = get_task(task_name)
robot = get_robot(args.robot)

# actual default
# camera = PinholeCameraCfg(width=1024, height=1024, pos=(0.9, -0.3, 0.9), look_at=(0.0, 0.0, 0.0))



robot_offset = 1.15

# default
camera = PinholeCameraCfg(width=1024, height=1024, pos=(0.5, -0.5, 0.5), look_at=(0.0, 0.0, 0.0))

# robot is at bottom
# camera = PinholeCameraCfg(width=1024, height=1024, pos=(-0.8, 0.0, 0.8), look_at=(0.4, 0.0, 0.0))

# robot is at upper center
# camera = PinholeCameraCfg(width=1024, height=1024, pos=(1.5, 0.0, 1.5), look_at=(0.0, 0.0, 0.0))

# this code -- default
# camera = PinholeCameraCfg(width=1024, height=1024, pos=(0.0, 0.0, 1.0), look_at=(1.0, 0.0, 0.0))
camera = PinholeCameraCfg(width=1024, height=1024, pos=(0.0, 0.0, 0.8), look_at=(0.9, 0.0, 0.0))

camera2 = PinholeCameraCfg(width=1024, height=1024, pos=(0.9, 1.0, 0.6), look_at=(0.9, 0.0, 0.0))
log.info(f"Camera position: {camera.pos}")
log.info(f"Camera look_at: {camera.look_at}")

log.info(f"Camera2 position: {camera2.pos}")
log.info(f"Camera2 look_at: {camera2.look_at}")

scenario = ScenarioCfg(
    task=task_name,
    cameras=[camera],
    robots=[args.robot],
    try_add_table=False,
    sim=args.sim,
    headless=args.headless,
    num_envs=args.num_envs,
)

scenario2 = ScenarioCfg(
    task=task_name,
    cameras=[camera2],
    robots=[args.robot],
    try_add_table=False,
    sim=args.sim,
    headless=args.headless,
    num_envs=args.num_envs,
)

# add cameras
# scenario.cameras = [
#     # PinholeCameraCfg(width=1024, height=1024, pos=(0.0, 0.0, 1.5), look_at=(1.0, 0.0, 0.0)),
#     # PinholeCameraCfg(width=1024, height=1024, pos=(-0.6106, 0.0051, 1.5), look_at=(0.6106, 0.0051, 0.0)),
#     # PinholeCameraCfg(width=1024, height=1024, pos=(1.5, 0.0, 1.5), look_at=(0.0, 0.0, 0.0)),

# ]

# add objects
# scenario.objects = [
#     RigidObjCfg(
#         name="bbq_sauce",
#         scale=(1.5, 1.5, 1.5),
#         physics=PhysicStateType.RIGIDBODY,
#         usd_path="get_started/example_assets/bbq_sauce/usd/bbq_sauce.usd",
#         urdf_path="get_started/example_assets/bbq_sauce/urdf/bbq_sauce.urdf",
#         mjcf_path="get_started/example_assets/bbq_sauce/mjcf/bbq_sauce.xml",
#     ),
# ]


log.info(f"Using simulator: {args.sim}")
env_class = get_sim_env_class(SimType(args.sim))
env = env_class(scenario)

# init_states = [
#     {
#         "objects": {
#             "bbq_sauce": {
#                 "pos": torch.tensor([0.7, -0.2, 0.07]),
#                 "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
#             },
#         },
#         "robots": {
#             "franka": {
#                 "pos": torch.tensor([0.0, 0.0, 0.0]),
#                 "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
#                 "dof_pos": {
#                     "panda_joint1": 0.0,
#                     "panda_joint2": -0.785398,
#                     "panda_joint3": 0.0,
#                     "panda_joint4": -2.356194,
#                     "panda_joint5": 0.0,
#                     "panda_joint6": 1.570796,
#                     "panda_joint7": 0.785398,
#                     "panda_finger_joint1": 0.04,
#                     "panda_finger_joint2": 0.04,
#                 },
#             },
#         },
#     }
#     for _ in range(args.num_envs)
# ]

init_states, all_actions, all_states = get_traj(task, robot, env.handler)
num_demos = len(init_states)

log.info(f"Number of demos: {num_demos}")
log.info(f"robot position: {init_states[0]['robots'][robot.name]['pos']} | rotation: {init_states[0]['robots'][robot.name]['rot']}")
# log all objects position
for obj_name, obj_pos in init_states[0]['objects'].items():
    log.info(f"object {obj_name} position: {obj_pos['pos']} | rotation: {obj_pos['rot']}")
# log.info(f"")

# translate robot and all objects to make robot at the origin
robot_pos = init_states[0]['robots'][robot.name]['pos']
for obj_name, obj_pos in init_states[0]['objects'].items():
    init_states[0]['objects'][obj_name]['pos'] -= robot_pos
init_states[0]['robots'][robot.name]['pos'] = torch.tensor([0.0, 0.0, 0.0])

# translate robot to (3.0, 0.0, 0.0) and rotate 180 degrees around z axis
init_states[0]['robots'][robot.name]['pos'] = torch.tensor([robot_offset, 0.0, 0.0])
init_states[0]['robots'][robot.name]['rot'] = torch.tensor([0.0, 0.0, 0.0, 1.0])
# translate/rotate all objects relative to robot
q_z_180 = torch.tensor([0.0, 1.0, 0.0, 0.0])
q_x_180 = torch.tensor([1.0, 0.0, 0.0, 0.0])
# Convert existing quaternion to scipy Rotation
# r_existing = R.from_quat(q_existing)
for obj_name, obj_pos in init_states[0]['objects'].items():
    init_states[0]['objects'][obj_name]['pos'][0] = robot_offset - init_states[0]['objects'][obj_name]['pos'][0] + 0.05
    init_states[0]['objects'][obj_name]['pos'][1] = -init_states[0]['objects'][obj_name]['pos'][1]
    # print((R.from_quat(q_z_180) * R.from_quat(init_states[0]['objects'][obj_name]['rot'])).as_quat())
    if obj_name == "ketchup" or obj_name == "salad_dressing":
        init_states[0]['objects'][obj_name]['rot'] = torch.tensor((R.from_quat(q_z_180) * R.from_quat(init_states[0]['objects'][obj_name]['rot'])).as_quat())
    if obj_name == "bbq_sauce":
        init_states[0]['objects'][obj_name]['rot'] = torch.tensor((R.from_quat(q_x_180) * R.from_quat(init_states[0]['objects'][obj_name]['rot'])).as_quat())
    # init_states[0]['objects'][obj_name]['rot'][-1] = -init_states[0]['objects'][obj_name]['rot'][-1]

log.info(f"[After translation] robot position: {init_states[0]['robots'][robot.name]['pos']} | rotation: {init_states[0]['robots'][robot.name]['rot']}")
for obj_name, obj_pos in init_states[0]['objects'].items():
    log.info(f"[After translation] object {obj_name} position: {obj_pos['pos']} | rotation: {obj_pos['rot']}")


robot = scenario.robots[0]
robot.default_position = torch.tensor([robot_offset, 0.0, 0.0])
robot.default_orientation = torch.tensor([0.0, 0.0, 0.0, 1.0])
print(robot)
*_, robot_ik = get_curobo_models(robot, no_gnd=True)
curobo_n_dof = len(robot_ik.robot_config.cspace.joint_names)
ee_n_dof = len(robot.gripper_open_q)

obs, extras = env.reset(states=init_states[0:1])

dummy_action = [
        {"dof_pos_target": dict(zip(robot.actuators.keys(), [0.0] * len(robot.actuators.keys())))} for _ in range(scenario.num_envs)
    ]
for _ in range(120):
    obs, _, _, _, _ = env.step(dummy_action)

############ TEMPORARY ############
# just save second camera viewpoint
# img2 = obs.cameras["camera0"].rgb
# depth2 = obs.cameras["camera0"].depth
# img2_path = "get_started/output/motion_planning/3_object_grasping_vlm/img2.png"
# depth2_path = "get_started/output/motion_planning/3_object_grasping_vlm/depth2.png"
# # save depth2 array as npy
# np.save("get_started/output/motion_planning/3_object_grasping_vlm/depth2.npy", depth2[0].cpu().numpy())
# # save img2 array as npy
# np.save("get_started/output/motion_planning/3_object_grasping_vlm/img2.npy", img2[0].cpu().numpy())

# max_depth2 = np.max(depth2[0].cpu().numpy())
# scene_depth2 = depth2 / max_depth2 * 255.0 # normalize depth to [0, 1]
# scene_img2 = Image.fromarray(img2[0].cpu().numpy())
# scene_depth2 = Image.fromarray((scene_depth2[0].squeeze(-1).cpu().numpy() / max_depth2 * 255.0).astype('uint8'))
# scene_img2.save(img2_path)
# scene_depth2.save(depth2_path)
# exit()
####################################

# sim_steps = int(2.0 / env.sim_dt)  # env.sim_dt is the simulation time step
# dummy_action = [
#         {"dof_pos_target": dict(zip(robot.actuators.keys(), [0.0] * len(robot.actuators.keys())))} for _ in range(scenario.num_envs)
#     ]
# for _ in range(120):
#     obs, _, _, _, _ = env.step(dummy_action)
# no_robot_init_states = copy.deepcopy(init_states[0])
# no_robot_init_states["robots"][robot.name]["pos"][0] = 1e5
# no_robot_obs, _ = env.reset(states=[no_robot_init_states])
os.makedirs("get_started/output", exist_ok=True)


## Main loop
obs_saver = ObsSaver(video_path=f"get_started/output/motion_planning/3_object_grasping_vlm/{task_name}_{object_name}_{args.sim}.mp4")
obs_saver.add(obs)


def get_point_cloud_from_obs(obs, save_pcd=False):
    """Get the point cloud from the observation."""
    img = obs.cameras["camera0"].rgb
    depth = obs.cameras["camera0"].depth
    # load img2 and depth2 to tensor
    img2 = torch.from_numpy(np.load("get_started/output/motion_planning/3_object_grasping_vlm/img2.npy")).unsqueeze(0)
    depth2 = torch.from_numpy(np.load("get_started/output/motion_planning/3_object_grasping_vlm/depth2.npy")).unsqueeze(0)
    # check shapes
    log.info(f"img shape: {img.shape}, depth shape: {depth.shape}")
    log.info(f"img2 shape: {img2.shape}, depth2 shape: {depth2.shape}")
    ########################################################################
    # save img and depth as png
    # os.makedirs("get_started/output/motion_planning/3_object_grasping_vlm", exist_ok=True)
    # print(f"img shape: {img.shape}, depth shape: {depth.shape}")
    img_path = "get_started/output/motion_planning/3_object_grasping_vlm/img.png"
    depth_path = "get_started/output/motion_planning/3_object_grasping_vlm/depth.png"
    max_depth = np.max(depth[0].cpu().numpy())
    scene_depth = depth / max_depth * 255.0 # normalize depth to [0, 1]
    scene_img = Image.fromarray(img[0].cpu().numpy())
    scene_depth = Image.fromarray((scene_depth[0].squeeze(-1).cpu().numpy() / max_depth * 255.0).astype('uint8'))
    scene_img.save(img_path)
    scene_depth.save(depth_path)
    img2_path = "get_started/output/motion_planning/3_object_grasping_vlm/img2.png"
    depth2_path = "get_started/output/motion_planning/3_object_grasping_vlm/depth2.png"
    max_depth2 = np.max(depth2[0].cpu().numpy())
    scene_depth2 = depth2 / max_depth2 * 255.0 # normalize depth to [0, 1]
    scene_img2 = Image.fromarray(img2[0].cpu().numpy())
    scene_depth2 = Image.fromarray((scene_depth2[0].squeeze(-1).cpu().numpy() / max_depth2 * 255.0).astype('uint8'))
    scene_img2.save(img2_path)
    scene_depth2.save(depth2_path)
    #######################################################################
    extr, intr = get_cam_params(
        cam_pos=torch.tensor([scenario.cameras[i].pos for i in range(len(scenario.cameras))]),
        cam_look_at=torch.tensor([scenario.cameras[i].look_at for i in range(len(scenario.cameras))]),
        width=scenario.cameras[0].width,
        height=scenario.cameras[0].height,
        focal_length=scenario.cameras[0].focal_length,
        horizontal_aperture=scenario.cameras[0].horizontal_aperture,
    )

    extr2, intr2 = get_cam_params(
        cam_pos=torch.tensor([scenario2.cameras[i].pos for i in range(len(scenario2.cameras))]),
        cam_look_at=torch.tensor([scenario2.cameras[i].look_at for i in range(len(scenario2.cameras))]),
        width=scenario2.cameras[0].width,
        height=scenario2.cameras[0].height,
        focal_length=scenario2.cameras[0].focal_length,
        horizontal_aperture=scenario2.cameras[0].horizontal_aperture,
    )

    pcd = get_pcd_from_rgbd(depth.cpu()[0], img.cpu()[0], intr[0], extr[0])
    pcd2 = get_pcd_from_rgbd(depth2.cpu()[0], img2.cpu()[0], intr2[0], extr2[0])

    # Estimate normals for both point clouds
    pcd.estimate_normals()
    pcd2.estimate_normals()
    # info pcd and pcd2 shape
    log.info(f"pcd shape: {np.array(pcd.points).shape}, pcd2 shape: {np.array(pcd2.points).shape}")

    # merge pcds
    # before merging, we need to align the two point clouds
    # Align the second viewpoint to the first viewpoint
    # 1. Get the transformation matrix from the second viewpoint to the first viewpoint
    # 2. Apply the transformation matrix to the second viewpoint
    # 3. Merge the two point clouds
    # 4. Save the merged point cloud
    reg = o3d.pipelines.registration.registration_icp(pcd2, pcd, 0.0025, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPlane())


    # Convert to numpy arrays
    pts1 = np.asarray(pcd.points)         # shape (N, 3)
    pts2 = np.asarray(pcd2.points)        # shape (M, 3)

    # Apply transformation to pcd2
    T = reg.transformation                # shape (4, 4)
    pts2_h = np.hstack((pts2, np.ones((pts2.shape[0], 1))))  # shape (M, 4)
    pts2_transformed = (T @ pts2_h.T).T[:, :3]

    pts1 = np.asarray(pcd.points)
    all_points = np.vstack((pts1, pts2_transformed))

    colors1 = np.asarray(pcd.colors)
    colors2 = np.asarray(pcd2.colors)
    all_colors = np.vstack((colors1, colors2))

    pcd_merged = o3d.geometry.PointCloud()
    pcd_merged.points = o3d.utility.Vector3dVector(all_points)
    pcd_merged.colors = o3d.utility.Vector3dVector(all_colors)

    # pcd_merged = o3d.geometry.PointCloud()
    # pcd_merged.points = o3d.utility.Vector3dVector(np.concatenate([np.array(pcd.points), np.array(reg.transformation @ pcd2.points)], axis=0))
    # pcd_merged.colors = o3d.utility.Vector3dVector(np.concatenate([np.array(pcd.colors), np.array(pcd2.colors)], axis=0))
    o3d.io.write_point_cloud("get_started/output/motion_planning/3_object_grasping_vlm/pcd_merged.ply", pcd_merged)

    # print(pcd)
    # if save_pcd:
    #     convert_to_ply(np.array(pcd.points), "get_started/output/motion_planning/3_object_grasping_vlm.ply")
    # return pcd, depth[0], intr[0], extr[0]
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

    min_distance = float('inf')
    closest_grasp_idx = 0

    for i, grasp in enumerate(grasp_group):
        grasp_position = grasp.translation

        # Calculate distance to target point
        distance = np.linalg.norm(grasp_position - target_point)
        if distance < min_distance:
            min_distance = distance
            closest_grasp_idx = i

    return closest_grasp_idx


def get_3d_point_from_pixel_v2(pixel_point, depth, cam_intr_mat, cam_extr_mat):
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

def get_3d_point_from_pixel(pixel_points, pcd_array, camera_intrinsics=None, camera_extrinsics=None):
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
    points_3d = []

    # Get image dimensions from camera config
    img_width, img_height = 1024, 1024  # Based on camera config

    # Since get_pcd_from_rgbd() creates a point cloud where each pixel maps to a 3D point,
    # we can directly index into the point cloud using pixel coordinates
    # The point cloud is organized as a flattened image: (height * width, 3)

    for pixel_point in pixel_points:
        x, y = int(pixel_point[0]), int(pixel_point[1])

        # Ensure pixel coordinates are within bounds
        if x < 0 or x >= img_width or y < 0 or y >= img_height:
            log.warning(f"Pixel coordinates ({x}, {y}) out of bounds ({img_width}x{img_height})")
            points_3d.append(np.array([0, 0, 0]))
            continue

        # Calculate the index in the flattened point cloud
        # Open3D creates point cloud in row-major order: (y * width + x)
        point_idx = y * img_width + img_width - x

        # Check if the index is valid
        if point_idx < len(pcd_array):
            point_3d = pcd_array[point_idx]

            # Check if the point is valid (not NaN, not behind camera)
            if not np.any(np.isnan(point_3d)) and not np.any(np.isinf(point_3d)) and point_3d[2] > 0:
                points_3d.append(point_3d)
            else:
                # Point is invalid, find the nearest valid point
                log.warning(f"Invalid 3D point at pixel ({x}, {y}), finding nearest valid point")

                # Find valid points in a small region around this pixel
                search_radius = 10  # pixels
                valid_candidates = []

                for dy in range(-search_radius, search_radius + 1):
                    for dx in range(-search_radius, search_radius + 1):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < img_width and 0 <= ny < img_height:
                            n_idx = ny * img_width + nx
                            if n_idx < len(pcd_array):
                                n_point = pcd_array[n_idx]
                                if not np.any(np.isnan(n_point)) and not np.any(np.isinf(n_point)) and n_point[2] > 0:
                                    # Calculate distance from target pixel
                                    pixel_dist = np.sqrt(dx*dx + dy*dy)
                                    valid_candidates.append((n_point, pixel_dist))

                if valid_candidates:
                    # Choose the closest valid point
                    best_point, _ = min(valid_candidates, key=lambda x: x[1])
                    points_3d.append(best_point)
                else:
                    # Fallback: use any valid point from the point cloud
                    valid_points = pcd_array[(pcd_array[:, 2] > 0) &
                                            ~np.any(np.isnan(pcd_array), axis=1) &
                                            ~np.any(np.isinf(pcd_array), axis=1)]
                    if len(valid_points) > 0:
                        points_3d.append(valid_points[0])
                    else:
                        points_3d.append(np.array([0, 0, 0]))
        else:
            log.warning(f"Point index {point_idx} out of range for point cloud size {len(pcd_array)}")
            points_3d.append(np.array([0, 0, 0]))

    return points_3d

def quat_inverse(q):
    return torch.cat([-q[:, :3], q[:, 3:]], dim=-1)

def quat_rotate(q, v):
    """Rotate vector v by quaternion q"""
    qvec = q[:, :3]
    uv = torch.cross(qvec, v, dim=-1)
    uuv = torch.cross(qvec, uv, dim=-1)
    return v + 2 * (q[:, 3:]*uv + uuv)

def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    return torch.stack([x, y, z, w], dim=-1)

def find_closest_point_in_pcd(pcd, query_point):
    """
    Find the closest point in the Open3D point cloud to the given 3D point.
    """
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    [_, idx, _] = pcd_tree.search_knn_vector_3d(query_point, 1)
    nearest_point = np.asarray(pcd.points)[idx[0]]
    return nearest_point

def move_to_pose(
    obs, obs_saver, robot_ik, robot, scenario, ee_pos_target, ee_quat_target, steps=10, open_gripper=False
):
    """Move the robot to the target pose."""
    curr_robot_q = obs.robots[robot.name].joint_pos

    seed_config = curr_robot_q[:, :curobo_n_dof].unsqueeze(1).tile([1, robot_ik._num_seeds, 1])

    result = robot_ik.solve_batch(Pose(ee_pos_target, ee_quat_target), seed_config=seed_config)

    q = torch.zeros((scenario.num_envs, robot.num_joints), device="cuda:0")
    ik_succ = result.success.squeeze(1)
    q[ik_succ, :curobo_n_dof] = result.solution[ik_succ, 0].clone()
    q[:, -ee_n_dof:] = 0.04 if open_gripper else 0.0
    actions = [
        {"dof_pos_target": dict(zip(robot.actuators.keys(), q[i_env].tolist()))} for i_env in range(scenario.num_envs)
    ]
    dummy_action = [
            {"dof_pos_target": dict(zip(robot.actuators.keys(), [0.0] * len(robot.actuators.keys())))} for _ in range(scenario.num_envs)
    ]
    # print(actions)
    for i in range(steps):
        obs, reward, success, time_out, extras = env.step(actions)
        # obs, _, _, _, _ = env.step(dummy_action)
        obs_saver.add(obs)
    return obs


step = 0
robot_joint_limits = scenario.robots[0].joint_limits
for step in range(1):
    log.debug(f"Step {step}")
    dummy_action = [
            {"dof_pos_target": dict(zip(robot.actuators.keys(), [0.0] * len(robot.actuators.keys())))} for _ in range(scenario.num_envs)
        ]
    for _ in range(120):
        obs, _, _, _, _ = env.step(dummy_action)
    obs_saver.add(obs)
    states = env.handler.get_states()
    curr_robot_q = states.robots[robot.name].joint_pos.cuda()

    seed_config = curr_robot_q[:, :curobo_n_dof].unsqueeze(1).tile([1, robot_ik._num_seeds, 1])

    pcd, depth, cam_intr_mat, cam_extr_mat = get_point_cloud_from_obs(obs)

    # Method 5: Print point cloud statistics
    points_array = np.array(pcd.points)
    colors_array = np.array(pcd.colors) if pcd.has_colors() else None
    log.info(f"Point cloud shape: {points_array.shape}")
    log.info(f"Point cloud bounds: X[{points_array[:, 0].min():.3f}, {points_array[:, 0].max():.3f}], "
             f"Y[{points_array[:, 1].min():.3f}, {points_array[:, 1].max():.3f}], "
             f"Z[{points_array[:, 2].min():.3f}, {points_array[:, 2].max():.3f}]")

    points = np.array(pcd.points)
    # print("points shape:", points.shape)
    colors = np.array(pcd.colors)
    # do not consider the arm points
    # point_mask= points[:,2] <= 0.1
    point_mask = np.ones(points.shape[0], dtype=bool)
    point_mask = np.logical_and(
        points[:, 2] <= 0.2,
        np.logical_and(points[:, 0] <= 1.0, np.logical_and(points[:, 1] >= -1.0, points[:, 1] <= 1.0)),
    )
    # point_mask = np.logical_and(points[:,2] <= 0.1 , points[:,0] >= 0.1)
    points_masked = points[point_mask]
    colors_masked = colors[point_mask]
    points_masked[:, 2] = -points_masked[:, 2]
    points_masked[:, 1] = -points_masked[:, 1]
    pcd.points = o3d.utility.Vector3dVector(points_masked)
    pcd.colors = o3d.utility.Vector3dVector(colors_masked)
    log.info(f"New Point cloud bounds: X[{points_masked[:, 0].min():.3f}, {points_masked[:, 0].max():.3f}], "
             f"Y[{points_masked[:, 1].min():.3f}, {points_masked[:, 1].max():.3f}], "
             f"Z[{points_masked[:, 2].min():.3f}, {points_masked[:, 2].max():.3f}]")
    gsnet = GSNet()
    gg = gsnet.inference(np.array(pcd.points))
    log.info(f"Total grasp candidates: {len(gg)}")
    # log best grasp candidate
    log.info(f"Best grasp candidate: {gg[0].translation}")
    log.info(f"Best grasp candidate rotation: {gg[0].rotation_matrix}")
    # gsnet.visualize(pcd, gg)

    # remove file: /home/lukesong_google_com/RoboVerse/get_started/output/motion_planning/3_object_grasping_vlm/gsnet_visualization.png
    if os.path.exists("get_started/output/motion_planning/3_object_grasping_vlm/gsnet_visualization_new.png"):
        os.remove("get_started/output/motion_planning/3_object_grasping_vlm/gsnet_visualization_new.png")

    pcd_clone1 = copy.deepcopy(pcd)
    gsnet.visualize(pcd_clone1, gg, image_only=True, save_dir="3_object_grasping_vlm", filename=f"{task_name}")
    # filter gg
    # # import pdb; pdb.set_trace()
    # mask = [0.0 <= -gg_i.translation[2] <= 0.1 for gg_i in gg]
    # gg = gg[mask]
    # print(gg[:4])

    # print(gg[-4:])
    pcd_clone2 = copy.deepcopy(pcd)
    gsnet.visualize(pcd_clone2, gg[:1], image_only=True, save_dir="3_object_grasping_vlm", filename=f"gsnet_top_one_{task_name}")


    # # Qwen2.5-VL
    log.info(f"VLM prompt: {prompt}")
    img = obs.cameras["camera0"].rgb
    img = img[0].cpu().numpy()
    img = Image.fromarray(img)
    log.info(f"Image shape: {img.size}")

    system_content = "You are a helpful assistant."
    messages = [
            {"role": "system", "content": system_content},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image":img},
                    {"type": "text", "text": prompt},
                ],
            }
    ]
    text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
            text=text,
            images=img,
            return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
       ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    log.info(f"Qwen2.5-VL output: {output_text}")
    # point = parse_point(output_text, (img.width, img.height))
    point = extract_points(output_text, img.width, img.height)
    log.info(f"Qwen2.5-VL point: {point}")

    # save point with image
    img_with_point = img.copy()
    draw = ImageDraw.Draw(img_with_point)
    for p in point:
        # Draw a larger circle with red fill and white border for better visibility
        x, y = int(p[0]), int(p[1])
        radius = 8
        # Draw white border
        draw.ellipse([x-radius-2, y-radius-2, x+radius+2, y+radius+2], fill="white", outline="black", width=2)
        # Draw red center
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill="red")
        # Add a small black dot in the center
        draw.ellipse([x-2, y-2, x+2, y+2], fill="black")
    img_with_point.save(f"get_started/output/motion_planning/3_object_grasping_vlm/img_with_point_{task_name}_{object_name}.png")

    # get 3d point of pixel (`point`) in the point cloud
    if len(point) > 0:
        # point_3d = get_3d_point_from_pixel(point, points_array)
        point_3d = [get_3d_point_from_pixel_v2(point[0], depth, cam_intr_mat, cam_extr_mat)]
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
        pcd_clone3 = copy.deepcopy(pcd)
        gsnet.visualize(pcd_clone3, gg[closest_grasp:closest_grasp+1], image_only=True, save_dir="3_object_grasping_vlm", filename=f"qwen2.5vl_top_one_{task_name}_{object_name}")

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

    # position[0] = robot_offset-position[0]

    position[0] = robot_offset-position[0]
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


    # rot = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]).float()
    # calib_rotation = rotation @ rot

    quat = R.from_matrix(rotation).as_quat()
    quat *= np.array([1., -1., -1., 1.])

    # q_90 = R.from_euler('z', 90, degrees=True).as_quat()

    # q = [0, 1, 0, 0]
    # r = R.from_quat(q)
    # quat = (r * R.from_quat(quat)).as_quat()


    # quat = (R.from_quat([0, 0, 1, 0]) * R.from_quat(quat)).as_quat()

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
    # grasp_pos[:, 2] += 0.01
    # grasp_pos[:, 0] -= 0.01
    log.info(f"gripper_out: {gripper_out}")
    log.info(f"pre_grasp_pos: {pre_grasp_pos}")
    log.info(f"grasp_pos: {grasp_pos}")
    lift_pos[:, 2] += 0.3
    grasp_pos[:, 2] += 0.06

    # TODO: line 965-971
    obs = move_to_pose(
        obs, obs_saver, robot_ik, robot, scenario, pre_grasp_pos, ee_quat_target, steps=50, open_gripper=True
    )
    # break
    obs = move_to_pose(
        obs, obs_saver, robot_ik, robot, scenario, grasp_pos, ee_quat_target, steps=50, open_gripper=True
    )
    obs = move_to_pose(
        obs, obs_saver, robot_ik, robot, scenario, grasp_pos, ee_quat_target, steps=50, open_gripper=False
    )
    obs = move_to_pose(
        obs, obs_saver, robot_ik, robot, scenario, lift_pos, ee_quat_target, steps=50, open_gripper=False
    )

    step += 1

obs_saver.save()


# if __name__ == "__main__":
#     import open3d as o3d

#     cloud = o3d.io.read_point_cloud("third_party/gsnet/assets/test.ply")

#     gsnet = GSNet()
#     gg = gsnet.inference(np.array(cloud.points))
#     gsnet.visualize(cloud, gg)


# grasp_position = filtered_gg[0].translation
# grasp_position[2] = -grasp_position[2]

# print(grasp_position, place_position)
# delta_m = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
# # import pdb; pdb.set_trace()
# rotation_output = filtered_gg[0].rotation_matrix.copy()
# print(rotation_output)
# rotation_output[2, :] = -rotation_output[2, :]
# rotation_output[:, 2] = -rotation_output[:, 2]
# print(rotation_output)


# # grippers = filtered_gg[:1].to_open3d_geometry_list()
# # cloud = o3d.geometry.PointCloud()
# # cloud.points = o3d.utility.Vector3dVector(points_envs[0])
# # o3d.visualization.draw_geometries([cloud, *grippers])

# rotation_output = np.dot(rotation_output, delta_m)
# print(rotation_output)
# grasp_quat_R = R.from_matrix(rotation_output).as_quat()
# print(grasp_quat_R)
# rotation_input = grasp_quat_R
# rotation_input = np.array([rotation_input[3],rotation_input[0],rotation_input[1],rotation_input[2]])
# grasp_pose = np.concatenate([grasp_position, rotation_input])

# place_pose = np.concatenate([place_position, rotation_input])

# # R: xyzw
# # IsaacGym: xyzw

# # traj = self.plan_to_pose_curobo(torch.tensor(grasp_pose[:3], dtype = torch.float32), torch.tensor(rotation_input, dtype = torch.float32))
# # self.move_to_traj(traj, close_gripper=False, save_video=save_video, save_root = save_root, start_step = step_num)

# # self.refresh_observation(get_visual_obs=False)

# # R.from_quat(grasp_pose[3:]).as_matrix()
# # R.from_quat(rotation_input).as_matrix()
# # R.from_quat(self.hand_rot.cpu().numpy()).as_matrix()


# rotation_unit_vect = rotation_output[:,2]

# grasp_pre_grasp = grasp_pose.copy()
# grasp_pre_grasp[:3] -= rotation_unit_vect*0.2

# grasp_grasp = grasp_pose.copy()
# grasp_grasp[:3] -= rotation_unit_vect*0.05

# grasp_lift = grasp_pose.copy()
# grasp_lift[2] += 0.3
# # grasp_lift[:3] -= rotation_unit_vect*0.2

# place_pose[:3] -= rotation_unit_vect*0.05
# place_position_lift = place_pose.copy()
# place_position_lift[2] += 0.3
# place_position_place = place_pose.copy()
# place_position_place[2] += 0.05
# place_position_up = place_pose.copy()
# place_position_up[2] += 0.3

# finger_front = np.array([0, 0, -1])
# finger_side = np.array([0, 1, 0])
# finger_front_norm = finger_front / np.linalg.norm(finger_front)
# finger_side_norm = finger_side / np.linalg.norm(finger_side)
# finger_face_norm = np.cross(finger_side_norm, finger_front_norm)

# quaternion = R.from_matrix(np.concatenate([finger_face_norm.reshape(-1,1), finger_side_norm.reshape(-1,1), finger_front_norm.reshape(-1,1)], axis = 1)).as_quat()

# # points_envs, colors_envs, rgb_envs, depth_envs ,seg_envs, ori_points_envs, ori_colors_envs, pixel2pointid, pointid2pixel = self.refresh_observation(get_visual_obs=True)
# # prompt = grasp_obj_name
# # masks, bbox_axis_aligned_envs, grasp_envs = self.inference_gsam(rgb_envs[0][0], ori_points_envs[0][0], ori_colors_envs[0][0], text_prompt=prompt, save_dir=self.cfgs["SAVE_ROOT"])

# # grasp_envs[0] += 0.00

# step_num = 0
# #import pdb; pdb.set_trace()
# # move to pre-grasp
# print("grasp_pre_grasp: ", grasp_pre_grasp)


# self.prepare_curobo(use_mesh=self.cfgs["USE_MESH_COLLISION"])
# step_num, traj = self.control_to_pose(grasp_pre_grasp, close_gripper = False, save_video = save_video, save_root = save_root, step_num = step_num)
# import pdb; pdb.set_trace()
# points_envs, colors_envs, rgb_envs, depth_envs ,seg_envs, ori_points_envs, ori_colors_envs, pixel2pointid, pointid2pixel = self.refresh_observation(get_visual_obs=True)

# trajs = []
# fig_data = []
# for _ in range(5):
#     # add noise
#     _=0
#     target = filtered_gg[_].translation
#     target[2] = -target[2]
#     grasp_grasp[:3] = target
#     step_num, traj = self.control_to_pose(grasp_grasp, close_gripper = False, save_video = save_video, save_root = save_root, step_num = step_num)
#     trajs.append(traj)
#     config_file = load_yaml(join_path(get_robot_path(), "franka.yml"))
#     urdf_file = config_file["robot_cfg"]["kinematics"]["urdf_path"]  # Send global path starting with "/"
#     base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
#     ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]
#     tensor_args = TensorDeviceType()
#     robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)
#     kin_model = CudaRobotModel(robot_cfg.kinematics)
#     qpos = torch.tensor(traj.position, **vars(tensor_args))
#     out = kin_model.get_state(qpos)
#     traj_p = out.ee_position.cpu().numpy()
#     fig_data.append(go.Scatter3d(x=traj_p[:,0], y=traj_p[:,1], z=traj_p[:,2], mode='markers', name='waypoints', marker=dict(size=10, color='red')))
#     for i in range(0, traj_p[:,0].shape[0] - 1): fig_data.append(go.Scatter3d(x=traj_p[:,0][i:i+2], y=traj_p[:,1][i:i+2], z=traj_p[:,2][i:i+2], mode='lines', name='path', line=dict(width=10, color='yellow')))

# fig_data.append(go.Scatter3d(x=points_envs[0][:,0], y=points_envs[0][:,1], z=points_envs[0][:,2], mode='markers', name='waypoints', marker=dict(size=4, color=colors_envs[0])))
# # add lines between waypoints
# fig = go.Figure(data = fig_data)
# fig.show()
# fig.write_html("test.html")


# # move to grasp
# print("grasp_grasp: ", grasp_grasp)
# step_num, traj = self.control_to_pose(grasp_grasp, close_gripper = False, save_video = save_video, save_root = save_root, step_num = step_num)
# step_num = self.move_gripper(close_gripper = True, save_video=save_video, save_root = save_root, start_step = step_num)

# # move to lift
# print("grasp_lift: ", grasp_lift)
# step_num, traj = self.control_to_pose(grasp_lift, close_gripper = True, save_video = save_video, save_root = save_root, step_num = step_num)

# # move to pre-place
# print("place_position_lift: ", place_position_lift)
# step_num, traj = self.control_to_pose(place_position_lift, close_gripper = True, save_video = save_video, save_root = save_root, step_num = step_num)

# # move to place
# print("place_position_place: ", place_position_place)
# step_num, traj = self.control_to_pose(place_position_place, close_gripper = True, save_video = save_video, save_root = save_root, step_num = step_num)
# step_num = self.move_gripper(close_gripper = False, save_video=save_video, save_root = save_root, start_step = step_num)

# # move to pre-place

# print("place_position_up: ", place_position_up)
# step_num, traj = self.control_to_pose(place_position_up, close_gripper = False, save_video = save_video, save_root = save_root, step_num = step_num)
