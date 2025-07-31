# Third Party
import torch
import numpy as np

# CuRobo common
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.geom.types import WorldConfig, Cuboid # not used? 
from curobo.util.logger import setup_curobo_logger #

# Motion Gen
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.sdf.world import CollisionCheckerType

# Rollout based planning (mpc & trajopt)
from curobo.rollout.rollout_base import Goal
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig
from curobo.wrap.reacher.trajopt import TrajOptSolver, TrajOptSolverConfig

# motion generation profile 
PROFILE_ACTIVE = True
if PROFILE_ACTIVE:
    from torch.profiler import ProfilerActivity, profile

    def plot_traj(trajectory, fig_name):
        # Third Party
        import matplotlib.pyplot as plt

        _, axs = plt.subplots(1, 1)
        q = trajectory

        for i in range(q.shape[-1]):
            axs.plot(q[:, i], label=str(i))
        plt.legend()
        plt.savefig(fig_name)
        # plt.show()

    def motion_gen_profile():
        motion_gen = config_motion_gen_simple()

        start_state = []
        goal_pose = [-0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0] # x, y, z, qw, qx, qy, qz

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            traj = motion_gen_plan_single(motion_gen, start_state, goal_pose)

        prof.export_chrome_trace("trace.json")
        return 


_joint_names =[]

def config_motion_gen_simple(robot_file="rml63.yml", world_file="collision_test.yml"):
    world_config = {
        "cuboid": {
            "table": {
                "dims": [0.1, 0.1, 0.1],  # x, y, z
                "pose": [0.0, 0.0, -0.5, 1, 0, 0, 0.0],  # x, y, z, qw, qx, qy, qz
            },
        },
    }
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_config,
        interpolation_dt=0.01,
    )
    _joint_names = motion_gen_config.robot_cfg.cspace.joint_names
    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup()

    return motion_gen


def motion_gen_plan_single(motion_gen, start_state_js, goal_pose):
    curobo_goal_pose = Pose.from_list([-0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0])  
    curobo_start_state = JointState.from_position(
        torch.zeros(1, 6).cuda(),
        joint_names=[
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ],
    )
    result = motion_gen.plan_single(curobo_start_state, curobo_goal_pose)
    
    # validate planning result
    success = result.success.item()
    error_msg = 'NO_ERROR'
    if not success:
        error_msg = result.status.name
    print(f'Motion planning result: [{success}] <{error_msg}>')

    import ipdb; ipdb.set_trace()
    # get planned traj and plot
    traj = result.interpolated_plan
    plot_traj(traj.position.cpu().numpy(), 'planned_result/interp_traj.png')

    return traj

def main():
    motion_gen = config_motion_gen_simple()

    start_state = []
    goal_pose = [-0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0] # x, y, z, qw, qx, qy, qz
    traj = motion_gen_plan_single(motion_gen, start_state, goal_pose)

    return 

def robot_model_shpere_fit():
    import trimesh
    from curobo.geom.sphere_fit import SphereFitType, fit_spheres_to_mesh
    
    collision_link_names = [
      'base_link',
      'link1',
      'link2',
      'link3',
      'link4',
      'link5',
      'link6',
    ] 
    for link_name in collision_link_names:
        mesh_path = f'/home/hkclr/curobo/curobo/src/curobo/content/assets/robot/rm_description/meshes/rm_63_arm/{link_name}.STL'
        link_mesh = trimesh.load(mesh_path)

        pts, n_radius = fit_spheres_to_mesh(
            link_mesh, n_spheres=1, surface_sphere_radius=0.002, fit_type=SphereFitType.VOXEL_VOLUME_INSIDE, voxelize_method="ray"
        )
        print(f'[{link_name}]: {pts}, {n_radius}')

    import ipdb; ipdb.set_trace()
    return

if __name__ == "__main__":
    main()
    # motion_gen_profile()
    # robot_model_shpere_fit()