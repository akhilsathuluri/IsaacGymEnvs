import numpy as np
import os
import time

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.gymtorch import *

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask

import torch


# a class to create RL training tasks
class SoloJump(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]
        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.actions_cost_scale = self.cfg["env"]["actionsCost"]
        self.energy_cost_scale = self.cfg["env"]["energyCost"]
        self.joints_at_limit_cost_scale = self.cfg["env"]["jointsAtLimitCost"]
        self.death_cost = self.cfg["env"]["deathCost"]
        self.termination_height = self.cfg["env"]["terminationHeight"]

        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        # obs -- 2*2 joinq+v and base 2 z+dz = 6
        self.cfg["env"]["numObservations"] = 6
        self.cfg["env"]["numActions"] = 2

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        
        # default viewer state
        if self.viewer != None:
            cam_pos = gymapi.Vec3(0.7, 0.0, 0.7)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:,7:13] = 0  # set angular and lin vel = 0

        # ---------------------------------------------------
        # Check these !!!
        # ---------------------------------------------------
        # create some wrapper tensors for different slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.initial_dof_pos = torch.zeros_like(self.dof_pos, device=self.device, dtype=torch.float)
        zero_tensor = torch.tensor([0.0], device=self.device)
        self.initial_dof_pos = torch.where(self.dof_limits_lower > zero_tensor, self.dof_limits_lower,
                                           torch.where(self.dof_limits_upper < zero_tensor, self.dof_limits_upper, self.initial_dof_pos))
        self.initial_dof_vel = torch.zeros_like(self.dof_vel, device=self.device, dtype=torch.float)
        # ---------------------------------------------------

        # print(self.dof_state.shape)
        # print(self.root_states.shape)

    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        print(f'num envs {self.num_envs} env spacing {self.cfg["env"]["envSpacing"]}')
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        asset_file = self.cfg["env"]["asset"]["assetFileName"]

        # configure robot options
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = False
        asset_options.angular_damping = 0.01
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        # asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True

        solo_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(solo_asset)

        start_pose = gymapi.Transform()

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(solo_asset)

        self.solo_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            solo_handle = self.gym.create_actor(env_ptr, solo_asset, start_pose, "solo", i, 1, 0)

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, solo_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.97, 0.38, 0.06))

            self.envs.append(env_ptr)
            self.solo_handles.append(solo_handle)

        dof_prop = self.gym.get_actor_dof_properties(env_ptr, solo_handle)
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

    def compute_reward(self):
        # Compute and return the rewards
        base_pos = self.obs_buf[:,0]
        j1_pos = self.obs_buf[:,1]
        j2_pos = self.obs_buf[:,2]
        base_vel = self.obs_buf[:,3]
        j1_vel = self.obs_buf[:,4]
        j2_vel = self.obs_buf[:,5]

        # do cartpole style
        self.rew_buf[:], self.reset_buf[:] = compute_solo_reward(
            base_pos, 
            j1_pos, 
            j2_pos, 
            base_vel, 
            j1_vel, 
            j2_vel,
            self.reset_buf, 
            self.progress_buf, 
            self.max_episode_length
        )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        # from the trifinger example
        self.obs_buf = torch.cat([self.dof_pos, self.dof_vel], dim=-1)
        return self.obs_buf

    def reset_idx(self, env_ids):
        # positions = (torch.zeros((len(env_ids), self.num_dof), device=self.device))
        positions = (torch.rand((len(env_ids), self.num_dof), device=self.device))
        velocities = (torch.rand((len(env_ids), self.num_dof), device=self.device))
        # set initial base position at reset
        positions[:,0] *= 0.25
        positions[:,1] = np.pi/2
        positions[:,2] = 0

        self.dof_pos[env_ids, :] = positions[:]
        self.dof_vel[env_ids, :] = velocities[:]

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


        self.gym.set_dof_state_tensor_indexed(self.sim,
                                        gymtorch.unwrap_tensor(self.dof_state),
                                        gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        # copying the action application style for an underactuated system similar to the one applied to the cart-pole
        actions_tensor = torch.zeros(self.num_envs * self.num_dof, device=self.device, dtype=torch.float)
        actions_scale = 10.0
        actions_tensor.view(self.num_envs, self.num_dof)[:, 1:] = actions.to(self.device) * actions_scale
        forces = gymtorch.unwrap_tensor(actions_tensor)
        self.gym.set_dof_actuation_force_tensor(self.sim, forces)
        # print(actions_tensor)

    def post_physics_step(self):
        self.progress_buf += 1
        # self.randomize_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_solo_reward(base_pos, 
                        j1_pos, 
                        j2_pos, 
                        base_vel, 
                        j1_vel, 
                        j2_vel,
                        reset_buf, 
                        progress_buf, 
                        max_episode_length: int):
    # reward for reaching the final pose
    reward = 1.0 - 1e2*(base_pos - 0.34)**2 - (j1_pos - 3.14)**2 - (j2_pos)**2
    # lets try just jumping to a max height reward
    max_height = 0.36
    # reward = 1.0 - 1e2*(base_pos - max_height)**2
    # LATER ADD REWARD FOR LARGE ACTIONS
    # add a terminal state condition
    reset = torch.where(torch.abs(base_pos) > max_height, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)
    
    return reward, reset


