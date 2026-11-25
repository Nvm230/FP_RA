#!/usr/bin/env python3
"""
Entorno Gymnasium en MuJoCo para el cuadrúpedo Unitree Go1 (versión GOAT):
- Controlador PD articular de bajo nivel a 500 Hz (tau = Kp*(q_des - q) - Kd*qd)
- Reloj de fase de trote diagonal (2.2 Hz) en las observaciones (50 dims)
- Altura relativa sobre los pies (permite subir rampas y escalones sin penalización falsa)
- Recompensa de simetría de trote diagonal (FR+RL vs FL+RR) y velocidad suave
- Soporte para terreno plano ('flat') y rampa+escalones ('climb')
"""

import os
import math
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces


def quat_to_rot_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def quat_to_rpy(q):
    w, x, y, z = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(np.clip(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float64)


class UnitreeGo1Env(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        terrain: str = "flat",
        reward_mode: str = "full",
        max_episode_steps: int = 1000,
        fixed_cmd=None,
        domain_rand: bool = True,
        render_mode: str = None,
    ):
        super().__init__()
        self.terrain = terrain
        self.reward_mode = reward_mode
        self.max_episode_steps = max_episode_steps
        self.fixed_cmd = fixed_cmd
        self.domain_rand = domain_rand
        self.render_mode = render_mode

        env_dir = os.path.dirname(os.path.abspath(__file__))
        xml_file = "scene_climb.xml" if terrain in ("climb", "rough") else "scene_flat.xml"
        self.model = mujoco.MjModel.from_xml_path(os.path.join(env_dir, xml_file))
        self.model.opt.timestep = 0.002  # 500 Hz física y PD
        self.data = mujoco.MjData(self.model)

        self.decimation = 10  # 50 Hz RL
        self.dt = self.model.opt.timestep * self.decimation  # 0.02 s

        # Controlador PD de bajo nivel
        self.kp = 32.0
        self.kd = 0.9
        self.action_scale = 0.25
        self.torque_limits = np.array([23.7, 23.7, 35.55] * 4, dtype=np.float64)

        self.q_default = np.array([0.0, 0.9, -1.8] * 4, dtype=np.float64)

        self.trunk_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.foot_geom_names = ["FR", "FL", "RR", "RL"]
        self.foot_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in self.foot_geom_names
        ]
        self.foot_site_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in self.foot_geom_names
        ]

        # Reloj de fase de trote diagonal (2.2 Hz)
        self.gait_freq = 2.2
        self.phase = 0.0

        # 50 dimensiones: 48 base + 2 reloj de fase (sin, cos)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(50,), dtype=np.float32)

        self.cmd_vel = np.array([0.65, 0.0, 0.0], dtype=np.float64)
        self.prev_action = np.zeros(12, dtype=np.float64)
        self.step_count = 0
        self.renderer = None

    def _sample_command(self):
        if self.fixed_cmd is not None:
            return np.array(self.fixed_cmd, dtype=np.float64)
        if self.terrain in ("climb", "rough"):
            vx = self.np_random.uniform(0.45, 0.85)
            vy = self.np_random.uniform(-0.10, 0.10)
            wz = self.np_random.uniform(-0.25, 0.25)
            return np.array([vx, vy, wz], dtype=np.float64)
        r = self.np_random.random()
        if r < 0.08:
            return np.array([0.0, 0.0, 0.0], dtype=np.float64)
        vx = self.np_random.uniform(0.40, 1.0)
        vy = self.np_random.uniform(-0.20, 0.20)
        wz = self.np_random.uniform(-0.55, 0.55)
        return np.array([vx, vy, wz], dtype=np.float64)

    def _get_foot_forces(self):
        fz = np.zeros(4, dtype=np.float64)
        cf = np.zeros(6, dtype=np.float64)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            for leg_idx, gid in enumerate(self.foot_geom_ids):
                if con.geom1 == gid or con.geom2 == gid:
                    mujoco.mj_contactForce(self.model, self.data, i, cf)
                    fz[leg_idx] += abs(cf[0])
        return fz

    def _get_foot_heights(self):
        return np.array([self.data.site_xpos[sid][2] for sid in self.foot_site_ids], dtype=np.float64)

    def _get_obs(self):
        quat = self.data.qpos[3:7]
        R_wb = quat_to_rot_matrix(quat)
        R_bw = R_wb.T

        lin_vel_body = R_bw @ self.data.qvel[0:3]
        ang_vel_body = R_bw @ self.data.qvel[3:6]
        proj_gravity = R_bw @ np.array([0.0, 0.0, -1.0], dtype=np.float64)

        q = self.data.qpos[7:19]
        qd = self.data.qvel[6:18]
        cmd_scaled = self.cmd_vel * np.array([2.0, 2.0, 0.25])
        phase_vec = np.array([
            math.sin(2.0 * math.pi * self.phase),
            math.cos(2.0 * math.pi * self.phase),
        ])

        obs = np.concatenate([
            lin_vel_body * 2.0,
            ang_vel_body * 0.25,
            proj_gravity,
            cmd_scaled,
            (q - self.q_default) * 1.0,
            qd * 0.05,
            self.prev_action,
            phase_vec,
        ])
        return np.clip(obs, -10.0, 10.0).astype(np.float32), lin_vel_body, ang_vel_body, proj_gravity

    def set_command(self, vx: float, vy: float, wz: float):
        self.cmd_vel = np.array([vx, vy, wz], dtype=np.float64)

    def apply_external_force(self, force_xyz):
        self.data.xfrc_applied[self.trunk_id, :3] = force_xyz

    def clear_external_force(self):
        self.data.xfrc_applied[self.trunk_id, :] = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        if self.domain_rand:
            if self.terrain in ("climb", "rough"):
                r = self.np_random.random()
                if r < 0.30:
                    # Practicar bajada de rampa y transicion a escaleras
                    self.data.qpos[0] = self.np_random.uniform(3.30, 4.40)
                    self.data.qpos[2] = 0.42
                elif r < 0.55:
                    # Practicar subida directa frente a escaleras
                    self.data.qpos[0] = self.np_random.uniform(4.55, 4.85)
            self.data.qpos[7:19] += self.np_random.uniform(-0.04, 0.04, size=12)
            self.data.qvel[6:18] += self.np_random.uniform(-0.08, 0.08, size=12)

        self.clear_external_force()
        mujoco.mj_forward(self.model, self.data)

        self.cmd_vel = self._sample_command()
        self.prev_action = np.zeros(12, dtype=np.float64)
        self.phase = float(self.np_random.random()) if self.domain_rand else 0.0
        self.step_count = 0

        obs, _, _, _ = self._get_obs()
        return obs, {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        q_des = self.q_default + self.action_scale * action

        tau_accum = np.zeros(12, dtype=np.float64)
        for _ in range(self.decimation):
            q = self.data.qpos[7:19]
            qd = self.data.qvel[6:18]
            tau = self.kp * (q_des - q) - self.kd * qd
            tau = np.clip(tau, -self.torque_limits, self.torque_limits)
            self.data.ctrl[:] = tau
            tau_accum += tau
            mujoco.mj_step(self.model, self.data)

        tau_mean = tau_accum / self.decimation
        self.step_count += 1
        self.phase = (self.phase + self.dt * self.gait_freq) % 1.0

        if self.domain_rand and self.fixed_cmd is None and self.step_count % 200 == 100:
            if self.np_random.random() < 0.45:
                push = self.np_random.uniform(-55.0, 55.0, size=2)
                self.data.xfrc_applied[self.trunk_id, :2] = push
            else:
                self.clear_external_force()
        elif self.domain_rand and self.fixed_cmd is None and self.step_count % 200 == 109:
            self.clear_external_force()

        obs, lin_vel_body, ang_vel_body, proj_gravity = self._get_obs()
        foot_fz = self._get_foot_forces()
        foot_z = self._get_foot_heights()
        contacts = foot_fz > 1.0

        rel_height = float(self.data.qpos[2] - np.mean(foot_z))

        lin_err = np.sum((lin_vel_body[:2] - self.cmd_vel[:2]) ** 2)
        ang_err = (ang_vel_body[2] - self.cmd_vel[2]) ** 2
        r_lin_vel = 2.5 * math.exp(-lin_err / 0.20)
        r_ang_vel = 1.0 * math.exp(-ang_err / 0.25)
        r_alive = 0.2

        moving = np.linalg.norm(self.cmd_vel[:2]) > 0.1 or abs(self.cmd_vel[2]) > 0.1
        if moving:
            if self.phase < 0.5:
                desired_contact = np.array([True, False, False, True])  # FR, RL
            else:
                desired_contact = np.array([False, True, True, False])  # FL, RR
            trot_match = np.mean(contacts == desired_contact)
            r_trot = 0.8 * trot_match
            swing_mask = ~desired_contact
            min_stance_z = np.min(foot_z[desired_contact])
            swing_clearance = np.clip(foot_z[swing_mask] - min_stance_z, 0.0, 0.08)
            r_clearance = 2.5 * float(np.mean(swing_clearance))
        else:
            r_trot = 0.5 * float(np.mean(contacts))
            r_clearance = 0.0

        if self.reward_mode == "full":
            r_vz = -1.2 * (lin_vel_body[2] ** 2)
            r_ang_xy = -0.06 * np.sum(ang_vel_body[:2] ** 2)
            r_orient = -2.5 * (proj_gravity[0]**2 * 0.4 + proj_gravity[1]**2 * 1.8)
            r_height = -10.0 * ((rel_height - 0.255) ** 2)
            r_torque = -0.00018 * np.sum(tau_mean ** 2)
            r_smooth = -0.025 * np.sum((action - self.prev_action) ** 2)
            r_no_back = -1.5 * max(0.0, -lin_vel_body[0]) if self.cmd_vel[0] > 0.2 else 0.0
            reward = (
                r_lin_vel + r_ang_vel + r_alive + r_trot + r_clearance
                + r_vz + r_ang_xy + r_orient + r_height
                + r_torque + r_smooth + r_no_back
            )
        else:
            reward = r_lin_vel + r_ang_vel + r_alive

        self.prev_action = action.copy()

        terminated = bool(rel_height < 0.13 or proj_gravity[2] > -0.45)
        if terminated:
            reward -= 2.5

        truncated = bool(self.step_count >= self.max_episode_steps)

        qd_now = self.data.qvel[6:18]
        mech_power = float(np.sum(np.abs(tau_mean * qd_now)))
        rpy = quat_to_rpy(self.data.qpos[3:7])

        info = {
            "base_pos": self.data.qpos[0:3].copy(),
            "rel_height": rel_height,
            "lin_vel_body": lin_vel_body.copy(),
            "ang_vel_body": ang_vel_body.copy(),
            "rpy": rpy,
            "tau": tau_mean.copy(),
            "foot_fz": foot_fz.copy(),
            "power": mech_power,
            "cmd_vel": self.cmd_vel.copy(),
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=720, width=1280)
        self.renderer.update_scene(self.data, camera="tracking")
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


if __name__ == "__main__":
    for t in ["flat", "climb"]:
        env = UnitreeGo1Env(terrain=t)
        obs, _ = env.reset()
        for _ in range(10):
            obs, rew, term, trunc, info = env.step(np.zeros(12))
        print(f"Terrain={t} OK! obs={obs.shape}, rel_height={info['rel_height']:.4f}")
        env.close()
