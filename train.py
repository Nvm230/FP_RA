#!/usr/bin/env python3
"""
Entrenamiento paralelo de locomoción cuadrúpeda (Unitree Go1) con PPO + Control PD
hasta 10,000,000 de pasos, guardando checkpoints reales en:
- 500 pasos, 500k pasos, 1M pasos, 5M pasos y 10M pasos
para generar el video evolutivo completo y las curvas de aprendizaje continuas.
"""

import os
import sys
import csv
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import imageio

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from envs.quadruped_env import UnitreeGo1Env


def save_training_snapshot(model, vec_env, out_path: str, terrain: str = "climb", steps: int = 160):
    """Renderiza una foto del robot ejecutando la politica actual en MuJoCo."""
    try:
        raw_env = UnitreeGo1Env(
            terrain=terrain, reward_mode="full", max_episode_steps=300,
            fixed_cmd=[0.70, 0.0, 0.0], domain_rand=False, render_mode="rgb_array"
        )
        dummy = DummyVecEnv([lambda: raw_env])
        eval_venv = VecNormalize(dummy, norm_obs=True, norm_reward=False, clip_obs=10.0)
        eval_venv.obs_rms = vec_env.obs_rms
        eval_venv.training = False

        obs = eval_venv.reset()
        frame = None
        for _ in range(steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = eval_venv.step(action)
            frame = raw_env.render()
            if dones[0]:
                break
        if frame is not None:
            imageio.v2.imwrite(out_path, frame)
            print(f"Snapshot guardado: {out_path}")
        eval_venv.close()
    except Exception as e:
        print(f"No se pudo guardar snapshot {out_path}: {e}")


class TrainingLoggerCallback(BaseCallback):
    """Guarda metricas continuas en CSV, checkpoints reales y snapshots visuales."""

    def __init__(self, csv_path: str, models_dir: str = None, snap_dir: str = None, milestones=None, append_csv: bool = False, verbose: int = 1):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.models_dir = models_dir
        self.snap_dir = snap_dir
        self.milestones = milestones or {}
        self.done_milestones = set()
        self.ep_rewards = []
        self.ep_lengths = []
        self.start_time = time.time()
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        if not append_csv:
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timesteps", "mean_reward", "mean_ep_len", "elapsed_s"])

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.ep_rewards.append(info["episode"]["r"])
                self.ep_lengths.append(info["episode"]["l"])
        return True

    def _on_rollout_end(self) -> None:
        if len(self.ep_rewards) > 0:
            mean_r = float(np.mean(self.ep_rewards[-50:]))
            mean_l = float(np.mean(self.ep_lengths[-50:]))
            elapsed = time.time() - self.start_time
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([self.num_timesteps, f"{mean_r:.4f}", f"{mean_l:.2f}", f"{elapsed:.1f}"])
            if self.verbose > 0 and (self.num_timesteps % 40960 < 10240 or self.num_timesteps < 30000):
                print(
                    f"[{self.num_timesteps:>9d} steps | {elapsed:6.1f}s] "
                    f"Mean Reward: {mean_r:7.2f} | Mean Ep Len: {mean_l:6.1f}"
                )

        for step_target, tag in self.milestones.items():
            if self.num_timesteps >= step_target and tag not in self.done_milestones:
                self.done_milestones.add(tag)
                if self.models_dir:
                    m_path = os.path.join(self.models_dir, f"ckpt_{tag}.zip")
                    v_path = os.path.join(self.models_dir, f"vecnorm_{tag}.pkl")
                    self.model.save(m_path)
                    self.training_env.save(v_path)
                    print(f">>> Checkpoint guardado: {m_path} ({self.num_timesteps} steps)")
                if self.snap_dir:
                    save_training_snapshot(
                        self.model, self.training_env,
                        os.path.join(self.snap_dir, f"snap_{tag}.png"),
                        terrain="climb", steps=160
                    )


def make_env(rank: int, reward_mode: str, terrain: str):
    def _init():
        t = ("flat" if rank % 2 == 0 else "climb") if terrain == "mixed" else terrain
        env = UnitreeGo1Env(terrain=t, reward_mode=reward_mode, domain_rand=True)
        return Monitor(env)
    return _init


def train_10m_full(total_timesteps: int = 10_000_000, n_envs: int = 20, resume_from_1m: bool = False):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    resume_log = os.path.join(results_dir, "train_log_full.csv.resume.csv")
    if os.path.exists(resume_log):
        os.remove(resume_log)

    model_path = os.path.join(models_dir, "ppo_go1_full.zip")
    vecnorm_path = os.path.join(models_dir, "vecnorm_full.pkl")
    log_csv = os.path.join(results_dir, "train_log_full.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy_kwargs = dict(
        activation_fn=nn.ELU,
        net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
    )

    if resume_from_1m and os.path.exists(os.path.join(models_dir, "ckpt_1M.zip")):
        # Recortar CSV hasta 1M para mantener curva continua sin saltos
        if os.path.exists(log_csv):
            with open(log_csv, "r") as f:
                lines = f.readlines()
            kept = [lines[0]] + [l for l in lines[1:] if float(l.split(",")[0]) <= 1_004_020]
            with open(log_csv, "w") as f:
                f.writelines(kept)

        remaining = max(100_000, total_timesteps - 1_004_020)
        print(f"=== Reanudando desde ckpt_1M.zip por {remaining:,} pasos (meta 10M) | Envs: {n_envs} ===")
        vec_env = SubprocVecEnv([make_env(i, "full", "mixed") for i in range(n_envs)])
        vec_env = VecNormalize.load(os.path.join(models_dir, "vecnorm_1M.pkl"), vec_env)
        vec_env.training = True
        vec_env.norm_reward = True
        model = PPO.load(os.path.join(models_dir, "ckpt_1M.zip"), env=vec_env, device=device)
        milestones = {5_000_000: "5M", 9_900_000: "10M"}
        callback = TrainingLoggerCallback(
            csv_path=log_csv, models_dir=models_dir, snap_dir=results_dir,
            milestones=milestones, append_csv=True, verbose=1
        )
        model.learn(total_timesteps=remaining, callback=callback, reset_num_timesteps=False)
    else:
        # Fase 0: 500 pasos reales para el checkpoint inicial ckpt_500
        print("=== Fase 0: Generando checkpoint inicial de 500 pasos ===")
        env_500 = DummyVecEnv([make_env(0, "full", "flat")])
        env_500 = VecNormalize(env_500, norm_obs=True, norm_reward=True, clip_obs=10.0)
        model_500 = PPO(
            "MlpPolicy", env_500, learning_rate=3e-4, n_steps=500, batch_size=250,
            n_epochs=4, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.004,
            policy_kwargs=policy_kwargs, device=device, verbose=0
        )
        model_500.learn(total_timesteps=500)
        model_500.save(os.path.join(models_dir, "ckpt_500.zip"))
        env_500.save(os.path.join(models_dir, "vecnorm_500.pkl"))
        save_training_snapshot(model_500, env_500, os.path.join(results_dir, "snap_500.png"), terrain="flat", steps=40)
        env_500.close()

        print(f"=== Fase 1: Entrenando PPO Completo hasta {total_timesteps:,} pasos | Envs: {n_envs} | Device: {device} ===")
        vec_env = SubprocVecEnv([make_env(i, "full", "mixed") for i in range(n_envs)])
        vec_env = VecNormalize.load(os.path.join(models_dir, "vecnorm_500.pkl"), vec_env)
        vec_env.training = True
        vec_env.norm_reward = True

        model = PPO.load(os.path.join(models_dir, "ckpt_500.zip"), env=vec_env, device=device)
        model.n_steps = 512
        model.batch_size = 640
        model.n_epochs = 5
        model.rollout_buffer = None
        model._setup_model()

        milestones = {
            500_000: "500k",
            1_000_000: "1M",
            5_000_000: "5M",
            9_900_000: "10M",
        }
        callback = TrainingLoggerCallback(
            csv_path=log_csv, models_dir=models_dir, snap_dir=results_dir, milestones=milestones, verbose=1
        )
        model.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=False)

    model.save(model_path)
    vec_env.save(vecnorm_path)
    model.save(os.path.join(models_dir, "ckpt_10M.zip"))
    vec_env.save(os.path.join(models_dir, "vecnorm_10M.pkl"))
    save_training_snapshot(model, vec_env, os.path.join(results_dir, "snap_10M.png"), terrain="climb", steps=220)
    vec_env.close()
    print(f"Entrenamiento 10M completado! Modelo final: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento PPO 10M pasos para Unitree Go1")
    parser.add_argument("--steps", type=int, default=10_000_000)
    parser.add_argument("--envs", type=int, default=20)
    parser.add_argument("--resume-1m", action="store_true")
    args = parser.parse_args()
    train_10m_full(total_timesteps=args.steps, n_envs=args.envs, resume_from_1m=args.resume_1m)
