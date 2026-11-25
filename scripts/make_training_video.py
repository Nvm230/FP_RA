#!/usr/bin/env python3
"""
Genera video_entrenamiento.mp4 mostrando la evolucion REAL de la politica:
500 pasos -> 500k pasos -> 1M pasos -> 5M pasos -> 10M pasos (Plano + Empujones ±55N + Rampa/Escaleras).
"""

import os
import sys
import cv2
import numpy as np
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.quadruped_env import UnitreeGo1Env


def overlay_banner(frame, title, subtitle):
    img = frame.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 68), (11, 60, 93), -1)
    cv2.putText(img, title, (22, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, subtitle, (22, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (130, 220, 255), 1, cv2.LINE_AA)
    return img


def record_stage(base_dir, ckpt_tag, terrain, steps, title, push_events=None, cmd=(0.65, 0.0, 0.0)):
    raw_env = UnitreeGo1Env(
        terrain=terrain, reward_mode="full", max_episode_steps=2000,
        fixed_cmd=list(cmd), domain_rand=False, render_mode="rgb_array"
    )
    dummy = DummyVecEnv([lambda: raw_env])
    vec_env = VecNormalize.load(os.path.join(base_dir, "models", f"vecnorm_{ckpt_tag}.pkl"), dummy)
    vec_env.training = False
    model = PPO.load(os.path.join(base_dir, "models", f"ckpt_{ckpt_tag}.zip"), device="cpu")

    obs = vec_env.reset()
    frames = []
    for s in range(steps):
        t = s * raw_env.dt
        active_fy = 0.0
        if push_events:
            for s0, s1, fy in push_events:
                if s0 <= s <= s1:
                    raw_env.apply_external_force([0.0, fy, 0.0])
                    active_fy = fy
                    break
            else:
                raw_env.clear_external_force()

        if terrain == "climb" and cmd[2] == 0.0:
            from envs.quadruped_env import quat_to_rpy
            rpy = quat_to_rpy(raw_env.data.qpos[3:7])
            y_pos = raw_env.data.qpos[1]
            wz = float(np.clip(-1.5 * rpy[2] - 0.8 * y_pos, -0.55, 0.55))
            raw_env.set_command(cmd[0], cmd[1], wz)

        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = vec_env.step(action)
        if dones[0]:
            break
        info = infos[0]
        frame = raw_env.render()
        sub = (
            f"t={t:4.1f}s | x={info['base_pos'][0]:4.2f}m, z={info['base_pos'][2]:4.2f}m | "
            f"vx={info['lin_vel_body'][0]:4.2f} m/s | Roll={np.degrees(info['rpy'][0]):+5.1f} deg"
        )
        out = overlay_banner(frame, title, sub)

        if push_events:
            for s0, s1, fy in push_events:
                if s0 <= s <= s1 + 22:
                    cv2.rectangle(out, (330, 80), (950, 142), (0, 0, 210), -1)
                    cv2.rectangle(out, (330, 80), (950, 142), (0, 255, 255), 3)
                    cv2.putText(out, f"EMPUJON LATERAL {fy:+.0f} N!", (370, 122),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.05, (255, 255, 255), 3, cv2.LINE_AA)
                    if fy > 0:
                        cv2.arrowedLine(out, (265, 111), (325, 111), (0, 255, 255), 6, tipLength=0.4)
                    else:
                        cv2.arrowedLine(out, (1015, 111), (955, 111), (0, 255, 255), 6, tipLength=0.4)
                    break
        frames.append(out)
    vec_env.close()
    return frames


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(base_dir, "results", "video_entrenamiento.mp4")

    all_frames = []
    all_frames += record_stage(base_dir, "500", "flat", 100, "ETAPA 1: 500 Pasos (Politica Inicial Sin Entrenar)")
    all_frames += record_stage(base_dir, "500k", "flat", 150, "ETAPA 2: 500k Pasos (Aprendiendo Equilibrio y Apoyo)")
    all_frames += record_stage(base_dir, "1M", "flat", 175, "ETAPA 3: 1M Pasos (Trote Diagonal Estable en Plano)")
    all_frames += record_stage(base_dir, "5M", "climb", 250, "ETAPA 4: 5M Pasos (Subiendo Rampa y Plataforma)", cmd=(0.70, 0.0, 0.0))
    all_frames += record_stage(
        base_dir, "10M", "flat", 350,
        "ETAPA 5: 10M Pasos (Plano + Empujones Laterales +-55 N)",
        push_events=[(100, 109, 55.0), (220, 229, -55.0)]
    )
    all_frames += record_stage(
        base_dir, "10M", "climb", 900,
        "ETAPA 6: 10M Pasos (Rampa + Plataforma + Bajada + Escaleras)",
        cmd=(0.70, 0.0, 0.0)
    )

    imageio.mimsave(out_path, all_frames, fps=50)
    print(f"Video de evolucion guardado en: {out_path} ({len(all_frames)} frames)")


if __name__ == "__main__":
    main()
