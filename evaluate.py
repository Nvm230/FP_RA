#!/usr/bin/env python3
"""
Evaluación cuantitativa GOAT, generación de 8 figuras científicas (300 DPI),
secuencia de fotos de entrenamiento y subida de rampa, y video HD MP4.
"""

import os
import sys
import csv
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from envs.quadruped_env import UnitreeGo1Env


plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})


def load_policy(base_dir: str, mode: str, terrain: str = "flat", render_mode=None):
    model_path = os.path.join(base_dir, "models", f"ppo_go1_{mode}.zip")
    vecnorm_path = os.path.join(base_dir, "models", f"vecnorm_{mode}.pkl")

    raw_env = UnitreeGo1Env(
        terrain=terrain,
        reward_mode=mode,
        max_episode_steps=2000,
        fixed_cmd=[0.65, 0.0, 0.0],
        domain_rand=False,
        render_mode=render_mode,
    )
    dummy = DummyVecEnv([lambda: raw_env])
    vec_env = VecNormalize.load(vecnorm_path, dummy)
    vec_env.training = False
    vec_env.norm_reward = False
    model = PPO.load(model_path, device="cpu")
    return model, vec_env, raw_env


def read_train_log(csv_path: str):
    rows = []
    for p in [csv_path, csv_path + ".resume.csv"]:
        if os.path.exists(p):
            with open(p, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append((
                        float(row["timesteps"]),
                        float(row["mean_reward"]),
                        float(row["mean_ep_len"]),
                    ))
    if not rows:
        return np.array([]), np.array([]), np.array([])
    rows.sort(key=lambda x: x[0])
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def plot_training_progression(out_dir: str):
    snaps = [
        ("snap_500.png", "500 pasos: Política Inicial (Sin entrenar)"),
        ("snap_500k.png", "500k pasos: Aprendiendo Equilibrio"),
        ("snap_1M.png", "1M pasos: Trote Diagonal Estable"),
        ("snap_10M.png", "10M pasos: Rampa, Empujones ±55 N y Escaleras"),
    ]
    available = [(os.path.join(out_dir, f), t) for f, t in snaps if os.path.exists(os.path.join(out_dir, f))]
    if len(available) < 2:
        return
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6))
    for ax, (path, title) in zip(axes.flat, available):
        img = imageio.v2.imread(path)
        ax.imshow(img)
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=8)
        ax.axis("off")
    fig.suptitle("Evolución Visual del Aprendizaje por Refuerzo (PPO + Control PD)", fontsize=14, fontweight="bold", y=0.96)
    fig.subplots_adjust(top=0.90, bottom=0.03, left=0.02, right=0.98, hspace=0.20, wspace=0.04)
    out = os.path.join(out_dir, "fig0_training_progression.png")
    plt.savefig(out)
    plt.close()
    print(f"Guardado: {out}")


def plot_learning_curves(base_dir: str, out_dir: str):
    s_f, r_f, l_f = read_train_log(os.path.join(base_dir, "results", "train_log_full.csv"))
    s_b, r_b, l_b = read_train_log(os.path.join(base_dir, "results", "train_log_baseline.csv"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    if len(s_f) > 0:
        ax1.plot(s_f / 1e3, r_f, color="#1f77b4", lw=2, label="PPO Completo (Trote + Rampa + PD)")
        ax2.plot(s_f / 1e3, l_f, color="#1f77b4", lw=2, label="PPO Completo")
    if len(s_b) > 0:
        ax1.plot(s_b / 1e3, r_b, color="#ff7f0e", lw=2, ls="--", label="PPO Baseline (Ablación)")
        ax2.plot(s_b / 1e3, l_b, color="#ff7f0e", lw=2, ls="--", label="PPO Baseline")

    ax1.set_title("Curva de Recompensa Durante el Entrenamiento")
    ax1.set_xlabel("Pasos de simulación (miles)")
    ax1.set_ylabel("Recompensa promedio por episodio")
    ax1.legend()

    ax2.set_title("Duración Promedio del Episodio (Estabilidad)")
    ax2.set_xlabel("Pasos de simulación (miles)")
    ax2.set_ylabel("Pasos sin caer (máx. 1000)")
    ax2.legend()

    plt.tight_layout()
    out = os.path.join(out_dir, "fig1_learning_curves.png")
    plt.savefig(out)
    plt.close()
    print(f"Guardado: {out}")


def run_eval_rollout(model, vec_env, raw_env, steps=600, cmd_schedule=None, push_events=None, record_frames=False):
    obs = vec_env.reset()
    logs = {
        "t": [], "cmd": [], "vel": [], "ang": [],
        "rpy": [], "x": [], "z": [], "fz": [], "tau": [], "power": [], "push": []
    }
    frames = []

    for step in range(steps):
        t = step * raw_env.dt
        if cmd_schedule is not None:
            cmd = cmd_schedule(t)
            raw_env.set_command(*cmd)

        push_f = 0.0
        if push_events is not None:
            applied = False
            for t_start, t_end, f_xyz in push_events:
                if t_start <= t <= t_end:
                    raw_env.apply_external_force(f_xyz)
                    push_f = float(np.linalg.norm(f_xyz))
                    applied = True
                    break
            if not applied:
                raw_env.clear_external_force()

        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = vec_env.step(action)
        if dones[0]:
            print(f"[AVISO] Caida detectada en t={t:.2f}s")
            break

        info = infos[0]
        if record_frames:
            frame = raw_env.render().copy()
            # HUD superior con telemetria en vivo
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 56), (11, 60, 93), -1)
            hud_txt = (
                f"t={t:4.1f}s | Pos: x={info['base_pos'][0]:4.2f}m, z={info['base_pos'][2]:4.2f}m | "
                f"vx={info['lin_vel_body'][0]:4.2f} m/s | Roll={np.degrees(info['rpy'][0]):+5.1f} deg"
            )
            cv2.putText(frame, hud_txt, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

            # Indicador visual llamativo cuando se aplica un empujon externo
            if push_events is not None:
                for t_start, t_end, f_xyz in push_events:
                    if t_start <= t <= (t_end + 0.45):
                        fy = f_xyz[1]
                        label = f"EMPUJON LATERAL {fy:+.0f} N!"
                        cv2.rectangle(frame, (340, 75), (940, 138), (0, 0, 210), -1)
                        cv2.rectangle(frame, (340, 75), (940, 138), (0, 255, 255), 3)
                        cv2.putText(frame, label, (375, 118), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (255, 255, 255), 3, cv2.LINE_AA)
                        # Flecha direccional
                        if fy > 0:
                            cv2.arrowedLine(frame, (280, 106), (335, 106), (0, 255, 255), 6, tipLength=0.4)
                        else:
                            cv2.arrowedLine(frame, (1000, 106), (945, 106), (0, 255, 255), 6, tipLength=0.4)
                        break
            frames.append(frame)

        logs["t"].append(t)
        logs["cmd"].append(info["cmd_vel"].copy())
        logs["vel"].append(info["lin_vel_body"].copy())
        logs["ang"].append(info["ang_vel_body"].copy())
        logs["rpy"].append(np.degrees(info["rpy"].copy()))
        logs["x"].append(info["base_pos"][0])
        logs["z"].append(info["base_pos"][2])
        logs["fz"].append(info["foot_fz"].copy())
        logs["tau"].append(info["tau"].copy())
        logs["power"].append(info["power"])
        logs["push"].append(push_f)

    for k in logs:
        logs[k] = np.array(logs[k])
    return logs, frames


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "results")
    os.makedirs(out_dir, exist_ok=True)

    # 0. Fotos del progreso de entrenamiento
    plot_training_progression(out_dir)

    # 1. Curvas de aprendizaje
    plot_learning_curves(base_dir, out_dir)

    model_f, venv_f, raw_f = load_policy(base_dir, "full", terrain="flat", render_mode="rgb_array")

    # 2. Seguimiento de velocidad variable (12s = 600 pasos)
    def cmd_sched(t):
        if t < 4.0:
            return (0.65, 0.0, 0.0)
        elif t < 8.0:
            return (0.85, 0.0, 0.45)
        else:
            return (0.55, 0.15, 0.0)

    logs_vel, _ = run_eval_rollout(model_f, venv_f, raw_f, steps=600, cmd_schedule=cmd_sched)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    axes[0].plot(logs_vel["t"], logs_vel["cmd"][:, 0], "k--", lw=2, label="Referencia $v_x^{cmd}$")
    axes[0].plot(logs_vel["t"], logs_vel["vel"][:, 0], "#1f77b4", lw=1.0, alpha=0.45, label="Instantánea $v_x$")
    win = 25
    vx_smooth = np.convolve(logs_vel["vel"][:, 0], np.ones(win)/win, mode="same")
    axes[0].plot(logs_vel["t"], vx_smooth, "#1f77b4", lw=2.2, label="Media móvil $v_x$ (0.5 s)")
    axes[0].set_ylabel("$v_x$ [m/s]")
    axes[0].set_title("Seguimiento de Comandos de Velocidad (Unitree Go1 - Trote Diagonal PPO + PD)")
    axes[0].legend(loc="upper right", ncol=2)

    axes[1].plot(logs_vel["t"], logs_vel["cmd"][:, 1], "k--", lw=2, label="Referencia $v_y^{cmd}$")
    axes[1].plot(logs_vel["t"], logs_vel["vel"][:, 1], "#2ca02c", lw=1.8, label="Real $v_y$")
    axes[1].set_ylabel("$v_y$ [m/s]")
    axes[1].legend(loc="upper right")

    axes[2].plot(logs_vel["t"], logs_vel["cmd"][:, 2], "k--", lw=2, label="Referencia $\\omega_z^{cmd}$")
    axes[2].plot(logs_vel["t"], logs_vel["ang"][:, 2], "#d62728", lw=1.8, label="Real $\\omega_z$")
    axes[2].set_ylabel("$\\omega_z$ [rad/s]")
    axes[2].set_xlabel("Tiempo [s]")
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2_velocity_tracking.png"))
    plt.close()

    # 3. Recuperacion ante empujones externos fuertes (+-55 N, 0.18s)
    pushes = [
        (3.0, 3.18, [0.0, 55.0, 0.0]),
        (6.5, 6.68, [0.0, -55.0, 0.0]),
    ]
    logs_push, frames_flat = run_eval_rollout(
        model_f, venv_f, raw_f, steps=500, cmd_schedule=lambda t: (0.65, 0.0, 0.0),
        push_events=pushes, record_frames=True
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(logs_push["t"], logs_push["rpy"][:, 0], label="Roll ($\\phi$)", color="#1f77b4", lw=1.8)
    ax1.plot(logs_push["t"], logs_push["rpy"][:, 1], label="Pitch ($\\theta$)", color="#ff7f0e", lw=1.8)
    ax1.axvspan(3.0, 3.18, color="red", alpha=0.25, label="Empujón lateral (+55 N)")
    ax1.axvspan(6.5, 6.68, color="purple", alpha=0.25, label="Empujón lateral (-55 N)")
    ax1.set_ylabel("Ángulo [grados]")
    ax1.set_title("Estabilidad y Recuperación ante Perturbaciones Externas (±55 N)")
    ax1.legend(loc="upper right")

    ax2.plot(logs_push["t"], logs_push["z"], color="#2ca02c", lw=2, label="Altura del CoM ($z$)")
    ax2.axhline(0.275, color="k", ls="--", lw=1.2, label="Altura nominal (0.275 m)")
    ax2.axvspan(3.0, 3.18, color="red", alpha=0.20)
    ax2.axvspan(6.5, 6.68, color="purple", alpha=0.20)
    ax2.set_ylabel("Altura $z$ [m]")
    ax2.set_xlabel("Tiempo [s]")
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3_push_recovery.png"))
    plt.close()

    # 4. Fuerzas de reaccion en el suelo (GRF) - Trote diagonal
    mask = (logs_push["t"] >= 0.5) & (logs_push["t"] <= 2.5)
    t_gait = logs_push["t"][mask]
    fz_gait = logs_push["fz"][mask]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    labels = ["FR (Delantera Der.)", "FL (Delantera Izq.)", "RR (Trasera Der.)", "RL (Trasera Izq.)"]
    styles = ["-", "--", "--", "-"]
    colors = ["#1f77b4", "#ff7f0e", "#ff7f0e", "#1f77b4"]
    for i in range(4):
        ax.plot(t_gait, fz_gait[:, i], ls=styles[i], color=colors[i], lw=1.6, alpha=0.85, label=labels[i])
    ax.set_title("Fuerzas de Reacción Verticales ($F_z$) - Patrón de Trote Diagonal")
    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("Fuerza normal $F_z$ [N]")
    ax.legend(ncol=2, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig4_grf_gait_pattern.png"))
    plt.close()

    # 5. Subida de rampa + plataforma + escalones (15s = 750 pasos) con guiado de rumbo al centro y=0
    model_c, venv_c, raw_c = load_policy(base_dir, "full", terrain="climb", render_mode="rgb_array")

    def climb_cmd_sched(t):
        y_pos = raw_c.data.qpos[1]
        yaw = np.degrees(raw_c.data.qpos[3:7])  # unused
        from envs.quadruped_env import quat_to_rpy
        rpy = quat_to_rpy(raw_c.data.qpos[3:7])
        wz = float(np.clip(-1.5 * rpy[2] - 0.8 * y_pos, -0.55, 0.55))
        return (0.70, 0.0, wz)

    logs_climb, frames_climb = run_eval_rollout(
        model_c, venv_c, raw_c, steps=900, cmd_schedule=climb_cmd_sched, record_frames=True
    )

    # Secuencia fotografica 2x3 de la subida de rampa y escalones
    if len(frames_climb) >= 6:
        idxs = np.linspace(20, len(frames_climb) - 1, 6, dtype=int)
        fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
        for ax, idx in zip(axes.flat, idxs):
            ax.imshow(frames_climb[idx])
            t_s = idx * raw_c.dt
            x_m = logs_climb["x"][idx]
            z_m = logs_climb["z"][idx]
            ax.set_title(f"t = {t_s:.1f}s | x = {x_m:.2f}m, z = {z_m:.2f}m", fontsize=10, fontweight="bold")
            ax.axis("off")
        plt.suptitle("Secuencia de Locomoción Subiendo Rampa (10°), Plataforma y Escalones", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fig6_climb_sequence.png"))
        plt.close()

    # Perfil de altura z(x) subiendo la rampa
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(logs_climb["x"], logs_climb["z"], color="#1f77b4", lw=2.5, label="Trayectoria del CoM ($z$ vs $x$)")
    ax.set_title("Perfil de Ascenso y Descenso del Centro de Masa sobre Rampa y Escalones")
    ax.set_xlabel("Distancia avanzada $x$ [m]")
    ax.set_ylabel("Altura absoluta del CoM $z$ [m]")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig7_climb_profile.png"))
    plt.close()

    # 6. Ablacion: PPO Full vs Baseline
    model_b, venv_b, raw_b = load_policy(base_dir, "baseline", terrain="flat")
    logs_base, _ = run_eval_rollout(model_b, venv_b, raw_b, steps=400, cmd_schedule=lambda t: (0.65, 0.0, 0.0))

    p_full = float(np.mean(logs_push["power"][:400]))
    p_base = float(np.mean(logs_base["power"])) if len(logs_base["power"]) > 0 else 0.0
    tau_full = float(np.sqrt(np.mean(logs_push["tau"][:400] ** 2)))
    tau_base = float(np.sqrt(np.mean(logs_base["tau"] ** 2))) if len(logs_base["tau"]) > 0 else 0.0
    tilt_full = float(np.sqrt(np.mean(logs_push["rpy"][:200, :2] ** 2)))
    tilt_base = float(np.sqrt(np.mean(logs_base["rpy"][:, :2] ** 2))) if len(logs_base["rpy"]) > 0 else 0.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    names = ["PPO Completo\n(Regularizado)", "PPO Baseline\n(Sin regularizar)"]
    bar_colors = ["#1f77b4", "#ff7f0e"]

    axes[0].bar(names, [p_full, p_base], color=bar_colors, width=0.5)
    axes[0].set_title("Potencia Mecánica Media")
    axes[0].set_ylabel("Potencia $\\sum|\\tau \\dot{q}|$ [W]")

    axes[1].bar(names, [tau_full, tau_base], color=bar_colors, width=0.5)
    axes[1].set_title("Torque Articular RMS")
    axes[1].set_ylabel("Torque RMS [N·m]")

    axes[2].bar(names, [tilt_full, tilt_base], color=bar_colors, width=0.5)
    axes[2].set_title("Error de Inclinación (Roll/Pitch)")
    axes[2].set_ylabel("RMSE [grados]")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig5_torques_energy_ablation.png"))
    plt.close()

    # Guardar video MP4 combinado
    all_frames = frames_flat + frames_climb
    video_path = os.path.join(out_dir, "demo_cuadrupedo_ppo.mp4")
    imageio.mimsave(video_path, all_frames, fps=50)
    print(f"Video guardado en: {video_path} ({len(all_frames)} frames)")

    print("\n=== METRICAS PARA REPORTE ===")
    print(f"Potencia Media  -> Full: {p_full:.2f} W | Baseline: {p_base:.2f} W")
    print(f"Torque RMS      -> Full: {tau_full:.2f} Nm | Baseline: {tau_base:.2f} Nm")
    print(f"Inclinacion RMSE-> Full: {tilt_full:.2f} deg | Baseline: {tilt_base:.2f} deg")
    print(f"Subida de Rampa+Escalones: x_max={np.max(logs_climb['x']):.2f}m, z_max={np.max(logs_climb['z']):.2f}m, t={len(logs_climb['t'])*raw_c.dt:.2f}s")

    venv_f.close()
    venv_b.close()
    venv_c.close()


if __name__ == "__main__":
    main()
