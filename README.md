# Proyecto Final MT5305: Locomoción Robusta de Cuadrúpedo con PPO + Control PD en MuJoCo

Proyecto desarrollado para **MT5305 – Robótica Avanzada (UTEC)** aplicando **Aprendizaje por Refuerzo Profundo (PPO)** y **Control PD Articular de Bajo Nivel** sobre el robot cuadrúpedo de base flotante **Unitree Go1** en el simulador dinámico **MuJoCo**.

---

## 🎯 Alineación con el Sílabo (MT5305)
- **Robot de Base Flotante (Tema 6.4):** Dinámica completa de cuerpo rígido flotante (18 GDL: 6 de base flotante + 12 articulares) con contactos elípticos y fuerzas de reacción en el suelo (GRF).
- **Control Articular de Bajo Nivel (PD):** $\tau_i = K_p (q_{des,i} - q_i) - K_d \dot{q}_i$ ejecutado a **500 Hz** con saturación física de torques ($\pm 23.7$ N·m en cadera/muslo y $\pm 35.55$ N·m en rodilla).
- **Aprendizaje por Refuerzo en Espacios Continuos (Tema 6.8):** Formulación MDP (48 observaciones, 12 acciones residuales) optimizada con **PPO (Proximal Policy Optimization)** a **50 Hz**, entrenada con aleatorización de dominio y perturbaciones externas.
- **Estudio de Ablación y Pruebas de Robustez:** Comparación entre política completa regularizada vs. *baseline* sin regularización, seguimiento de velocidad variable ($v_x, v_y, \omega_z$), resistencia a empujones laterales y caminata sobre terreno irregular.

---

## 📁 Estructura del Proyecto

```text
Cuadrupedo_PPO/
├── README.md                  # Guía de ejecución y estructura para reporte
├── train.py                   # Entrenamiento paralelo PPO (16 entornos + GPU)
├── evaluate.py                # Evaluación, 5 figuras científicas (300 DPI) y video MP4
├── viewer_demo.py             # Visor 3D interactivo en vivo para exposición oral
├── envs/
│   ├── go1_torque.xml         # Modelo Unitree Go1 con actuadores de torque (<motor>)
│   ├── scene_flat.xml         # Escenario plano
│   ├── scene_rough.xml        # Escenario con desniveles/obstáculos bajos
│   └── quadruped_env.py       # Entorno Gymnasium + Controlador PD 500 Hz
├── models/
│   ├── ppo_go1_full.zip       # Política PPO completa entrenada
│   ├── vecnorm_full.pkl       # Normalizador de observaciones
│   ├── ppo_go1_baseline.zip   # Política baseline (estudio de ablación)
│   └── vecnorm_baseline.pkl
└── results/
    ├── fig1_learning_curves.png         # Curvas de recompensa y estabilidad
    ├── fig2_velocity_tracking.png       # Seguimiento de comandos (vx, vy, wz)
    ├── fig3_push_recovery.png           # Recuperación de Roll/Pitch/Altura ante empujones
    ├── fig4_grf_gait_pattern.png        # Fuerzas verticales Fz (patrón de trote)
    ├── fig5_torques_energy_ablation.png # Ablación: Potencia, Torque RMS e Inclinación
    └── demo_cuadrupedo_ppo.mp4          # Video demostración en HD (1280x720, 50 fps)
```

---

## ▶️ Comandos de Ejecución (en Distrobox `ros2_humble`)

### 1. Entrenar ambas políticas (Completa + Baseline para ablación)
```bash
distrobox enter ros2_humble -- python3 -u /home/nvm/Documents/UTEC/Cuadrupedo_PPO/train.py --mode both --steps 1200000 --envs 16
```

### 2. Evaluar, generar las 5 figuras y grabar el video MP4
```bash
distrobox enter ros2_humble -- python3 /home/nvm/Documents/UTEC/Cuadrupedo_PPO/evaluate.py
```

### 3. Demostración en Vivo 3D (para la Exposición Oral)
```bash
# Caminata estable con empujones automáticos cada 4 segundos:
distrobox enter ros2_humble -- python3 /home/nvm/Documents/UTEC/Cuadrupedo_PPO/viewer_demo.py --terrain flat --vx 0.65 --push

# Caminata sobre terreno irregular con desniveles:
distrobox enter ros2_humble -- python3 /home/nvm/Documents/UTEC/Cuadrupedo_PPO/viewer_demo.py --terrain rough --vx 0.70
```
