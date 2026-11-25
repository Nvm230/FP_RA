# Guía y Prompt para Generar la Presentación Web (15 Minutos — Rúbrica MT5305)

Adjunta el archivo **`Paquete_PPT_Web.zip`** junto con este documento en tu generador de presentaciones web. Todas las imágenes citadas están dentro del `.zip`.

---

## 📋 Estructura Obligatoria según Rúbrica (10 puntos — 15 minutos)
- **[1 pt] Introducción** (Diapositiva 2)
- **[2 pts] Marco Teórico** (Diapositivas 3 y 4)
- **[3 pts] Metodología** (Diapositivas 5, 6 y 7)
- **[3 pts] Resultados + Video/Demo + Análisis de Figuras** (Diapositivas 8, 9, 10, 11 y 12)
- **[1 pt] Conclusiones + Aplicaciones Reales** (Diapositiva 13)

---

## 🎨 Especificaciones Diapositiva por Diapositiva

### Diapositiva 1: Portada
- **Título:** Locomoción Robusta de un Robot Cuadrúpedo mediante Aprendizaje por Refuerzo Profundo (PPO) y Control PD en MuJoCo
- **Subtítulo:** Proyecto Final — MT5305 Robótica Avanzada | UTEC | Plataforma: Unitree Go1 (18 GDL)
- **Imagen opcional:** `snap_10M.png`

### Diapositiva 2: Introducción `[1 pt]`
- **Puntos clave:**
  - Ventaja de cuadrúpedos frente a robots con ruedas en escaleras, rampas y terrenos irregulares.
  - Reto: sistema de **base flotante** (6 GDL no actuados + 12 actuados), dinámica no lineal y contactos intermitentes.
  - Objetivo: diseñar un controlador jerárquico **PPO (50 Hz) + PD articular (500 Hz)** para trotar, seguir comandos $(v_x, v_y, \omega_z)$, resistir empujones laterales ($\pm 55\text{ N}$) y cruzar rampas + escalones.

### Diapositiva 3: Marco Teórico (1/2) — Base Flotante y Control PD `[2 pts]`
- **Ecuación de base flotante:** $M(q)\ddot{q} + C(q,\dot{q})\dot{q} + g(q) = S^T\tau + J_c^T(q)\lambda$ ($q \in \mathbb{R}^{19}$, $\tau \in \mathbb{R}^{12}$).
- **Fuerzas de Reacción en el Suelo (GRF $\lambda$):** Restringidas al cono de fricción elíptico.
- **Controlador PD Articular (500 Hz):** $\tau_i = K_p(q_{des,i} - q_i) - K_d\dot{q}_i$ ($K_p = 32.0$, $K_d = 0.9$), con $q_{des} = q_{default} + 0.25\,a_t$.

### Diapositiva 4: Marco Teórico (2/2) — MDP y PPO `[2 pts]`
- **MDP Continuo:** Tupla $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$ con $\gamma = 0.99$ y ventaja generalizada GAE ($\lambda = 0.95$).
- **Arquitectura Actor-Critic:** MLP `[256, 256, 128]` con activación ELU (~285k parámetros).
- **Objetivo Clipped PPO:** $L^{CLIP}(\theta) = \mathbb{E}[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$ ($\epsilon = 0.2$).
- **Reloj de fase (2.2 Hz):** Señales $[\sin(2\pi\phi), \cos(2\pi\phi)]$ para inducir trote diagonal alternado.

### Diapositiva 5: Metodología (1/3) — Modelo del Robot Unitree Go1 `[3 pts]`
- **Especificaciones:** Masa $\approx 12.8\text{ kg}$, 18 GDL (6 base flotante + 12 actuados en patas FR, FL, RR, RL).
- **Límites físicos:** $\pm 23.7\text{ N}\cdot\text{m}$ (cadera/muslo) y $\pm 35.55\text{ N}\cdot\text{m}$ (rodilla). Actuadores `<motor>` en MuJoCo.
- **Imagen:** `snap_1M.png`

### Diapositiva 6: Metodología (2/3) — Espacio de Estados (50D), Acciones (12D) y Recompensa `[3 pts]`
- **Observaciones (50D):** $v_{body}$ (3), $\omega_{body}$ (3), gravedad proyectada (3), comando $[v_x, v_y, \omega_z]$ (3), $q - q_0$ (12), $\dot{q}$ (12), $a_{t-1}$ (12), fase $[\sin, \cos]$ (2).
- **Recompensa:** Seguimiento exponencial + simetría diagonal + despeje en vuelo + altura relativa sobre los pies (`z_trunk - mean(z_feet)`) + regularización energética y de orientación.
- **Ablación:** Comparación entre `PPO Completo` vs `PPO Baseline` (sin regularización).

### Diapositiva 7: Metodología (3/3) — Coste Computacional y Entrenamiento Paralelo `[3 pts]`
- **Estrategia:** 20 entornos MuJoCo independientes en paralelo (sin apilar 100 robots en una escena).
- **Hardware:** Intel Core Ultra 7 270K Plus (24 núcleos), 30 GiB RAM, NVIDIA RTX 5070 (12 GB VRAM).
- **Métricas reales:** **10,000,000 de pasos** (55.5 horas simuladas) en **22.6 minutos** (~7,350 pasos/s). CPU ~48%, RAM ~26/30 GiB, GPU CUDA 18% (**1,376 MiB VRAM**, 47.4 °C).

### Diapositiva 8: Resultados — Evolución del Aprendizaje (500 $\to$ 10M pasos) `[3 pts]`
- **Imágenes:** `fig0_training_progression.png` y `fig1_learning_curves.png`
- **Análisis:** Muestra cómo pasa de colapsar en 500 pasos a trotar establemente y subir rampas a los 10M pasos.

### Diapositiva 9: Resultados — Seguimiento de Velocidad y Trote Diagonal `[3 pts]`
- **Imágenes:** `fig2_velocity_tracking.png` y `fig4_grf_gait_pattern.png`
- **Análisis:** Velocidad lineal constante sin retrocesos negativos; respuesta ante giros ($\omega_z = 0.45\text{ rad/s}$) y desplazamiento lateral ($v_y = 0.15\text{ m/s}$). Fuerzas $F_z$ confirman alternancia `FR+RL` vs `FL+RR`.

### Diapositiva 10: Resultados — Recuperación ante Empujones Laterales ($\pm 55\text{ N}$) `[3 pts]`
- **Imagen:** `fig3_push_recovery.png`
- **Análisis:** Ante impulsos de $+55\text{ N}$ ($t=3.0\text{ s}$) y $-55\text{ N}$ ($t=6.5\text{ s}$), el ángulo Roll se recupera en $< 0.5\text{ s}$ sin perder la marcha.

### Diapositiva 11: Resultados — Rampa, Escalones y Estudio de Ablación `[3 pts]`
- **Imágenes:** `fig6_climb_sequence.png`, `fig7_climb_profile.png` y `fig5_torques_energy_ablation.png`
- **Tabla comparativa:**
  - Torque RMS: **6.44 N·m** (Full) vs 6.47 N·m (Baseline)
  - Inclinación RMSE: **3.46°** (Full) vs **4.19°** (Baseline) — **-17.4 % de error**
  - Recorrido completo Rampa + Plataforma + Bajada + 3 Escalones: **$x_{\max} = 9.17\text{ m}$**, **$z_{\max} = 0.46\text{ m}$**, **$18.00\text{ s}$ con 0 caídas**.

### Diapositiva 12: Demostración en Video / Vivo `[3 pts]`
- **Reproducir en clase:**
  - `video_entrenamiento.mp4` (evolución `500` $\to$ `500k` $\to$ `1M` $\to$ `5M` $\to$ `10M` pasos)
  - `demo_cuadrupedo_ppo.mp4` (empujones `±55 N` con alerta visual + recorrido completo de 18 s)

### Diapositiva 13: Conclusiones y Aplicaciones en la Vida Real `[1 pt]`
- **Aplicaciones Reales:** Inspección en minería/industria (Perú), búsqueda y rescate (*Search & Rescue*) tras sismos, y transferencia *Sim-to-Real* al Unitree Go1 físico.
- **Conclusiones:** El controlador jerárquico PPO (50 Hz) + PD (500 Hz) con reloj de fase y altura relativa logra locomoción robusta, estable y eficiente energéticamente.
