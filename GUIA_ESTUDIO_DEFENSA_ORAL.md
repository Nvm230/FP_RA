# Guía Ultra-Detallada del Proyecto y Defensa Oral (MT5305 – Robótica Avanzada)

Esta guía explica **todo paso a paso** (qué usamos, cómo funciona por dentro, qué significa cada número, cómo se relaciona con el sílabo de MT5305 y cómo se aplica en la vida real) para que domines el proyecto al 100% en tu exposición oral de 15 minutos.

---

## 1. ¿Qué Software, Librerías y Modelo de Robot Usamos?

* **Simulador Dinámico — MuJoCo (`mujoco`):** Motor físico de alta precisión estándar en investigación de robótica de patas. Modela contactos elípticos, fricción, inercia multicuerpo y actuadores de torque en tiempo real.
* **Modelo del Robot — Unitree Go1 (`mujoco_menagerie`):** Cuadrúpedo comercial real de **~12.8 kg** de masa y **18 Grados de Libertad (GDL)**:
  * **6 GDL de base flotante** en el torso (`x, y, z, roll, pitch, yaw`) que **no tienen motor**.
  * **12 GDL actuados** (3 por pata: abducción de cadera `±23.7 N·m`, muslo `±23.7 N·m` y rodilla `±35.55 N·m`, en patas `FR, FL, RR, RL`).
* **Interfaz de RL — Gymnasium + Stable-Baselines3 (PPO) + PyTorch CUDA:**
  * `Gymnasium` define el entorno (`reset()`, `step()`).
  * `PyTorch` + `Stable-Baselines3` ejecutan la red neuronal **Actor-Critic** con **PPO** en tu GPU **NVIDIA RTX 5070**.
* **Paralelización (`SubprocVecEnv`):** En lugar de poner 100 robots amontonados en una misma escena, ejecutamos **20 simulaciones independientes de MuJoCo en paralelo** (1 robot limpio por proceso, usando tus 24 núcleos de CPU). Esto nos da **~7,350 pasos/segundo** sin saturar memoria.

---

## 2. ¿Qué es un "Paso" (*Step*) y Qué Significan `10,000,000` Pasos?

En robótica con RL hay **dos frecuencias de reloj** distintas (arquitectura jerárquica):

1. **Paso físico y de Control PD (`dt_sim = 0.002 s` $\to$ `500 Hz`):**
   * Es el paso interno de MuJoCo. Cada `2 milisegundos`, nuestro controlador PD calcula el torque exacto $\tau_i = K_p(q_{des, i} - q_i) - K_d\dot{q}_i$ ($K_p = 32.0$, $K_d = 0.9$) y avanza la física.
2. **Paso de Decisión de la Red PPO (`dt_rl = 0.02 s` $\to$ `50 Hz`, `decimation = 10`):**
   * Cada `10` subpasos físicos (`20 milisegundos`), la red neuronal observa el estado del robot (`50` números) y decide una nueva acción (`12` posiciones articulares objetivo).
   * **Un "paso" de entrenamiento (`1 step`) = `0.02 segundos` de vida simulada del robot.**
   * Por lo tanto, **10,000,000 de pasos = `200,000 segundos` = `55.5 horas` de práctica continua caminando, cayendo y aprendiendo**, ¡comprimidas en solo **22.6 minutos** reales gracias a los 20 entornos paralelos y la RTX 5070!

---

## 3. ¿Cómo Funciona el Controlador por Dentro?

### A. Espacio de Observaciones (50 dimensiones)
Lo que "siente" el robot en cada instante (propiocepción pura, sin cámara):
* **Velocidad lineal del torso** en su propio eje $(v_x, v_y, v_z)$ (3)
* **Velocidad angular del torso** $(\omega_x, \omega_y, \omega_z)$ (3)
* **Vector de gravedad proyectado** en el torso $(g_x, g_y, g_z)$ (3) — le indica si está inclinado (`Roll`/`Pitch`).
* **Comando deseado** $(v_x^{cmd}, v_y^{cmd}, \omega_z^{cmd})$ (3) — hacia dónde le pedimos caminar.
* **Error articular** $q - q_{default}$ (12) y **velocidades articulares** $\dot{q}$ (12).
* **Acción previa** $a_{t-1}$ (12) — para penalizar movimientos bruscos.
* **Reloj de fase sinusoidal** $[\sin(2\pi\phi), \cos(2\pi\phi)]$ a `2.2 Hz` (2) — actúa como metrónomo interno para que las patas diagonales (`FR+RL` vs `FL+RR`) troten alternadas.

### B. Acciones y Controlador PD (500 Hz)
La red PPO no predice torques crudos (eso haría vibrar el robot); predice residuos suaves $a_t \in [-1, 1]^{12}$ alrededor de la postura de pie $q_{default} = [0, 0.9, -1.8]\times 4$:
$$q_{des} = q_{default} + 0.25 \cdot a_t, \qquad \tau = K_p(q_{des} - q) - K_d\dot{q}$$

### C. Función de Recompensa y Altura Relativa
* **Seguimiento:** Premia exponencialmente acercarse a $(v_x^{cmd}, v_y^{cmd}, \omega_z^{cmd})$.
* **Trote diagonal y despeje:** Premia que `FR+RL` toquen el suelo cuando $\phi < 0.5$ y `FL+RR` cuando $\phi \ge 0.5$, levantando las patas en vuelo (*swing clearance*).
* **Altura relativa (`rel_height = z_trunk - mean(z_feet)`):** Truco clave para subir rampas y escalones; mide la altura respecto a sus propios pies, no respecto al suelo $z=0$.
* **Regularización (Estudio de Ablación):** Penaliza inclinación (`Roll/Pitch`), torques ($\sum \tau^2$) y cambios bruscos ($\|a_t - a_{t-1}\|^2$). En el modelo **Baseline** quitamos estas penalizaciones para demostrar científicamente que la versión **Full** reduce la inclinación (`3.46°` vs `4.19°`) y el torque RMS (`6.44` vs `6.47 N·m`).

---

## 4. Relación Directa con el Sílabo de MT5305

| Tema del Sílabo | Aplicación Concreta en el Proyecto |
|:---|:---|
| **6.4 Modelado de Robots con Base Flotante** | Dinámica multicuerpo de 18 GDL ($M\ddot{q} + C\dot{q} + g = S^T\tau + J_c^T\lambda$) con 6 GDL libres en el torso y contactos unilaterales con fricción. |
| **6.3 Control de Fuerza / Impedancia** | Controlador PD articular a 500 Hz con saturación física de actuadores y análisis de Fuerzas de Reacción en el Suelo (GRF $F_z$). |
| **6.8 Aprendizaje por Refuerzo (MDP + PPO)** | Formulación MDP en espacios continuos, red Actor-Critic (`[256, 256, 128]` ELU), ventaja GAE ($\lambda=0.95$) y PPO *clipped* ($\epsilon=0.2$). |
| **RA2 y RA3 del Curso** | Control de cuerpo completo en base flotante sobre múltiples terrenos + generación de movimiento mediante RL. |

---

## 5. Aplicaciones en la Vida Real (Impacto Práctico)

1. **Inspección Industrial y Minera (Perú):** Túneles subterráneos, subestaciones eléctricas y plantas industriales con escaleras, rampas y escombros donde un robot con ruedas queda inmovilizado.
2. **Búsqueda y Rescate (*Search & Rescue*):** Ingreso autónomo a zonas colapsadas tras sismos manteniendo estabilidad ante impactos externos (demostrado con empujones de $\pm 55\text{ N}$).
3. **Transferencia *Sim-to-Real*:** Al usar actuadores de torque + PD a 500 Hz, aleatorización de dominio y sensores propioceptivos internos, esta política es directamente transferible a un **Unitree Go1** físico.

---

## 6. Coste Computacional Real Medido

* **Estación:** Intel Core Ultra 7 270K Plus (24 núcleos) | 30 GiB RAM | NVIDIA RTX 5070 (12 GB VRAM).
* **Rendimiento:** `10,000,000` de pasos en **22.6 minutos** (~7,350 pasos/s con 20 entornos paralelos).
* **Consumo medido:** CPU ~48%, RAM ~26 / 30 GiB, GPU CUDA 18% (**1,376 MiB VRAM**, 47.4 °C).

---

## 7. Preguntas Clave que Puede Hacer el Profesor (y Respuestas)

1. **¿Por qué usan PPO a 50 Hz y PD a 500 Hz?**
   * *Respuesta:* Controlar torques directo a 50 Hz inestabiliza los contactos. El PD a 500 Hz aporta amortiguamiento físico inmediato y permite que PPO aprenda referencias suaves transferibles a hardware real.
2. **¿Por qué un reloj de fase de 2.2 Hz?**
   * *Respuesta:* Sin él, la política cae en saltos simétricos donde la velocidad oscila. El reloj rompe la simetría y genera un trote diagonal alternado (`FR+RL` vs `FL+RR`), visible en la gráfica de $F_z$.
3. **¿Por qué necesitó un controlador de rumbo (`wz_cmd`) al cruzar el circuito largo?**
   * *Respuesta:* La política sigue comandos $(v_x, v_y, \omega_z)$ en marco local sin cámara externa. En recorridos largos sobre rampas, pequeñas asimetrías de contacto acumulan deriva en `yaw`; un regulador simple en $\omega_z$ simula al operador manteniendo el robot centrado en la rampa.
4. **¿Qué demostró el estudio de ablación?**
   * *Respuesta:* La política completa regularizada redujo el error de inclinación RMSE de `4.19°` a `3.46°` (-17.4%) y el torque RMS de `6.47` a `6.44 N·m` manteniendo estabilidad total.
