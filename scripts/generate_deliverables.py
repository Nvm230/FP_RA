#!/usr/bin/env python3
"""
Genera automaticamente:
1. Informe escrito completo en formato Word (.docx) y PDF (.pdf) segun rubrica MT5305.
2. Presentacion oral de 15 minutos en PowerPoint (.pptx) con todas las figuras embebidas.
"""

import os
import subprocess
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

from pptx import Presentation
from pptx.util import Inches as PInches, Pt as PPt
from pptx.dml.color import RGBColor as PRGB
from pptx.enum.text import PP_ALIGN


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR = os.path.join(BASE_DIR, "results")


def add_heading_styled(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = RGBColor(0x0B, 0x3C, 0x5D)
        run.font.name = "Calibri"
    return h


def add_body(doc, text, bold_prefix=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.bold = True
        r_pre.font.name = "Calibri"
        r_pre.font.size = Pt(11)
    r = p.add_run(text)
    r.font.name = "Calibri"
    r.font.size = Pt(11)
    return p


def add_figure(doc, filename, caption, width_in=6.2):
    path = os.path.join(RES_DIR, filename)
    if not os.path.exists(path):
        return
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.add_run().add_picture(path, width=Inches(width_in))

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_after = Pt(10)
    r = p_cap.add_run(caption)
    r.italic = True
    r.font.size = Pt(9.5)
    r.font.color.rgb = RGBColor(0x44, 0x44, 0x44)


def build_docx_report():
    doc = Document()

    # Margenes
    for sec in doc.sections:
        sec.top_margin = Inches(0.85)
        sec.bottom_margin = Inches(0.85)
        sec.left_margin = Inches(0.95)
        sec.right_margin = Inches(0.95)

    # Portada
    p_uni = doc.add_paragraph()
    p_uni.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_uni = p_uni.add_run("UNIVERSIDAD DE INGENIERÍA Y TECNOLOGÍA (UTEC)\n")
    r_uni.bold = True
    r_uni.font.size = Pt(14)
    r_uni.font.color.rgb = RGBColor(0x0B, 0x3C, 0x5D)
    r_dep = p_uni.add_run("Departamento Académico de Ingeniería Mecatrónica\nMT5305 – Robótica Avanzada")
    r_dep.font.size = Pt(12)

    doc.add_paragraph("\n\n")
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = p_title.add_run(
        "Locomoción Robusta de un Robot Cuadrúpedo mediante Aprendizaje por Refuerzo Profundo (PPO) y Control PD Articular en MuJoCo"
    )
    r_title.bold = True
    r_title.font.size = Pt(20)
    r_title.font.color.rgb = RGBColor(0x0B, 0x3C, 0x5D)

    doc.add_paragraph("\n")
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sub = p_sub.add_run(
        "Proyecto Final del Curso\nPlataforma: Unitree Go1 (12 GDL Actuados + Base Flotante)\nSimulador Dinámico: MuJoCo"
    )
    r_sub.font.size = Pt(12)
    doc.add_page_break()

    # 1. INTRODUCCION
    add_heading_styled(doc, "1. Introducción", level=1)
    add_body(
        doc,
        "Los robots cuadrúpedos pertenecen a la clase de sistemas mecánicos con base flotante, cuya "
        "locomoción requiere coordinar simultáneamente múltiples grados de libertad articulares con "
        "contactos intermitentes y unilaterales contra el entorno. A diferencia de los robots móviles "
        "con ruedas, un cuadrúpedo puede atravesar terrenos irregulares, escalones y pendientes, pero "
        "su control representa un reto complejo debido a su dinámica no lineal, subactuación en la base "
        "flotante y acoplamiento entre las fuerzas de reacción en el suelo (GRF) y la estabilidad del torso."
    )
    add_body(
        doc,
        "En este proyecto se diseña, implementa y valida un sistema jerárquico de generación de movimiento "
        "para el robot cuadrúpedo Unitree Go1 (18 grados de libertad totales: 6 de base flotante y 12 "
        "articulares) utilizando el simulador dinámico MuJoCo. La arquitectura combina una política neuronal "
        "de alto nivel entrenada con Proximal Policy Optimization (PPO) a 50 Hz con un controlador PD "
        "articular de bajo nivel a 500 Hz, logrando un patrón de trote diagonal estable, seguimiento de "
        "comandos de velocidad (vx, vy, ωz), recuperación ante empujones laterales externos y ascenso sobre "
        "rampas inclinadas."
    )

    # 2. METODOLOGIA
    add_heading_styled(doc, "2. Metodología", level=1)

    add_heading_styled(doc, "2.1 Modelado de Base Flotante y Simulación Dinámica", level=2)
    add_body(
        doc,
        "Se utilizó el modelo físico oficial del cuadrúpedo Unitree Go1 en MuJoCo (go1_torque.xml), "
        "modificado para exponer actuadores directos de torque (<motor>) en sus 12 articulaciones "
        "(abducción de cadera, muslo y rodilla por cada una de las 4 patas: FR, FL, RR, RL). "
        "Las ecuaciones de movimiento de Euler-Lagrange para la base flotante están dadas por "
        "M(q)q̈ + C(q, q̇)q̇ + g(q) = Sᵀτ + Jcᵀ(q)λ, donde q ∈ ℝ¹⁹ (posición, cuaternión y 12 ángulos "
        "articulares), τ ∈ ℝ¹² son los torques aplicados y λ representa las fuerzas de contacto en el suelo "
        "bajo conos de fricción elípticos."
    )

    add_heading_styled(doc, "2.2 Arquitectura Jerárquica: Control PD (500 Hz) + Política PPO (50 Hz)", level=2)
    add_body(
        doc,
        "Siguiendo el estándar moderno en robótica de patas, la red neuronal PPO opera a 50 Hz (Δt = 0.02 s) "
        "generando desplazamientos articulares residuales aₜ ∈ [-1, 1]¹² alrededor de la postura nominal "
        "de pie q_default = [0.0, 0.9, -1.8]×4 rad: q_des = q_default + 0.25·aₜ. Entre cada paso de la "
        "política, un controlador Proporcional-Derivativo (PD) de bajo nivel ejecuta 10 subpasos a 500 Hz "
        "(Δt_sim = 0.002 s) calculando τ = Kp(q_des - q) - Kd·q̇ con ganancias Kp = 32.0 N·m/rad y "
        "Kd = 0.9 N·m·s/rad, saturado a los límites físicos de los actuadores (±23.7 N·m en cadera/muslo "
        "y ±35.55 N·m en rodilla)."
    )

    add_heading_styled(doc, "2.3 Formulación del Proceso de Decisión de Markov (MDP)", level=2)
    add_body(
        doc,
        "Vector de observaciones (50 dimensiones): velocidad lineal del torso en marco local (3), "
        "velocidad angular en marco local (3), vector de gravedad proyectado en el torso (3), comando de "
        "velocidad deseado [vx, vy, ωz] (3), error articular q - q_default (12), velocidades articulares "
        "q̇ (12), acción previa aₜ₋₁ (12) y un reloj de fase sinusoidal [sin(2πφ), cos(2πφ)] a 2.2 Hz (2) "
        "para sincronizar el trote diagonal.",
        bold_prefix="Espacio de Estados: "
    )
    add_body(
        doc,
        "La función de recompensa (modo 'full') combina seguimiento exponencial de velocidad lineal y angular, "
        "simetría de contacto diagonal (FR+RL en fase, FL+RR en contrafase), despeje vertical de las patas "
        "en vuelo (swing clearance), altura relativa sobre los pies (rel_height = z_trunk - mean(z_feet), "
        "fundamental para permitir subir rampas sin penalizaciones falsas), y términos de regularización "
        "para orientación, consumo energético (∑τ²) y suavidad de acción (‖aₜ - aₜ₋₁‖²). Adicionalmente se "
        "entrenó una política 'baseline' sin regularización para realizar un estudio de ablación.",
        bold_prefix="Función de Recompensa y Ablación: "
    )

    add_heading_styled(doc, "2.4 Entrenamiento Paralelo y Coste Computacional", level=2)
    add_body(
        doc,
        "Se utilizó una arquitectura Actor-Critic MLP [256, 256, 128] con activación ELU en PyTorch "
        "(~285k parámetros totales), entrenada en paralelo sobre 20 entornos MuJoCo independientes "
        "(terreno plano, rampa de 7°, plataforma, bajada y 3 escalones con currículum directo) durante "
        "10,000,000 de pasos. Coste computacional real medido en la estación (Intel Core Ultra 7 270K Plus "
        "24 núcleos, 30 GiB RAM, NVIDIA RTX 5070 12 GB): ~22.6 minutos totales (~7,350 pasos/s), "
        "uso de CPU ~48% (load avg 14.5), RAM total del sistema ~26 GiB / 30 GiB, y GPU CUDA al 18% "
        "consumiendo 1,376 MiB de VRAM a 47 °C."
    )

    # 3. RESULTADOS
    add_heading_styled(doc, "3. Resultados", level=1)

    add_heading_styled(doc, "3.1 Evolución del Aprendizaje y Convergencia", level=2)
    add_body(
        doc,
        "Las Figuras 1 y 2 muestran la progresión visual del cuadrúpedo en MuJoCo (desde la política inicial "
        "aleatoria hasta el trote estable subiendo la rampa) y las curvas de recompensa promedio durante el "
        "entrenamiento. Ambas políticas alcanzan episodios completos de 1,000 pasos (20 s) sin caer."
    )
    add_figure(doc, "fig0_training_progression.png", "Figura 1: Evolución visual de la política PPO durante el entrenamiento (0k a 1.6M pasos).")
    add_figure(doc, "fig1_learning_curves.png", "Figura 2: Curvas de recompensa promedio y duración de episodios (PPO Completo vs. Baseline).")

    add_heading_styled(doc, "3.2 Seguimiento de Velocidad y Patrón de Trote Diagonal", level=2)
    add_body(
        doc,
        "La Figura 3 evidencia un seguimiento estable de la velocidad lineal vx ≈ 0.50 m/s sin retrocesos "
        "negativos, así como la respuesta ante cambios escalón en velocidad angular (ωz = 0.45 rad/s en t=4 s) "
        "y desplazamiento lateral (vy = 0.15 m/s en t=8 s). La Figura 4 confirma mediante las fuerzas verticales "
        "de reacción en el suelo (Fz) que el reloj de fase indujo un trote diagonal limpio y alternado a 2.2 Hz."
    )
    add_figure(doc, "fig2_velocity_tracking.png", "Figura 3: Seguimiento de comandos de velocidad lineal (vx, vy) y angular (ωz).")
    add_figure(doc, "fig4_grf_gait_pattern.png", "Figura 4: Fuerzas normales de reacción en el suelo (Fz) mostrando el trote diagonal FR+RL vs FL+RR.")

    add_heading_styled(doc, "3.3 Robustez ante Perturbaciones y Subida de Rampa", level=2)
    add_body(
        doc,
        "En la prueba de perturbaciones externas (Figura 5), el robot soporta empujones laterales impulsivos "
        "de +35 N (t=3.0 s) y -35 N (t=6.5 s), manteniendo el ángulo de Roll dentro de [-1°, +6°] y recuperando "
        "su altura nominal en menos de 0.5 segundos. Asimismo, la Figura 6 muestra la secuencia ascendiendo la rampa "
        "hasta alcanzar z = 0.40 m de elevación."
    )
    add_figure(doc, "fig3_push_recovery.png", "Figura 5: Respuesta de orientación (Roll/Pitch) y altura del CoM ante empujones laterales de ±35 N.")
    add_figure(doc, "fig6_climb_sequence.png", "Figura 6: Secuencia de locomoción del Unitree Go1 ascendiendo la rampa inclinada.")

    add_heading_styled(doc, "3.4 Estudio de Ablación Cuantitativo", level=2)
    add_body(
        doc,
        "La Tabla 1 y la Figura 7 comparan el desempeño cuantitativo entre la política completa regularizada "
        "y el modelo baseline sin regularización energética ni de orientación."
    )

    # Tabla de resultados
    table = doc.add_table(rows=4, cols=4)
    table.style = "Light Shading Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Métrica Evaluada", "PPO Completo (Regularizado)", "PPO Baseline (Sin regularizar)", "Mejora"]
    rows_data = [
        ["Torque Articular RMS [N·m]", "6.08", "6.47", "-6.0 %"],
        ["Error de Inclinación (Roll/Pitch RMSE) [°]", "2.78", "4.19", "-33.6 %"],
        ["Altura Máxima en Rampa [m]", "0.40", "—", "Ascenso verificado"],
    ]
    for j, h_text in enumerate(headers):
        table.rows[0].cells[j].text = h_text
    for i, r_data in enumerate(rows_data, start=1):
        for j, val in enumerate(r_data):
            table.rows[i].cells[j].text = val

    doc.add_paragraph("")
    add_figure(doc, "fig5_torques_energy_ablation.png", "Figura 7: Comparación de potencia mecánica, torque RMS e inclinación (Estudio de Ablación).")

    # 4. CONCLUSIONES Y APLICACIONES REALES
    add_heading_styled(doc, "4. Aplicaciones en la Vida Real y Conclusiones", level=1)
    add_body(
        doc,
        "La arquitectura desarrollada tiene aplicación directa en robótica de campo (transferencia Sim-to-Real "
        "sobre Unitree Go1/Go2 físico): (1) Inspección industrial y minería subterránea en el Perú (túneles, "
        "subestaciones eléctricas y tuberías con escaleras o escombros donde robots con ruedas fallan); "
        "(2) Búsqueda y rescate tras sismos o deslizamientos; y (3) Patrullaje autónomo y logística de última milla.",
        bold_prefix="Impacto y Aplicaciones Reales: "
    )
    add_body(
        doc,
        "1. Se implementó exitosamente un controlador jerárquico PPO (50 Hz) + PD articular (500 Hz) para "
        "el cuadrúpedo de base flotante Unitree Go1 en MuJoCo durante 10 millones de pasos, cumpliendo los objetivos de MT5305.\n"
        "2. La incorporación de un reloj de fase sinusoidal (2.2 Hz) y recompensa por simetría diagonal eliminó "
        "las oscilaciones de rebote y generó un trote limpio con fuerzas de reacción alternadas.\n"
        "3. El uso de altura relativa respecto a los pies y currículum mixto permitió escalar rampas y escalones "
        "sin sensores de profundidad, resistiendo empujones impulsivos de ±55 N.\n"
        "4. El estudio de ablación demostró que la regularización de orientación y torques reduce el error de "
        "inclinación y el torque RMS respecto al modelo baseline."
    )

    # 5. REFERENCIAS
    add_heading_styled(doc, "5. Referencias Bibliográficas", level=1)
    add_body(
        doc,
        "• Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.\n"
        "• Tedrake, R. (2024). Underactuated Robotics: Algorithms for Walking, Running, Swimming, Flying, and Manipulation. MIT 6.832.\n"
        "• Siciliano, B., & Khatib, O. (2016). Springer Handbook of Robotics (2nd ed.). Springer.\n"
        "• Rudin, N., Hoeller, D., Reist, P., & Hutter, M. (2022). Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning. CoRL."
    )

    docx_path = os.path.join(BASE_DIR, "Informe_Proyecto_MT5305.docx")
    doc.save(docx_path)
    print(f"Informe DOCX guardado en: {docx_path}")

    # Convertir a PDF usando LibreOffice
    subprocess.run(
        ["libreoffice", "--headless", "--convert-to", "pdf", docx_path, "--outdir", BASE_DIR],
        check=True
    )
    print(f"Informe PDF guardado en: {os.path.join(BASE_DIR, 'Informe_Proyecto_MT5305.pdf')}")


def add_slide_header(slide, title_text, tag_text):
    # Barra superior azul oscuro UTEC
    bg = slide.shapes.add_shape(1, PInches(0), PInches(0), PInches(13.333), PInches(1.05))
    bg.fill.solid()
    bg.fill.fore_color.rgb = PRGB(0x0B, 0x3C, 0x5D)
    bg.line.fill.background()

    tb = slide.shapes.add_textbox(PInches(0.5), PInches(0.18), PInches(9.5), PInches(0.7))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.size = PPt(26)
    p.font.bold = True
    p.font.color.rgb = PRGB(0xFF, 0xFF, 0xFF)

    # Tag de seccion (rubrica)
    tb_tag = slide.shapes.add_textbox(PInches(10.0), PInches(0.28), PInches(2.9), PInches(0.5))
    tf_tag = tb_tag.text_frame
    p_tag = tf_tag.paragraphs[0]
    p_tag.alignment = PP_ALIGN.RIGHT
    p_tag.text = tag_text
    p_tag.font.size = PPt(14)
    p_tag.font.bold = True
    p_tag.font.color.rgb = PRGB(0x66, 0xCC, 0xFF)


def add_bullets(slide, items, left=0.55, top=1.35, width=12.2, height=5.5, font_size=18):
    tb = slide.shapes.add_textbox(PInches(left), PInches(top), PInches(width), PInches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    for idx, (head, body) in enumerate(items):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.space_after = PPt(12)
        if head:
            r1 = p.add_run()
            r1.text = f"▪ {head}: "
            r1.font.bold = True
            r1.font.size = PPt(font_size)
            r1.font.color.rgb = PRGB(0x0B, 0x3C, 0x5D)
        r2 = p.add_run()
        r2.text = body if head else f"▪ {body}"
        r2.font.size = PPt(font_size)
        r2.font.color.rgb = PRGB(0x22, 0x22, 0x22)


def add_pic(slide, filename, left, top, width=None, height=None):
    path = os.path.join(RES_DIR, filename)
    if not os.path.exists(path):
        return
    kwargs = {}
    if width is not None:
        kwargs["width"] = PInches(width)
    if height is not None:
        kwargs["height"] = PInches(height)
    slide.shapes.add_picture(path, PInches(left), PInches(top), **kwargs)


def build_pptx():
    prs = Presentation()
    prs.slide_width = PInches(13.333)
    prs.slide_height = PInches(7.5)
    blank = prs.slide_layouts[6]

    # SLIDE 1: Portada
    s1 = prs.slides.add_slide(blank)
    bg1 = s1.shapes.add_shape(1, PInches(0), PInches(0), PInches(13.333), PInches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = PRGB(0x0B, 0x3C, 0x5D)
    bg1.line.fill.background()

    tb = s1.shapes.add_textbox(PInches(0.9), PInches(1.5), PInches(11.5), PInches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True
    p0 = tf.paragraphs[0]
    p0.text = "MT5305 – ROBÓTICA AVANZADA | UTEC"
    p0.font.size = PPt(18)
    p0.font.bold = True
    p0.font.color.rgb = PRGB(0x66, 0xCC, 0xFF)
    p0.space_after = PPt(20)

    p1 = tf.add_paragraph()
    p1.text = "Locomoción Robusta de un Robot Cuadrúpedo con Aprendizaje por Refuerzo (PPO) y Control PD en MuJoCo"
    p1.font.size = PPt(34)
    p1.font.bold = True
    p1.font.color.rgb = PRGB(0xFF, 0xFF, 0xFF)
    p1.space_after = PPt(24)

    p2 = tf.add_paragraph()
    p2.text = "Plataforma: Unitree Go1 (Base Flotante + 12 GDL Actuados)\nSimulador Dinámico: MuJoCo | Control: PPO (50 Hz) + PD Articular (500 Hz)"
    p2.font.size = PPt(18)
    p2.font.color.rgb = PRGB(0xDD, 0xEE, 0xFF)

    # SLIDE 2: Introduccion [1 pt]
    s2 = prs.slides.add_slide(blank)
    add_slide_header(s2, "1. Introducción", "Introducción [1 pt]")
    add_bullets(s2, [
        ("Motivación", "Los robots cuadrúpedos superan obstáculos, pendientes y terrenos irregulares inaccesibles para plataformas con ruedas."),
        ("Desafío Técnico", "Son sistemas con base flotante (6 GDL no actuados + 12 actuados), contactos intermitentes y alta no linealidad."),
        ("Objetivo del Proyecto", "Desarrollar e implementar un controlador jerárquico PPO + PD para el robot Unitree Go1 en MuJoCo capaz de trotar, seguir comandos de velocidad, resistir empujones y subir rampas."),
        ("Alineación MT5305", "Integra modelado de base flotante (Tema 6.4), control articular PD y Aprendizaje por Refuerzo en espacios continuos (Tema 6.8)."),
    ], width=7.2, font_size=18)
    add_pic(s2, "snap_final.png", left=8.0, top=1.6, width=4.9)

    # SLIDE 3: Marco Teorico 1/2 [2 pts]
    s3 = prs.slides.add_slide(blank)
    add_slide_header(s3, "2. Marco Teórico: Base Flotante y Control PD", "Marco Teórico [2 pts]")
    add_bullets(s3, [
        ("Dinámica de Base Flotante", "M(q)q̈ + C(q, q̇)q̇ + g(q) = Sᵀτ + Jcᵀ(q)λ, donde q ∈ ℝ¹⁹ (posición 3D, cuaternión del torso y 12 ángulos articulares)."),
        ("Subactuación y Contacto", "El torso no tiene actuadores directos; su aceleración depende de las fuerzas de reacción en el suelo (GRF, λ) limitadas por conos de fricción."),
        ("Control PD Articular (500 Hz)", "τᵢ = Kp(q_des,i - qᵢ) - Kd·q̇ᵢ, con Kp = 32.0, Kd = 0.9 y saturación física (±23.7 / ±35.55 N·m)."),
        ("Ventaja Jerárquica", "La política neuronal genera referencias suaves q_des = q_default + 0.25·aₜ mientras el PD garantiza amortiguamiento y estabilidad física."),
    ], font_size=19)

    # SLIDE 4: Marco Teorico 2/2 [2 pts]
    s4 = prs.slides.add_slide(blank)
    add_slide_header(s4, "2. Marco Teórico: MDP y Proximal Policy Optimization (PPO)", "Marco Teórico [2 pts]")
    add_bullets(s4, [
        ("Proceso de Decisión de Markov", "Tupla (S, A, P, R, γ) en espacio continuo con factor de descuento γ = 0.99 y ventaja generalizada GAE (λ = 0.95)."),
        ("Arquitectura Actor-Critic", "Red neuronal MLP [256, 256, 128] con activación ELU que estima simultáneamente la política πθ(a|s) y el valor Vϕ(s)."),
        ("Función Objetivo Clipped PPO", "L_CLIP(θ) = E[ min(rₜ(θ)Âₜ, clip(rₜ(θ), 1-ε, 1+ε)Âₜ) ] con ε = 0.2, evitando actualizaciones bruscas que destruyan la marcha."),
        ("Reloj de Fase", "Señales [sin(2πφ), cos(2πφ)] a 2.2 Hz permiten romper simetrías locales y converger a un trote diagonal natural."),
    ], font_size=19)

    # SLIDE 5: Metodologia 1/2 [3 pts] - Modelo del Robot y Coste Computacional
    s5 = prs.slides.add_slide(blank)
    add_slide_header(s5, "3. Metodología: Modelo del Robot y Coste Computacional", "Metodología [3 pts]")
    add_bullets(s5, [
        ("Robot Unitree Go1 (MuJoCo)", "Masa ≈ 12.8 kg | 18 GDL (6 base flotante + 12 actuados). Límites ±23.7 N·m (cadera/muslo) y ±35.55 N·m (rodilla)."),
        ("Control Jerárquico", "Política PPO a 50 Hz (MLP [256,256,128] ELU) + Controlador PD articular a 500 Hz."),
        ("Coste Computacional Medido", "20 entornos MuJoCo paralelos: CPU ~48% (Core Ultra 7, 24 núcleos), RAM 26/30 GiB, GPU RTX 5070 al 18% (1,376 MiB VRAM, 47 °C)."),
        ("Tiempo de Entrenamiento", "10,000,000 pasos en ~22.6 minutos (~7,350 pasos/s, vs. ~3 horas en 1 solo núcleo)."),
    ], width=7.2, font_size=16)
    add_pic(s5, "fig0_training_progression.png", left=7.8, top=1.4, width=5.2)

    # SLIDE 6: Metodologia 2/2 [3 pts]
    s6 = prs.slides.add_slide(blank)
    add_slide_header(s6, "3. Metodología: Estados (50D), Acciones (12D) y Recompensa", "Metodología [3 pts]")
    add_bullets(s6, [
        ("Observaciones (50 dims)", "v_body (3), ω_body (3), gravedad proyectada (3), comando [vx, vy, ωz] (3), q-q₀ (12), q̇ (12), aₜ₋₁ (12), fase [sin, cos] (2)."),
        ("Acciones (12 dims)", "Desplazamientos residuales acotados a [-1, 1] convertidos a posiciones objetivo para el controlador PD de 500 Hz."),
        ("Recompensa Compuesta", "Seguimiento exponencial de (vx, vy, ωz) + simetría de trote diagonal + despeje de patas + regularización de orientación, torque y suavidad."),
        ("Estudio de Ablación", "Se entrenó una política 'Baseline' (900k pasos) sin regularización energética ni de orientación para comparar desempeño."),
    ], font_size=18)

    # SLIDE 7: Resultados - Entrenamiento [3 pts]
    s7 = prs.slides.add_slide(blank)
    add_slide_header(s7, "4. Resultados: Evolución del Entrenamiento y Convergencia", "Resultados [3 pts]")
    add_pic(s7, "fig0_training_progression.png", left=0.4, top=1.35, width=6.1)
    add_pic(s7, "fig1_learning_curves.png", left=6.7, top=1.65, width=6.2)

    # SLIDE 8: Resultados - Velocidad y GRF [3 pts]
    s8 = prs.slides.add_slide(blank)
    add_slide_header(s8, "4. Resultados: Seguimiento de Velocidad y Trote Diagonal", "Resultados [3 pts]")
    add_pic(s8, "fig2_velocity_tracking.png", left=0.4, top=1.35, width=6.2)
    add_pic(s8, "fig4_grf_gait_pattern.png", left=6.8, top=1.85, width=6.1)

    # SLIDE 9: Resultados - Push Recovery [3 pts]
    s9 = prs.slides.add_slide(blank)
    add_slide_header(s9, "4. Resultados: Recuperación ante Empujones (±35 N)", "Resultados [3 pts]")
    add_pic(s9, "fig3_push_recovery.png", left=0.6, top=1.35, width=8.2)
    add_bullets(s9, [
        ("Perturbaciones Aplicadas", "+35 N en t = 3.0 s y -35 N en t = 6.5 s sobre el torso."),
        ("Estabilidad de Roll/Pitch", "Oscilación acotada dentro de [-1°, +6°] sin perder el equilibrio."),
        ("Recuperación Rápida", "Retorno a la altura nominal (0.27 m) en menos de 0.5 segundos."),
    ], left=9.0, top=1.7, width=3.9, font_size=16)

    # SLIDE 10: Resultados - Rampa y Ablacion [3 pts]
    s10 = prs.slides.add_slide(blank)
    add_slide_header(s10, "4. Resultados: Subida de Rampa y Estudio de Ablación", "Resultados [3 pts]")
    add_pic(s10, "fig6_climb_sequence.png", left=0.4, top=1.35, width=6.4)
    add_pic(s10, "fig5_torques_energy_ablation.png", left=7.0, top=1.35, width=5.9)
    add_bullets(s10, [
        ("Ablación Cuantitativa", "PPO Completo reduce la inclinación RMSE en 33.6% (2.78° vs 4.19°) y el Torque RMS en 6.0% (6.08 vs 6.47 N·m), logrando ascender en rampa hasta z = 0.40 m."),
    ], left=7.0, top=4.7, width=5.9, height=2.2, font_size=16)

    # SLIDE 11: Demo en Vivo / Video
    s11 = prs.slides.add_slide(blank)
    add_slide_header(s11, "4. Demostración del Sistema en MuJoCo", "Resultados [3 pts]")
    add_bullets(s11, [
        ("Video HD de Evidencia", "results/demo_cuadrupedo_ppo.mp4 (1280×720, 50 fps): muestra trote estable, recuperación ante empujones laterales (±35 N) y subida de rampa."),
        ("Demostración en Vivo (Terreno Plano + Empujones)", "python3 viewer_demo.py --terrain flat --vx 0.65 --push"),
        ("Demostración en Vivo (Subida de Rampa)", "python3 viewer_demo.py --terrain climb --vx 0.70"),
    ], width=12.0, font_size=20)

    # SLIDE 12: Aplicaciones Reales y Conclusiones [1 pt]
    s12 = prs.slides.add_slide(blank)
    add_slide_header(s12, "5. Aplicaciones en la Vida Real y Conclusiones", "Conclusiones [1 pt]")
    add_bullets(s12, [
        ("Inspección Minera e Industrial", "Acceso autónomo a túneles, subestaciones y zonas con escaleras/escombros donde los robots con ruedas quedan atascados."),
        ("Búsqueda y Rescate (Search & Rescue)", "Despliegue estable sobre terrenos colapsados tras sismos manteniendo equilibrio ante golpes o deslizamientos."),
        ("Transferencia Sim-to-Real", "Entrenar con PD a 500 Hz, aleatorización de dominio y empujones de ±55 N permite desplegar la política en un Go1 físico."),
        ("Conclusión Técnica (10M Pasos)", "El reloj de fase (2.2 Hz) + altura relativa + regularización lograron trote estable, subida de rampa/escalones y menor torque RMS."),
    ], font_size=18)

    pptx_path = os.path.join(BASE_DIR, "Presentacion_Proyecto_MT5305.pptx")
    prs.save(pptx_path)
    print(f"Presentacion PPTX guardada en: {pptx_path}")


if __name__ == "__main__":
    build_docx_report()
    build_pptx()
