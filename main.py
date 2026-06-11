import math
import os
import json
import urllib.request
import urllib.error
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import uvicorn

# Visión por computador para detectar el contorno del cráneo (opcional).
# Si no están instalados, el sistema sigue funcionando con el bbox que estima Claude.
try:
    import base64 as _b64
    import io as _io
    import numpy as _np
    import cv2 as _cv2
    from PIL import Image as _PILImage
    _OPENCV_OK = True
except Exception:
    _OPENCV_OK = False

app = FastAPI(title="OrthoAnalysis - Motor Cefalométrico v2.4")

# =================================================================
# MOTOR MATEMÁTICO v2.4
# - Fix A: NL_NSL = F4 + F7 (con signo, no |F4|)
# - Fix B: AG = F3 - F8 + 90 (F8 con signo, hiperflexión negativa)
# - Fix C: NL/NSLc marcado como NO validado (nslc_validado: false)
# - Fix D: APNI -> APNI_estimado (NO es el APDI real de OrthoTP)
# - Fix E: 33 -> 27 grupos alcanzables; categoria_advertencia si no mapea
# - Fix F: F1, F2, F8 usan calcular_angulo_signed() con la convención
#          OrthoTP validada: F1 = -signed, F2 = +signed, F8 = -signed
# - 27 grupos rotacionales alcanzables de Petrovic-Lavergne
# - Medidas lineales (TM derivado de Co)
# - Lógica basal: D->2, N->1, M->3 (derivada del sagital)
# =================================================================

def calcular_angulo_3_puntos(p1, vertice, p2):
    v1 = (p1[0] - vertice[0], vertice[1] - p1[1])
    v2 = (p2[0] - vertice[0], vertice[1] - p2[1])
    if math.sqrt(v1[0]**2+v1[1]**2) < 1 or math.sqrt(v2[0]**2+v2[1]**2) < 1:
        return None
    ang_v1 = math.atan2(v1[1], v1[0])
    ang_v2 = math.atan2(v2[1], v2[0])
    angulo = math.degrees(ang_v1 - ang_v2)
    if angulo > 180:    angulo -= 360
    elif angulo < -180: angulo += 360
    return round(angulo, 2)

def calcular_angulo_entre_lineas(p1, p2, p3, p4):
    """Ángulo sin signo entre dos líneas (0-90°)"""
    v1 = (p2[0]-p1[0], p1[1]-p2[1])
    v2 = (p4[0]-p3[0], p3[1]-p4[1])
    if math.sqrt(v1[0]**2+v1[1]**2) < 1 or math.sqrt(v2[0]**2+v2[1]**2) < 1:
        return None
    ang_v1 = math.atan2(v1[1], v1[0])
    ang_v2 = math.atan2(v2[1], v2[0])
    angulo = math.degrees(abs(ang_v1 - ang_v2))
    if angulo > 180: angulo = 360 - angulo
    if angulo > 90:  angulo = 180 - angulo
    return round(angulo, 2)

def calcular_angulo_signed(p1, p2, Po, Or):
    """
    Ángulo FIRMADO de la línea p1→p2 con la vertical T (perpendicular a FH).
    Convención geométrica interna: positivo si p2 se desvía hacia +X (anterior)
    respecto a la vertical T. Es robusta ante la inclinación de FH (no depende
    del truco 90-ang_FH ni de comparar coordenadas X sueltas).

    ⚠ NOTA DE SIGNO (validado contra OrthoTP en 3 casos reales):
       OrthoTP usa convenciones por factor que NO siempre coinciden con esta
       función geométrica. El mapeo verificado es:
         F1 (N→A)  = -calcular_angulo_signed(...)
         F2 (A→B)  = +calcular_angulo_signed(...)
         F8 (Co→Go)= -calcular_angulo_signed(...)
    """
    # Vector FH normalizado
    fhx = Or[0] - Po[0]
    fhy = -(Or[1] - Po[1])  # flip Y (matemático)
    fh_len = math.sqrt(fhx**2 + fhy**2)
    if fh_len < 1: return 0.0
    fhx /= fh_len; fhy /= fh_len

    # Vertical T = perpendicular a FH (rotada 90° CCW)
    vtx = -fhy; vty = fhx

    # Vector p1→p2 en coords matemáticas
    dx = p2[0] - p1[0]
    dy = -(p2[1] - p1[1])

    # Ángulo firmado desde vertical T hacia el vector línea
    ang = math.degrees(math.atan2(dx * vty - dy * vtx,
                                   dx * vtx + dy * vty))
    if ang > 90:  ang -= 180
    if ang < -90: ang += 180
    return round(ang, 2)

def proyectar_punto_en_linea(P, L1, L2):
    """Proyecta P perpendicularmente sobre la línea L1-L2. Devuelve (x, y)."""
    dx = L2[0] - L1[0]; dy = L2[1] - L1[1]
    t = ((P[0]-L1[0])*dx + (P[1]-L1[1])*dy) / (dx**2 + dy**2 + 1e-9)
    return (L1[0] + t*dx, L1[1] + t*dy)

def distancia(p1, p2):
    return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

# -----------------------------------------------------------------
# MOTOR PRINCIPAL
# -----------------------------------------------------------------
def calcular_factores_bimler(pts, escala_mm_px=None):
    """
    Calcula todos los factores de Bimler y medidas derivadas.
    escala_mm_px: mm por píxel (para medidas lineales). Si None, en píxeles.
    """
    Po, Or = pts["Po"], pts["Or"]

    # ── Factores angulares base ────────────────────────────────
    SNA = abs(calcular_angulo_3_puntos(pts["S"], pts["N"], pts["A"]))
    SNB = abs(calcular_angulo_3_puntos(pts["S"], pts["N"], pts["B"]))
    ANB = round(SNA - SNB, 2)

    # REGLA: F3, F4, F5, F7 → contra FH (líneas casi horizontales)
    #        F1, F2, F8     → firmados contra VT vía calcular_angulo_signed()
    def _ang_FH(p1,p2):
        return calcular_angulo_entre_lineas(p1,p2,Po,Or)

    # F3: plano mandibular con FH (sin signo)
    F3 = _ang_FH(pts["Me"], pts["Go"])

    # F4: plano palatino con FH — firmado: + si ENA más bajo que ENP
    F4 = round(_ang_FH(pts["ENA"],pts["ENP"]) *
               (1 if pts["ENA"][1] > pts["ENP"][1] else -1), 2)

    # F7: base craneal anterior con FH (sin signo)
    F7 = _ang_FH(pts["N"], pts["S"])

    # ── Fix F: F1, F2, F8 firmados con calcular_angulo_signed() ──
    # Convención OrthoTP validada en 3 casos (Mia/Nicolás/Piero):
    #   F1 = -signed(N,A)  (+ = maxilar prognático)
    #   F2 = +signed(A,B)  (+ = retrogenia / Clase II)
    #   F8 = -signed(Co,Go)(+ = ortoflexión; hiperflexión = negativa, igual que OrthoTP)
    F1 = round(-calcular_angulo_signed(pts["N"],  pts["A"],  Po, Or), 2)
    F2 = round( calcular_angulo_signed(pts["A"],  pts["B"],  Po, Or), 2)
    F8 = round(-calcular_angulo_signed(pts["Co"], pts["Go"], Po, Or), 2)

    # F5: clivus con FH — solo si están marcados Cls y Cli
    F5 = None
    if "Cls" in pts and "Cli" in pts:
        F5 = _ang_FH(pts["Cls"], pts["Cli"])

    # ML/NSL medido y calculado
    ML_NSL  = calcular_angulo_entre_lineas(pts["Me"], pts["Go"], pts["S"], pts["N"])

    # ── Fix A: NL/NSL = F4 + F7 (CON SIGNO, no |F4|) ───────────
    # Antes: |F4| + F7 inflaba NL/NSL cuando F4<0 (Nicolás: 16.71 vs 11.46 real).
    NL_NSL  = round(F4 + F7, 2)

    # ── Ángulos derivados ──────────────────────────────────────
    perfil = round(F1 + F2, 2)                     # Ángulo de Perfil NAB
    ABS    = round(abs(F4) + F5, 2) if F5 is not None else None
    ABI    = round(F3 - abs(F4), 2)                # Basal Inferior F3-|F4|
    ABT    = round(F3 + F5, 2) if F5 is not None else None

    # ── Fix B: AG = F3 - F8 + 90 (F8 con signo) ────────────────
    # Antes: F3 + |F8| + 90 (Mia daba 128.83 vs 118.07 real).
    AG     = round(F3 - F8 + 90, 2)                # Ángulo Gonial

    # ── Fix D: APNI_estimado (NO es el APDI real de OrthoTP) ───
    # El APDI verdadero de OrthoTP = (N-Pg) - F2 + F4 y requiere el punto Pg.
    # Esto es una aproximación interna, NO comparable con APDI ni clase esquelética.
    APNI_estimado = round(F2 + abs(F4), 2)

    ODI    = round(90 - ABI + F2, 2)               # ODI

    # ── TM = proyección de Co sobre FH ────────────────────────
    TM = proyectar_punto_en_linea(pts["Co"], Po, Or)

    # Proyecciones A' y B' sobre FH
    A_prima = proyectar_punto_en_linea(pts["A"], Po, Or)
    B_prima = proyectar_punto_en_linea(pts["B"], Po, Or)

    # Punto T = intersección de vertical desde tuber con FH
    # ⚠ PENDIENTE: T real = fisura pterigomaxilar proyectada en FH.
    #    Usamos Po como aproximación → A'-T NO es profundidad maxilar fiable.
    T = Po  # aproximación: T ≈ Po proyectado sobre FH

    # ── Medidas lineales (en píxeles, convertibles a mm) ──────
    lin = {
        "A_prima_T":   round(distancia(A_prima, T),    1),
        "A_prima_B_prima": round(distancia(A_prima, B_prima), 1),
        "A_prima_TM":  round(distancia(A_prima, TM),   1),
        "B_prima_TM":  round(distancia(B_prima, TM),   1),
        "T_TM":        round(distancia(T, TM),          1),
        "N_S":         round(distancia(pts["N"], pts["S"]), 1),
        "Co_Me":       round(distancia(pts["Co"], pts["Me"]), 1),  # diagonal mandibular
        "Co_Go":       round(distancia(pts["Co"], pts["Go"]), 1),  # altura rama
    }

    result = {
        "SNA": SNA, "SNB": SNB, "ANB": ANB,
        "F1": F1, "F2": F2, "F3": F3, "F4": F4,
        "F5": F5, "F7": F7, "F8": F8,
        "ML_NSL": ML_NSL, "NL_NSL": NL_NSL,
        "perfil": perfil, "ABS": ABS, "ABI": ABI, "ABT": ABT,
        "AG": AG, "APNI_estimado": APNI_estimado, "ODI": ODI,
        "lineales": lin,
    }
    return result

def margenes_borde(T1, T2, T3, tol=0.5):
    """Avisa si T1/T2/T3 están cerca de un umbral del árbol (±0.5°)."""
    avisos = []
    for nombre, val, umbrales in [
        ("T1", T1, [0, 9]), ("T2", T2, [-1, 3]), ("T3", T3, [0, 5]),
    ]:
        for u in umbrales:
            if val is not None and abs(val - u) <= tol:
                avisos.append(f"{nombre}={val} a {abs(val-u):.2f}° del umbral {u} — grupo puede cambiar con variación mínima de marcado.")
    return avisos


def calcular_indicadores_T(f):
    ML_NSLc = round(192 - (2 * f["SNB"]), 2)
    # ── Fix C: NL/NSLc NO validado contra OrthoTP ──────────────
    # La fórmula 0.198*SNA - 4.39 NO reproduce OrthoTP (Nicolás: 11.18 vs 9.31).
    # NL/NSLc NO es función lineal solo de SNA. Se expone marcado como no validado.
    # No afecta el diagnóstico: T1 usa ML/NSLc, y T2 usa NL/NSL MEDIDO (no NL/NSLc).
    NL_NSLc = round(f["ML_NSL"] / 2 - 7, 2)  # Petrovic-Lavergne validado
    T1 = round(ML_NSLc - f["ML_NSL"], 2)
    T2 = round(NL_NSLc - f["NL_NSL"], 2)
    T3 = f["ANB"]
    return T1, T2, T3, ML_NSLc, NL_NSLc

def arbol_decision(T1, T2, T3):
    """
    Árbol de decisión Lavergne-Petrovic.
    Genera el grupo trinomial {rot}{basal} {sag}{vert}

    Rotación (T1):
      A  si T1 > 9   (Anterior — cóndilo rota hacia adelante)
      R  si 0≤T1≤9  (Neutra)
      P  si T1 < 0   (Posterior — cóndilo rota hacia atrás)

    Sagital (T3 = ANB):
      D  si T3 > 5   (Distal — Clase II)
      N  si 0≤T3≤5  (Normal — Clase I)
      M  si T3 < 0   (Mesial — Clase III)

    Basal (derivado de sagital — relación mandíbula/maxila):
      2  si sag=D  (mandíbula < maxila → Clase II)
      1  si sag=N  (iguales → equilibrio)
      3  si sag=M  (mandíbula > maxila → Clase III)
    ⚠ LIMITACIÓN CONOCIDA: el basal se deriva del sagital. En Lavergne-Petrovic
      el basal es un eje independiente (diferencia de crecimiento basal real).
      Por eso el árbol sólo alcanza 27 de los 33 grupos teóricos.

    Vertical (T2):
      OB si T2 > 3   (Mordida Abierta)
      DB si T2 < -1  (Mordida Profunda)
      N  si -1≤T2≤3 (Normal)
    """
    if T1 > 9:    rot = "A"
    elif T1 >= 0: rot = "R"
    else:         rot = "P"

    if T3 > 5:    sag = "D"
    elif T3 >= 0: sag = "N"
    else:         sag = "M"

    # Basal determinado por la relación sagital
    basal = "2" if sag == "D" else ("3" if sag == "M" else "1")

    if T2 > 3:    vert = "OB"
    elif T2 < -1: vert = "DB"
    else:         vert = "N"

    return f"{rot}{basal} {sag}{vert}"

# ── Fix E: 27 grupos ALCANZABLES de Petrovic-Lavergne ─────────
# Se eliminaron las 6 filas inalcanzables por el acoplamiento basal↔sagital:
#   A1 DOB, A1 DN, A1 DDB  (sag=D fuerza basal=2, nunca 1)
#   P1 MOB, P1 MN, P1 MDB  (sag=M fuerza basal=3, nunca 1)
# Si en el futuro el basal se calcula de forma independiente, restituirlas.
GRUPOS_33 = {
    # Categoría 1 — Potencial Muy Bajo (P2D × 3)
    "P2 DOB": 1,  "P2 DN":  1,  "P2 DDB": 1,

    # Categoría 2 — Potencial Bajo (A2D × 3, P1N × 3)
    "A2 DOB": 2,  "A2 DN":  2,  "A2 DDB": 2,
    "P1 NOB": 2,  "P1 NN":  2,  "P1 NDB": 2,

    # Categoría 3 — Potencial Moderado (R2D × 3)
    "R2 DOB": 3,  "R2 DN":  3,  "R2 DDB": 3,

    # Categoría 4 — Potencial Neutro/Alto (R1N × 3)
    "R1 NOB": 4,  "R1 NN":  4,  "R1 NDB": 4,

    # Categoría 5 — Potencial Muy Alto (A1N, R3M × 3) [A1D y P1M eliminados]
    "A1 NOB": 5,  "A1 NN":  5,  "A1 NDB": 5,
    "R3 MOB": 5,  "R3 MN":  5,  "R3 MDB": 5,

    # Categoría 6 — Potencial Excesivo (A3M, P3M × 3)
    "A3 MOB": 6,  "A3 MN":  6,  "A3 MDB": 6,
    "P3 MOB": 6,  "P3 MN":  6,  "P3 MDB": 6,
}

def determinar_categoria(grupo):
    """
    Busca el grupo en la tabla de grupos alcanzables de Petrovic-Lavergne.
    Devuelve (categoria, advertencia):
      categoria   -> int 1-6, o None si no está mapeado
      advertencia -> None, o texto si el grupo no está en la tabla
    """
    cat = GRUPOS_33.get(grupo.strip())
    if cat is None:
        return None, (f"Grupo '{grupo}' no está en la tabla de 27 grupos "
                      f"alcanzables de Petrovic-Lavergne. Revise los puntos "
                      f"o considere que el basal puede requerir cálculo independiente.")
    return cat, None

# -----------------------------------------------------------------
# ENDPOINT: SUGERIR PUNTOS CON IA
# -----------------------------------------------------------------

def detectar_craneo_opencv(image_b64, img_w, img_h):
    """
    Detecta el contorno externo del cráneo con OpenCV y devuelve medidas en
    PÍXELES del espacio comprimido (img_w x img_h). Devuelve None ante cualquier
    fallo o si el resultado no es fiable (fallback silencioso al bbox de Claude).
    """
    if not _OPENCV_OK:
        return None
    try:
        raw = _b64.b64decode(image_b64)
        pil = _PILImage.open(_io.BytesIO(raw)).convert("RGB")
        if pil.size != (img_w, img_h):
            pil = pil.resize((img_w, img_h))
        gray = _cv2.cvtColor(_np.array(pil), _cv2.COLOR_RGB2GRAY)

        if gray.shape[0] < 100 or gray.shape[1] < 100:
            return None

        blur = _cv2.GaussianBlur(gray, (5, 5), 0)

        # Umbrales Canny adaptativos por mediana
        med = float(_np.median(blur))
        lo = int(max(0, 0.66 * med)); hi = int(min(255, 1.33 * med))
        if hi <= lo:
            lo, hi = 50, 150
        edges = _cv2.Canny(blur, lo, hi)

        # Cerrar bordes para un contorno externo continuo
        kernel = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (7, 7))
        closed = _cv2.morphologyEx(edges, _cv2.MORPH_CLOSE, kernel, iterations=2)
        closed = _cv2.dilate(closed, kernel, iterations=1)

        cnts, _h = _cv2.findContours(closed, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        biggest = max(cnts, key=_cv2.contourArea)
        x, y, w, h = _cv2.boundingRect(biggest)
        img_area  = float(img_w * img_h)
        bbox_area = float(w * h)

        # Robustez: el cráneo (bbox) debe ocupar >= 25% del frame
        if bbox_area < 0.25 * img_area:
            return None
        # Robustez: si el 2º contorno tiene bbox de tamaño similar, es ambiguo
        bboxes = sorted((_cv2.boundingRect(cc) for cc in cnts),
                        key=lambda b: b[2] * b[3], reverse=True)
        if len(bboxes) >= 2:
            a0 = bboxes[0][2] * bboxes[0][3]; a1 = bboxes[1][2] * bboxes[1][3]
            if a1 > 0.85 * a0:
                return None
        # Robustez: bbox degenerado o el frame entero (marco/borde)
        if w < 0.35 * img_w or h < 0.35 * img_h:
            return None
        if w >= 0.985 * img_w and h >= 0.985 * img_h and bbox_area >= 0.985 * img_area:
            return None

        pts = biggest.reshape(-1, 2)
        skull_left, skull_right  = int(x), int(x + w)
        skull_top,  skull_bottom = int(y), int(y + h)
        lower = pts[pts[:, 1] >= (img_h * 2 / 3)]
        profile_anterior_x  = int(lower[:, 0].max()) if len(lower) else skull_right
        profile_posterior_x = int(pts[:, 0].min())
        cranium_top_y = int(pts[:, 1].min())
        chin_y        = int(pts[:, 1].max())

        return {
            "skull_left": skull_left, "skull_right": skull_right,
            "skull_top": skull_top,   "skull_bottom": skull_bottom,
            "skull_width_px": int(w), "skull_height_px": int(h),
            "profile_anterior_x": profile_anterior_x,
            "profile_posterior_x": profile_posterior_x,
            "cranium_top_y": cranium_top_y, "chin_y": chin_y,
        }
    except Exception:
        return None

@app.post("/api/sugerir-puntos")
async def sugerir_puntos(request: Request):
    try:
        body      = await request.json()
        image_b64 = body.get("image", "")
        # img_w/h son las dimensiones de la imagen comprimida que Claude ve
        # orig_w/h son las dimensiones originales (para escalar de vuelta en el frontend)
        img_w     = body.get("width",  1000)   # dim comprimida = lo que Claude ve
        img_h     = body.get("height",  800)   # dim comprimida = lo que Claude ve

        if not image_b64:
            return {"success": False, "detail": "No se recibió imagen"}

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return {"success": False, "detail": "ANTHROPIC_API_KEY no configurada en el servidor"}

        # ── Detección real del contorno del cráneo con OpenCV (opcional) ──
        cv_skull = detectar_craneo_opencv(image_b64, img_w, img_h)
        print(f"[OpenCV] skull={'detectado' if cv_skull else 'fallback-Claude'}"
              + (f" bbox=({cv_skull['skull_left']},{cv_skull['skull_top']})-({cv_skull['skull_right']},{cv_skull['skull_bottom']})" if cv_skull else ""))
        if cv_skull:
            contorno_ctx = f"""
════════════════════════════════════════════════════
CONTORNO REAL DEL CRÁNEO detectado por OpenCV
════════════════════════════════════════════════════
Bounding box: x=[{cv_skull['skull_left']}-{cv_skull['skull_right']}], y=[{cv_skull['skull_top']}-{cv_skull['skull_bottom']}]
Ancho del cráneo: {cv_skull['skull_width_px']}px, Alto: {cv_skull['skull_height_px']}px
Perfil anterior (nariz/mentón): x≈{cv_skull['profile_anterior_x']}px
Occipital posterior: x≈{cv_skull['profile_posterior_x']}px
Bóveda craneal: y≈{cv_skull['cranium_top_y']}px
Mentón inferior: y≈{cv_skull['chin_y']}px

Usa estas coordenadas reales como referencia para ubicar los landmarks. Los puntos
de referencia del contorno son medidas reales en píxeles del espacio comprimido.
El bounding box de arriba ES el cráneo: ancla tus x_skull/y_skull a ESE rectángulo.
"""
        else:
            contorno_ctx = ""

        prompt = f"""Eres un especialista en cefalometría de Bimler-Lavergne-Petrovic. Analiza esta telerradiografía lateral de cráneo e identifica con MÁXIMA PRECISIÓN los 13 puntos cefalométricos.
{{contorno_ctx}}
════════════════════════════════════════════════════
PASO 1 — BOUNDING BOX DEL CRÁNEO (hazlo PRIMERO)
════════════════════════════════════════════════════
Antes de colocar cualquier punto, detecta el rectángulo mínimo que contiene
completamente el cráneo óseo (desde la bóveda craneal hasta el mentón,
desde el perfil anterior de la nariz hasta el occipital posterior).
NO incluyas el cuello, el rulero metálico ni las olivas del cefalostato.

Expresa estas 4 coordenadas como PORCENTAJE (%) de las dimensiones de la imagen:
- left_pct  : borde izquierdo del cráneo  (% del ancho)
- right_pct : borde derecho del cráneo    (% del ancho)
- top_pct   : borde superior del cráneo   (% del alto)
- bottom_pct: borde inferior del cráneo   (% del alto, incluye el mentón)

════════════════════════════════════════════════════
PASO 2 — LANDMARKS (relativo al bounding box del cráneo)
════════════════════════════════════════════════════
Ahora coloca cada landmark como porcentaje DENTRO del bounding box del cráneo:
  x_skull = 0  → borde izquierdo del cráneo
  x_skull = 100 → borde derecho del cráneo
  y_skull = 0  → borde superior del cráneo (bóveda)
  y_skull = 100 → borde inferior del cráneo (mentón)

Esto hace que las proporciones sean consistentes sin importar el zoom de la radiografía.

════════════════════════════════════════════════════
OBJETOS A IGNORAR
════════════════════════════════════════════════════
• RULERO / ESCALA METÁLICA: rectángulo con marcas de mm en una esquina.
• CEFALOSTATO / OLIVAS AURICULARES: piezas metálicas simétricas.
Todos los puntos deben caer DENTRO del contorno óseo del cráneo y la mandíbula.

════════════════════════════════════════════════════
DEFINICIÓN DE LANDMARKS (con referencias relativas)
════════════════════════════════════════════════════
Cara mirando a la DERECHA (anterior = mayor x_skull).

S — SELLA: centro de la silla turca (concavidad base craneal media).
   x_skull≈28-35, y_skull≈18-28. POSTERIOR respecto a N.

N — NASION: sutura frontonasal, concavidad ósea frente-nariz.
   x_skull≈52-62, y_skull≈8-18. ANTERIOR y más ARRIBA que Po.
   ERROR FRECUENTE: confundirlo con el rulero. N va en hueso, nunca en metal.

Po — PORION: borde más SUPERIOR del conducto auditivo externo óseo.
   x_skull≈25-35, y_skull≈30-42. Define con Or el plano de Frankfurt.

Or — ORBITARIO: borde más INFERIOR del reborde orbitario.
   x_skull≈52-65, y_skull≈28-40. SIEMPRE más abajo que Po (y_skull(Or) > y_skull(Po)+2).

A — SUBESPINAL: máxima concavidad perfil anterior del maxilar.
   x_skull≈72-85, y_skull≈42-55.

B — SUPRAMENTAL: máxima concavidad perfil anterior mandibular.
   x_skull≈68-82, y_skull≈58-72. Debajo de A, encima de Me.

Me — MENTÓN: punto más INFERIOR de la sínfisis mandibular.
   x_skull≈62-75, y_skull≈88-98. El de mayor y_skull del grupo anterior.

Go — GONION: ESQUINA del ángulo mandibular postero-inferior (bisectriz del ángulo).
   x_skull≈10-25, y_skull≈68-82. NO en mitad de la rama — en el codo del ángulo.
   ERROR FRECUENTE: subirlo sobre la rama. Debe estar en la esquina más postero-inferior.

ENA — ESPINA NASAL ANTERIOR: extremo anterior del paladar óseo.
   x_skull≈70-82, y_skull≈48-58. x_skull(ENA) > x_skull(ENP).

ENP — ESPINA NASAL POSTERIOR: extremo posterior del paladar óseo.
   x_skull≈42-55, y_skull≈48-58.

Co — CONDYLION: punto más postero-superior del cóndilo mandibular.
   x_skull≈15-28, y_skull≈28-42. Por encima y detrás de Go.

Cls — CLIVUS SUPERIOR: parte superior del clivus, justo debajo de S.
   x_skull≈22-32, y_skull≈28-38. Casi alineado verticalmente con S.

Cli — CLIVUS INFERIOR: extremo inferior del clivus (hacia el Basion).
   x_skull≈18-30, y_skull≈38-50. Debajo de Cls en la misma línea.

════════════════════════════════════════════════════
AUTO-VERIFICACIÓN OBLIGATORIA (antes de responder)
════════════════════════════════════════════════════
Verifica y corrige si alguna falla:
1. y_skull(Or) > y_skull(Po) + 2          (Or más bajo que Po)
2. x_skull(N)  > x_skull(S)               (Nasion anterior a Sella)
3. y_skull(Me) > y_skull(B) > y_skull(A)  (orden vertical Me > B > A)
4. x_skull(Go) < x_skull(Me)              (Go posterior al mentón)
5. y_skull(Cls) < y_skull(Cli)            (clivus de arriba a abajo)
6. x_skull(ENA) > x_skull(ENP)            (espina anterior por delante)
7. x_skull(N) > x_skull(S)               (nasion más anterior que sella)

CONFIANZA por punto: 0.9-1.0=nítido, 0.6-0.8=incierto, 0.3-0.5=difícil (típico Cls/Cli/Go).

Responde ÚNICAMENTE con JSON válido (sin texto), con esta estructura exacta:
{{
  "skull_bbox": {{
    "left_pct": 0.0, "right_pct": 0.0,
    "top_pct":  0.0, "bottom_pct": 0.0
  }},
  "landmarks": {{
    "S":   {{"x_skull": 0.0, "y_skull": 0.0, "confianza": 0.0}},
    "N":   {{"x_skull": 0.0, "y_skull": 0.0, "confianza": 0.0}},
    "A":   {{"x_skull": 0.0, "y_skull": 0.0, "confianza": 0.0}},
    "B":   {{"x_skull": 0.0, "y_skull": 0.0, "confianza": 0.0}},
    "Me":  {{"x_skull": 0.0, "y_skull": 0.0, "confianza": 0.0}},
    "Go":  {{"x_skull": 0.0, "y_skull": 0.0, "confianza": 0.0}},
    "ENA": {{"x_skull": 0.0, "y_skull": 0.0, "confianza": 0.0}},
    "ENP": {{"x_skull": 0.0, "y_skull": 0.0, "confianza": 0.0}},
    "Po":  {{"x_skull": 0.0, "y_skull": 0.0, "confianza": 0.0}},
    "Or":  {{"x_skull": 0.0, "y_skull": 0.0, "confianza": 0.0}},
    "Co":  {{"x_skull": 0.0, "y_skull": 0.0, "confianza": 0.0}},
    "Cls": {{"x_skull": 0.0, "y_skull": 0.0, "confianza": 0.0}},
    "Cli": {{"x_skull": 0.0, "y_skull": 0.0, "confianza": 0.0}}
  }}
}}"""

        payload = json.dumps({
            "model": "claude-opus-4-6",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                    "media_type": "image/jpeg", "data": image_b64}},
                {"type": "text", "text": prompt}
            ]}]
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={"x-api-key": api_key,
                     "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            data  = json.loads(response.read().decode("utf-8"))
        texto = data["content"][0]["text"].strip()

        if "```" in texto:
            texto = texto.split("```")[1]
            if texto.startswith("json"): texto = texto[4:]

        crudo = json.loads(texto)

        # Extraer bounding box del cráneo
        bbox = crudo.get("skull_bbox", {})
        lm   = crudo.get("landmarks",  crudo)  # fallback si no tiene estructura nueva

        # Fuente de verdad del bbox: OpenCV si detectó; si no, el que estimó Claude.
        if cv_skull:
            left_px   = float(cv_skull["skull_left"])
            right_px  = float(cv_skull["skull_right"])
            top_px    = float(cv_skull["skull_top"])
            bottom_px = float(cv_skull["skull_bottom"])
        else:
            left_px   = bbox.get("left_pct",   5)  / 100.0 * img_w
            right_px  = bbox.get("right_pct",  95) / 100.0 * img_w
            top_px    = bbox.get("top_pct",    5)  / 100.0 * img_h
            bottom_px = bbox.get("bottom_pct", 95) / 100.0 * img_h
        skull_w   = max(right_px  - left_px,  50)  # mínimo 50px para evitar div/0
        skull_h   = max(bottom_px - top_px,   50)

        puntos     = {}
        confianzas = {}
        for key, v in lm.items():
            if "x_skull" in v and "y_skull" in v:
                # Coordenadas relativas al cráneo → píxeles imagen comprimida
                x = left_px + (float(v["x_skull"]) / 100.0) * skull_w
                y = top_px  + (float(v["y_skull"]) / 100.0) * skull_h
            elif "x_pct" in v and "y_pct" in v:
                # Compatibilidad con formato anterior (% de imagen)
                x = float(v["x_pct"]) / 100.0 * img_w
                y = float(v["y_pct"]) / 100.0 * img_h
            else:
                x = float(v.get("x", 0))
                y = float(v.get("y", 0))
            # Clamp to skull bbox (not just image bounds)
            x_min = left_px if (right_px - left_px) > 50 else 0.0
            y_min = top_px  if (bottom_px - top_px) > 50 else 0.0
            x_max = right_px if (right_px - left_px) > 50 else float(img_w)
            y_max = bottom_px if (bottom_px - top_px) > 50 else float(img_h)
            puntos[key] = {
                "x": round(max(x_min, min(x, x_max)), 1),
                "y": round(max(y_min, min(y, y_max)), 1)
            }
            if "confianza" in v:
                try:
                    confianzas[key] = round(max(0.0, min(1.0, float(v["confianza"]))), 2)
                except (TypeError, ValueError):
                    confianzas[key] = None

        return {"success": True, "puntos": puntos, "confianza": confianzas}

    except json.JSONDecodeError as e:
        return {"success": False, "detail": f"Respuesta IA no válida: {str(e)}"}
    except Exception as e:
        return {"success": False, "detail": str(e)}


# -----------------------------------------------------------------
# ENDPOINT: ANALIZAR
# -----------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/analizar")
async def analizar(request: Request):
    try:
        body = await request.json()
        pts  = {}
        for nombre, coords in body.items():
            if isinstance(coords, dict):   pts[nombre] = (coords["x"], coords["y"])
            elif isinstance(coords, list): pts[nombre] = (coords[0], coords[1])
            else:                          pts[nombre] = tuple(coords)

        requeridos = ["S","N","A","B","Me","Go","ENA","ENP","Po","Or","Co"]
        for p in requeridos:
            if p not in pts:
                return {"success": False, "detail": f"Falta el punto: {p}"}

        factores            = calcular_factores_bimler(pts)
        T1, T2, T3, ML_NSLc, NL_NSLc = calcular_indicadores_T(factores)
        grupo               = arbol_decision(T1, T2, T3)
        categoria, advertencia = determinar_categoria(grupo)   # Fix E

        rot_letra  = grupo[0]
        basal_num  = grupo[1]
        sag_letra  = "D" if " D" in grupo else ("M" if " M" in grupo else "N")
        vert_letra = "OB" if "OB" in grupo else ("DB" if "DB" in grupo else "N")

        rot_map  = {"A":"Anterior","R":"Neutra","P":"Paralelo/Posterior"}
        sag_map  = {"D":"Distoclusión (Clase II)","N":"Normal (Clase I)","M":"Mesioclusión (Clase III)"}
        vert_map = {"OB":"Mordida Abierta","DB":"Mordida Profunda","N":"Normal"}
        basal_map = {"1":"Mandíbula = Maxila","2":"Mandíbula < Maxila (→ Clase II)","3":"Mandíbula > Maxila (→ Clase III)"}

        # Clasificaciones clínicas por factor
        def clasif_F3(v):
            if v < 20: return "Dólico (cara corta)"
            if v > 30: return "Lepto (cara larga)"
            return "Meso (norma)"

        def clasif_F4(v):
            if v > 2:  return "Pro-inclinado (mordida profunda)"
            if v < -2: return "Retro-inclinado (mordida abierta)"
            return "Orto-posición (norma)"

        def clasif_F7(v):
            if v > 9.5: return "Base vertical"
            if v < 5.5: return "Base horizontal"
            return "Neutra (norma)"

        def clasif_ABS(v):
            if v is None: return "—"
            if v < 60: return "Dólico"
            if v > 70: return "Lepto"
            return "Meso (norma)"

        return {
            "success": True,
            "factores_bimler": {
                "SNA": factores["SNA"], "SNB": factores["SNB"], "ANB": factores["ANB"],
                "F1": factores["F1"],   "F2": factores["F2"],
                "F3": factores["F3"],   "F4": factores["F4"],
                "F5": factores["F5"],   "F7": factores["F7"],   "F8": factores["F8"],
                "ML_NSL": factores["ML_NSL"], "NL_NSL": factores["NL_NSL"],
                "clasif_F3": clasif_F3(factores["F3"]),
                "clasif_F4": clasif_F4(factores["F4"]),
                "clasif_F7": clasif_F7(factores["F7"]),
            },
            "angulos_derivados": {
                "perfil": factores["perfil"],
                "ABS": factores["ABS"],
                "ABI": factores["ABI"],
                "ABT": factores["ABT"],
                "AG":  factores["AG"],
                # Fix D: renombrado + nota explícita
                "APNI_estimado": factores["APNI_estimado"],
                "APNI_nota": "Estimación interna (F2+|F4|). NO es el APDI real de OrthoTP, "
                             "que es (N-Pg)-F2+F4 y requiere el punto Pg. No usar para clase esquelética.",
                "ODI":  factores["ODI"],
                "clasif_ABS": clasif_ABS(factores["ABS"]),
            },
            "medidas_lineales": factores["lineales"],
            "indicadores_petrovic": {
                "T1": T1, "T2": T2, "T3": T3,
                "ML_NSLc": ML_NSLc,
                "NL_NSLc": NL_NSLc,
                # Fix C: NL/NSLc no validado contra OrthoTP
                "nslc_validado": False,
                "nslc_nota": "NL/NSLc (0.198*SNA-4.39) NO reproduce OrthoTP y NO es función "
                             "lineal solo de SNA. No afecta el diagnóstico (T2 usa NL/NSL medido).",
            },
            "avisos_limite": margenes_borde(T1, T2, T3),
            "diagnostico": {
                "grupo": grupo, "categoria": categoria,
                "categoria_advertencia": advertencia,   # Fix E: None si mapea OK
                "rotacion": rot_letra, "desc_rotacion": rot_map.get(rot_letra,"—"),
                "basal":    basal_num,  "desc_basal":    basal_map.get(basal_num,"—"),
                "sagital":  sag_letra,  "desc_sagital":  sag_map.get(sag_letra,"—"),
                "vertical": vert_letra, "desc_vertical": vert_map.get(vert_letra,"—"),
            }
        }
    except Exception as e:
        return {"success": False, "detail": str(e)}

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.4",
            "grupos_rotacionales": 27,
            "factores_bimler": 8,
            "casos_validados": 3}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
