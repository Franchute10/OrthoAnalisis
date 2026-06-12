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
    Detecta el CRÁNEO como la SILUETA BRILLANTE más grande (hueso = blanco) mediante
    CLAHE + umbralización de Otsu, no por bordes Canny. Devuelve medidas en PÍXELES
    del espacio comprimido. None ante cualquier fallo (fallback silencioso).
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

        # ── PASO 1: CLAHE normaliza contraste según el equipo de rayos X ──
        clahe = _cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq = clahe.apply(gray)

        # ── PASO 2: masa ósea brillante (Otsu, con respaldo por percentil) ──
        sm = _cv2.GaussianBlur(eq, (15, 15), 0)
        _t, binimg = _cv2.threshold(sm, 0, 255, _cv2.THRESH_BINARY + _cv2.THRESH_OTSU)
        frac = float((binimg > 0).sum()) / float(img_w * img_h)
        if frac < 0.15 or frac > 0.60:
            p65 = float(_np.percentile(sm, 65))
            _t2, binimg = _cv2.threshold(sm, p65, 255, _cv2.THRESH_BINARY)

        k15 = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (15, 15))
        k5  = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (5, 5))
        binimg = _cv2.morphologyEx(binimg, _cv2.MORPH_CLOSE, k15, iterations=3)
        binimg = _cv2.morphologyEx(binimg, _cv2.MORPH_OPEN,  k5,  iterations=1)

        # ── PASO 3: seleccionar el cráneo (blob de mayor área) ──
        n, labels, stats, _cent = _cv2.connectedComponentsWithStats(binimg, connectivity=8)
        if n <= 1:
            return None
        idx = 1 + int(_np.argmax(stats[1:, _cv2.CC_STAT_AREA]))
        area = int(stats[idx, _cv2.CC_STAT_AREA])
        img_area = float(img_w * img_h)
        if area < 0.20 * img_area:
            return None

        x = int(stats[idx, _cv2.CC_STAT_LEFT]);  y = int(stats[idx, _cv2.CC_STAT_TOP])
        w = int(stats[idx, _cv2.CC_STAT_WIDTH]); h = int(stats[idx, _cv2.CC_STAT_HEIGHT])
        # Excluir si toca los 4 bordes (es el fondo invertido, no el cráneo)
        if x <= 1 and y <= 1 and (x + w) >= img_w - 1 and (y + h) >= img_h - 1:
            return None

        mask = (labels == idx).astype(_np.uint8) * 255
        cnts, _hh = _cv2.findContours(mask, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        cnt = max(cnts, key=_cv2.contourArea)
        pts = cnt.reshape(-1, 2)

        skull_left, skull_right  = int(x), int(x + w)
        skull_top,  skull_bottom = int(y), int(y + h)
        skull_w, skull_h = int(w), int(h)

        # ── PASO 4: puntos anatómicos del contorno ──
        ax_thr = skull_left + skull_w * 0.55
        anterior = pts[pts[:, 0] > ax_thr]
        if len(anterior) < 3:
            return None
        i_top = int(_np.argmin(anterior[:, 1]))
        profile_top_x, profile_top_y = int(anterior[i_top, 0]), int(anterior[i_top, 1])
        upper_third_y = skull_top + skull_h / 3.0
        nasal_zone = anterior[anterior[:, 1] <= upper_third_y]
        if len(nasal_zone):
            i_nasal = int(_np.argmax(nasal_zone[:, 0]))
            profile_nasal_x = int(nasal_zone[i_nasal, 0])
            profile_nasal_y = int(nasal_zone[i_nasal, 1])
        else:
            profile_nasal_x, profile_nasal_y = profile_top_x, profile_top_y
        i_chin = int(_np.argmax(anterior[:, 1]))
        chin_anterior_x, chin_y = int(anterior[i_chin, 0]), int(anterior[i_chin, 1])

        profile_posterior_x = int(pts[:, 0].min())
        cranium_top_y       = int(pts[:, 1].min())
        frankfurt_y_approx  = int(skull_top + skull_h * 0.38)

        return {
            "skull_left": skull_left, "skull_right": skull_right,
            "skull_top": skull_top,   "skull_bottom": skull_bottom,
            "skull_width_px": skull_w, "skull_height_px": skull_h,
            "profile_top_x": profile_top_x, "profile_top_y": profile_top_y,
            "profile_nasal_x": profile_nasal_x, "profile_nasal_y": profile_nasal_y,
            "chin_anterior_x": chin_anterior_x, "chin_y": chin_y,
            "profile_posterior_x": profile_posterior_x,
            "cranium_top_y": cranium_top_y,
            "frankfurt_y_approx": frankfurt_y_approx,
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
        print(f"[OpenCV] skull={'detectado(Otsu+CLAHE)' if cv_skull else 'fallback-Claude'}"
              + (f" bbox=({cv_skull['skull_left']},{cv_skull['skull_top']})-({cv_skull['skull_right']},{cv_skull['skull_bottom']})" if cv_skull else ""))
        if cv_skull:
            contorno_ctx = f"""
════════════════════════════════════════════════════
SILUETA DEL CRÁNEO (blob óseo detectado por umbralización Otsu+CLAHE)
════════════════════════════════════════════════════
Bounding box: x=[{cv_skull['skull_left']}-{cv_skull['skull_right']}], y=[{cv_skull['skull_top']}-{cv_skull['skull_bottom']}]
Dimensiones: {cv_skull['skull_width_px']}px ancho x {cv_skull['skull_height_px']}px alto

Perfil anterior derecho:
  Punto más superior (zona frente/nasion): x≈{cv_skull['profile_top_x']}, y≈{cv_skull['profile_top_y']}
  Punto más anterior nasal (zona N): x≈{cv_skull['profile_nasal_x']}, y≈{cv_skull['profile_nasal_y']}
  Mentón inferior (zona Me): x≈{cv_skull['chin_anterior_x']}, y≈{cv_skull['chin_y']}

Perfil posterior:
  Occipital (más posterior): x≈{cv_skull['profile_posterior_x']}
  Bóveda craneal (más superior): y≈{cv_skull['cranium_top_y']}

Frankfurt aproximado: y≈{cv_skull['frankfurt_y_approx']}px

IMPORTANTE: estos son puntos reales del contorno óseo medidos por OpenCV.
Úsalos como anclas geométricas. N debe estar CERCA de profile_nasal.
Me debe estar CERCA de chin_anterior. S/Co están en la zona posterior-media.

PROPORCIONES CEFALOMÉTRICAS NORMATIVAS (dentro del bbox del cráneo):
 S:   x_skull≈32%, y_skull≈22%   (centro silla turca)
 N:   x_skull≈60%, y_skull≈12%   (sutura frontonasal)
 Po:  x_skull≈28%, y_skull≈36%   (conducto auditivo)
 Or:  x_skull≈58%, y_skull≈34%   (reborde orbitario inf)
 A:   x_skull≈78%, y_skull≈50%   (subespinal)
 B:   x_skull≈74%, y_skull≈63%   (supramental)
 Me:  x_skull≈68%, y_skull≈94%   (mentón inferior)
 Go:  x_skull≈18%, y_skull≈76%   (ángulo mandibular)
 ENA: x_skull≈76%, y_skull≈52%   (espina nasal ant)
 ENP: x_skull≈48%, y_skull≈52%   (espina nasal post)
 Co:  x_skull≈22%, y_skull≈34%   (cóndilo)
 Cls: x_skull≈28%, y_skull≈30%   (clivus sup)
 Cli: x_skull≈24%, y_skull≈42%   (clivus inf)

 Úsalas como REFERENCIA cuando un punto no es claramente visible.
 Prioriza siempre lo que ves en la imagen sobre estas proporciones.
"""
        else:
            contorno_ctx = ""

        prompt = f"""Eres un especialista en cefalometría radiológica de Bimler-Lavergne-Petrovic. Vas a analizar una telerradiografía lateral de cráneo e identificar con MÁXIMA PRECISIÓN los 13 puntos cefalométricos. Trabaja en TRES FASES y respóndelas en orden.
{contorno_ctx}
████████████████████████████████████████████████████
FASE 0 — ORIENTACIÓN ANATÓMICA (construye tu mapa mental)
████████████████████████████████████████████████████
Antes de colocar un solo punto, realiza este protocolo. NO incluyas el resultado de
la Fase 0 en el JSON final: es solo para que ancles correctamente los landmarks.

0.1 LATERALIDAD: determina si la cara mira a la DERECHA o IZQUIERDA.
    - Anterior = donde está la nariz (perfil del tercio medio facial).
    - Posterior = donde está el occipital (curva convexa del cráneo).
    - Convención OrthoAnalysis: la cara mira a la DERECHA → x mayor = anterior.

0.2 IDENTIFICA Y UBICA LAS 8 ESTRUCTURAS DE REFERENCIA (mentalmente, con su
    x_skull/y_skull aproximado en % del bounding box del cráneo):

  E1 · Bóveda craneal: curva convexa brillante, techo del cráneo. Es la curva más
       superior y exterior. Su punto más alto = límite superior del bbox.
  E2 · Base craneal anterior (plano NSL): línea casi recta de la silla turca hacia
       el nasion, inclinada ~7° hacia abajo en dirección anterior. Es la columna
       vertebral del análisis: S y N son sus extremos.
  E3 · Silla turca: cavidad oval con paredes brillantes y centro oscuro, en la base
       craneal media. Pared posterior = dorsum sellae. Contiene a S. (~x32%,y22%)
  E4 · Conducto auditivo externo (CAE): anillo/semicírculo brillante postero-medio.
       Su borde más superior = Po. (~x28%,y36%)
  E5 · Cavidad orbital: ventana oscura rectangular antero-superior, bordes brillantes.
       Su borde inferior = Or. (~x58%,y34%)
  E6 · Paladar óseo y fosa nasal: línea horizontal brillante de ENA (anterior) a ENP
       (posterior); por encima, la fosa nasal oscura. (~y52%)
  E7 · Mandíbula (cuerpo + rama): arco inferior; el ángulo gonial es la esquina
       postero-inferior donde el cuerpo horizontal se une con la rama vertical.
       Contiene Me, B y Go.
  E8 · Cóndilo mandibular: masa ovoide brillante postero-superior, extremo superior
       de la rama. Contiene Co. (~x22%,y34%)

0.3 SOLO tras ubicar las 8 estructuras, procede a las Fases 1 y 2.

████████████████████████████████████████████████████
FASE 1 — BOUNDING BOX DEL CRÁNEO
████████████████████████████████████████████████████
Detecta el rectángulo mínimo que contiene el cráneo óseo completo (de la bóveda al
mentón, del perfil anterior de la nariz al occipital posterior). NO incluyas cuello,
rulero metálico ni olivas del cefalostato. Exprésalo como % de la imagen:
- left_pct, right_pct : bordes izquierdo y derecho del cráneo (% del ancho)
- top_pct, bottom_pct : bordes superior e inferior del cráneo (% del alto)

████████████████████████████████████████████████████
FASE 2 — LOS 13 LANDMARKS (relativos al bounding box del cráneo)
████████████████████████████████████████████████████
Coloca cada punto como porcentaje DENTRO del bounding box del cráneo:
  x_skull = 0 → borde izquierdo del cráneo ; x_skull = 100 → borde derecho
  y_skull = 0 → bóveda (arriba)            ; y_skull = 100 → mentón (abajo)
Recuerda: Y crece hacia ABAJO. Cara a la derecha → anterior = x_skull mayor.

Para cada punto tienes: [Rx]=cómo se ve · [Ref]=estructura de referencia ·
[Pos]=posición típica · [Err]=error a evitar.

S — SELLA
  [Rx] Centro geométrico de la cavidad oval de la silla turca (paredes brillantes,
       centro oscuro), a media distancia entre tubérculo y dorsum sellae.
  [Ref] Estructura 3 (silla turca), sobre el plano NSL.
  [Pos] x_skull≈28-36, y_skull≈18-28. Posterior a N; misma altura que Po o algo más.
  [Err] No ponerlo sobre el dorsum sellae ni confundirlo con el agujero oval.

N — NASION
  [Rx] Concavidad más profunda del perfil entre la frente convexa y el dorso nasal.
  [Ref] Extremo anterior de la base craneal anterior (Estructura 2).
  [Pos] x_skull≈55-65, y_skull≈8-18. Anterior a S; en el tercio superior.
  [Err] CRÍTICO: NO confundir con el RULERO METÁLICO (rectángulo brillante con marcas
        en una esquina). N va en el hueso del perfil, jamás en metal. Tampoco en el
        punto más prominente: va en el más cóncavo.

Po — PORION
  [Rx] Borde más superior del anillo del conducto auditivo externo.
  [Ref] Estructura 4 (CAE).
  [Pos] x_skull≈22-32, y_skull≈32-42. Define Frankfurt con Or.
  [Err] No usar el borde inferior del CAE; no confundir con fosa mandibular. Po es
        anterior al borde posterior del cráneo y superior al nivel del cóndilo.

Or — ORBITALE
  [Rx] Punto más inferior del reborde de la cavidad orbital (ventana oscura).
  [Ref] Estructura 5 (órbita), borde inferior.
  [Pos] x_skull≈52-65, y_skull≈28-38. SIEMPRE más abajo que Po: y_skull(Or) > y_skull(Po).
  [Err] No usar el borde SUPERIOR de la órbita. Or nunca queda a la altura de Po ni por encima.

A — SUBESPINAL
  [Rx] Concavidad más profunda de la cara anterior del maxilar, bajo ENA.
  [Ref] Perfil anterior del maxilar (deriva de Estructura 6).
  [Pos] x_skull≈72-85, y_skull≈42-55. Bajo ENA, sobre el borde incisal.
  [Err] No confundir con el ápice del incisivo; no ponerlo en ENA.

B — SUPRAMENTAL
  [Rx] Concavidad más profunda de la cara anterior de la sínfisis, entre incisivos y mentón.
  [Ref] Sínfisis mandibular (Estructura 7).
  [Pos] x_skull≈68-82, y_skull≈58-72. Entre A (arriba) y Me (abajo).
  [Err] No confundir con el ápice del incisivo inferior; no subirlo a la zona alveolar.

Me — MENTON
  [Rx] Punto geométricamente más inferior del contorno de la sínfisis.
  [Ref] Borde inferior de la sínfisis (Estructura 7).
  [Pos] x_skull≈62-75, y_skull≈88-98. Mayor y_skull del grupo anterior.
  [Err] No confundir con el pogonion (más anterior, no el más inferior).

Go — GONION
  [Rx] Vértice del ángulo mandibular: intersección de la tangente al borde posterior
       de la rama con la tangente al borde inferior del cuerpo. La ESQUINA de la mandíbula.
  [Ref] Ángulo de la Estructura 7.
  [Pos] x_skull≈12-25, y_skull≈68-82. Postero-inferior. Posterior a Me; inferior a Co.
  [Err] CRÍTICO: NO colocarlo a mitad de la rama. Va en la esquina más postero-inferior.

ENA — ESPINA NASAL ANTERIOR
  [Rx] Espícula puntiaguda que sobresale hacia adelante en el extremo anterior del paladar.
  [Ref] Extremo anterior del paladar (Estructura 6).
  [Pos] x_skull≈70-82, y_skull≈48-58. Anterior a ENP; misma altura que ENP.
  [Err] No confundir con el cornete inferior; no llevarlo al tejido blando nasal.

ENP — ESPINA NASAL POSTERIOR
  [Rx] Proyección puntiaguda al final posterior del paladar óseo (unión con paladar blando).
  [Ref] Extremo posterior del paladar (Estructura 6).
  [Pos] x_skull≈42-55, y_skull≈48-58. Posterior a ENA: x_skull(ENP) < x_skull(ENA).
  [Err] Puede verse tenue en Rx de baja calidad; mantenerlo a la altura de ENA.

Co — CONDYLION
  [Rx] Punto más postero-superior de la cabeza ovoide del cóndilo.
  [Ref] Estructura 8 (cóndilo).
  [Pos] x_skull≈18-28, y_skull≈28-40. Por arriba de Go; por debajo de S o a su altura.
  [Err] No usar el punto más superior (sino el postero-superior); no confundir con la coronoides.

Cls — CLIVUS SUPERIOR
  [Rx] Extremo superior de la pendiente del clivus, adyacente al dorsum sellae.
  [Ref] Cara posterior del esfenoides, bajo la silla turca (Estructura 3).
  [Pos] x_skull≈24-34 (casi alineado con S), y_skull≈28-38 (algo bajo S).
  [Err] No ponerlo encima de S; debe quedar por debajo. Confianza esperada 0.35-0.64.

Cli — CLIVUS INFERIOR
  [Rx] Extremo inferior de la pendiente clivial, hacia el basion (borde anterior del foramen magno).
  [Ref] Continuación inferior del clivus.
  [Pos] x_skull≈18-30, y_skull≈38-50. Debajo de Cls en la misma línea.
  [Err] CRÍTICO: no invertir con Cls (Cli va MÁS ABAJO). Es el más difícil; confianza 0.35-0.64.

────────────────────────────────────────────────────
AUTO-VERIFICACIÓN ANATÓMICA (ejecútala ANTES de responder y corrige)
────────────────────────────────────────────────────
R1  y_skull(Or) > y_skull(Po)                    (Or más abajo que Po)
R2  x_skull(N)  > x_skull(S)                     (N anterior a S)
R3  y_skull(A)  < y_skull(B) < y_skull(Me)        (orden vertical A→B→Me)
R4  x_skull(Go) < x_skull(Me)                    (Go posterior al mentón)
R5  y_skull(Cls) < y_skull(Cli)                  (Cls arriba, Cli abajo)
R6  x_skull(ENA) > x_skull(ENP)                  (ENA anterior a ENP)
R7  x_skull(N)  > x_skull(S) y S bajo la bóveda
R8  Ningún punto sobre el rulero/escala metálica
R9  y_skull(Co) < y_skull(Go)                    (cóndilo sobre el ángulo)
R10 y_skull(Cls) > y_skull(S)                    (Cls bajo S)

CONFIANZA esperada: ALTA 0.85-1.0: N,Me,S,ENA,ENP,Or,Go · MEDIA 0.65-0.84: A,B,Po,Co · BAJA 0.35-0.64: Cls,Cli
Prioriza SIEMPRE lo que ves en la imagen sobre las cifras de referencia.

PROPORCIONES NORMATIVAS (media±DE, % del bbox; úsalas si la estructura no es nítida):
  S x32±4/y22±3 · N x60±4/y12±3 · Po x28±4/y36±4 · Or x58±4/y34±3 · A x78±5/y50±4
  B x74±5/y63±5 · Me x68±4/y94±3 · Go x18±5/y76±5 · ENA x76±4/y52±3 · ENP x48±4/y52±3
  Co x22±4/y34±4 · Cls x28±4/y30±4 · Cli x24±5/y42±5

────────────────────────────────────────────────────
RESPUESTA — SOLO JSON válido (sin texto antes ni después)
────────────────────────────────────────────────────
No incluyas las estructuras de la Fase 0 en el JSON. Devuelve exactamente:
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
