import math
import os
import json
import urllib.request
import urllib.error
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="OrthoAnalysis - Motor Cefalométrico v2.3")

# =================================================================
# MOTOR MATEMÁTICO v2.3
# - 33 grupos rotacionales completos de Petrovic-Lavergne
# - 8 factores de Bimler (F1,F2,F3,F4,F5,F7,F8 + derivados)
# - Medidas lineales (TM derivado de Co)
# - Lógica basal corregida: D→2, N→1, M→3
# =================================================================

def calcular_angulo_3_puntos(p1, vertice, p2):
    v1 = (p1[0] - vertice[0], vertice[1] - p1[1])
    v2 = (p2[0] - vertice[0], vertice[1] - p2[1])
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
    ang_v1 = math.atan2(v1[1], v1[0])
    ang_v2 = math.atan2(v2[1], v2[0])
    angulo = math.degrees(abs(ang_v1 - ang_v2))
    if angulo > 180: angulo = 360 - angulo
    if angulo > 90:  angulo = 180 - angulo
    return round(angulo, 2)

def calcular_angulo_signed(p1, p2, Po, Or):
    """
    Ángulo FIRMADO de la línea p1→p2 con la vertical T (perpendicular a FH).
    Positivo: p2 está anterior (prognático) respecto a p1.
    Negativo: p2 está posterior (retrognático).
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
    #        F1, F2, F8     → contra VT (líneas casi verticales = 90-FH)
    def _ang_FH(p1,p2):
        return calcular_angulo_entre_lineas(p1,p2,Po,Or)
    def _ang_VT(p1,p2):
        return round(90 - _ang_FH(p1,p2), 2)

    # F3: plano mandibular con FH (sin signo)
    F3 = _ang_FH(pts["Me"], pts["Go"])

    # F4: plano palatino con FH — firmado: + si ENA más bajo que ENP
    F4 = round(_ang_FH(pts["ENA"],pts["ENP"]) *
               (1 if pts["ENA"][1] > pts["ENP"][1] else -1), 2)

    # F7: base craneal anterior con FH (sin signo)
    F7 = _ang_FH(pts["N"], pts["S"])

    # F8: rama mandibular con VT — firmado: + si Go más anterior (mayor X) que Co
    F8 = round(_ang_VT(pts["Co"],pts["Go"]) *
               (1 if pts["Go"][0] > pts["Co"][0] else -1), 2)

    # F1: N-A con VT — firmado: + si A anterior a N (mayor X = prognático)
    F1 = round(_ang_VT(pts["N"],pts["A"]) *
               (1 if pts["A"][0] > pts["N"][0] else -1), 2)

    # F2: A-B con VT — firmado: + si B posterior a A (menor X = retrogenia Cl.II)
    F2 = round(_ang_VT(pts["A"],pts["B"]) *
               (1 if pts["B"][0] < pts["A"][0] else -1), 2)

    # F5: clivus con FH — solo si están marcados Cls y Cli
    F5 = None
    if "Cls" in pts and "Cli" in pts:
        F5 = _ang_FH(pts["Cls"], pts["Cli"])

    # ML/NSL medido y calculado
    ML_NSL  = calcular_angulo_entre_lineas(pts["Me"], pts["Go"], pts["S"], pts["N"])
    NL_NSL  = round(abs(F4) + F7, 2)

    # ── Ángulos derivados ──────────────────────────────────────
    perfil = round(F1 + F2, 2)                     # Ángulo de Perfil NAB
    ABS    = round(abs(F4) + (F5 or 0), 2)         # Basal Superior F4+F5
    ABI    = round(F3 - abs(F4), 2)                # Basal Inferior F3-|F4|
    ABT    = round(F3 + (F5 or 0), 2)              # Basal Total F3+F5
    AG     = round(F3 + abs(F8) + 90, 2)           # Ángulo Gonial
    APNI   = round(F2 + abs(F4), 2)                # APNI = F2+F4
    ODI    = round(90 - ABI + F2, 2)               # ODI

    # ── TM = proyección de Co sobre FH ────────────────────────
    TM = proyectar_punto_en_linea(pts["Co"], Po, Or)

    # Proyecciones A' y B' sobre FH
    A_prima = proyectar_punto_en_linea(pts["A"], Po, Or)
    B_prima = proyectar_punto_en_linea(pts["B"], Po, Or)

    # Punto T = intersección de vertical desde tuber con FH
    # (Usamos Po como referencia para T en nuestro sistema)
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
        "AG": AG, "APNI": APNI, "ODI": ODI,
        "lineales": lin,
    }
    return result

def calcular_indicadores_T(f):
    ML_NSLc = round(192 - (2 * f["SNB"]), 2)
    NL_NSLc = round(0.198 * f["SNA"] - 4.39, 2)
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

# 33 grupos rotacionales de Petrovic-Lavergne
# 11 tipos base × 3 variantes verticales (OB/N/DB)
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

    # Categoría 5 — Potencial Muy Alto (A1D, A1N, P1M, R3M × 3)
    "A1 DOB": 5,  "A1 DN":  5,  "A1 DDB": 5,
    "A1 NOB": 5,  "A1 NN":  5,  "A1 NDB": 5,
    "P1 MOB": 5,  "P1 MN":  5,  "P1 MDB": 5,
    "R3 MOB": 5,  "R3 MN":  5,  "R3 MDB": 5,

    # Categoría 6 — Potencial Excesivo (A3M, P3M × 3)
    "A3 MOB": 6,  "A3 MN":  6,  "A3 MDB": 6,
    "P3 MOB": 6,  "P3 MN":  6,  "P3 MDB": 6,
}

def determinar_categoria(grupo):
    """Busca el grupo en la tabla de 33 grupos de Petrovic-Lavergne."""
    return GRUPOS_33.get(grupo.strip(), "—")

# -----------------------------------------------------------------
# ENDPOINT: SUGERIR PUNTOS CON IA
# -----------------------------------------------------------------
@app.post("/api/sugerir-puntos")
async def sugerir_puntos(request: Request):
    try:
        body      = await request.json()
        image_b64 = body.get("image", "")
        img_w     = body.get("width", 1000)
        img_h     = body.get("height", 800)

        if not image_b64:
            return {"success": False, "detail": "No se recibió imagen"}

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return {"success": False, "detail": "ANTHROPIC_API_KEY no configurada en el servidor"}

        prompt = f"""Eres un especialista en cefalometría de Bimler-Lavergne-Petrovic. Analiza esta telerradiografía lateral de cráneo e identifica con MÁXIMA PRECISIÓN los 13 puntos cefalométricos.

Dimensiones de imagen: {img_w}px ancho × {img_h}px alto. Eje Y crece hacia ABAJO.

═══════════════════════════════════════
ADVERTENCIA GLOBAL — LEE ANTES DE MARCAR
═══════════════════════════════════════
⚠️ OBJETOS A IGNORAR COMPLETAMENTE:
• RULERO / ESCALA METÁLICA: objeto rectangular con marcas de mm visible en las esquinas. NO es anatomía.
• SOPORTE DE CABEZA / CEFALOSTATO: estructura metálica que sujeta la cabeza. NO es anatomía.
• ARTEFACTOS METÁLICOS: cualquier objeto brillante/rectangular fuera del contorno óseo del cráneo.
Todos los puntos deben estar DENTRO del contorno óseo del cráneo y la mandíbula.

═══════════════════════════════════════
INSTRUCCIONES CRÍTICAS PUNTO POR PUNTO
═══════════════════════════════════════

S — SELLA TURCA:
• Centro geométrico de la fosa pituitaria (concavidad ósea en la base del cráneo)
• Debe estar bien POSTERIOR, aproximadamente sobre la vertical del CAE
• ERROR: marcarlo demasiado anterior. Verificar: S debe estar claramente a la izquierda de N en imagen lateral derecha

N — NASION:
• Intersección de la sutura frontonasal — en la CONCAVIDAD entre frente y nariz
• Debe estar DENTRO del cráneo, en la depresión ósea frente-nariz
• ERROR CRÍTICO: confundirlo con el RULERO METÁLICO de la esquina de la radiografía
• Si ves escala/rulero en esquina superior derecha → N NO va allí, va en la sutura ósea frente-nariz

Or — ORBITARIO:
• Punto MÁS INFERIOR del reborde orbitario óseo
• CRÍTICO: Or.y debe ser > Po.y + 15px (Or claramente más bajo que Po)
• La línea Frankfurt (Po→Or) tiene ~7-10° de inclinación respecto a S-N
• Si Or y Po tienen la misma altura → POSICIÓN INCORRECTA

Po — PORION:
• Punto más SUPERIOR del conducto auditivo externo óseo

A — Punto A (Subespinal): máxima concavidad del perfil anterior del maxilar superior
B — Punto B (Supramental): máxima concavidad del perfil anterior mandibular
Me — Mentón: punto más INFERIOR de la sínfisis
Go — Gonion: ángulo mandibular postero-inferior (bisectriz de tangentes)
ENA — Espina Nasal Anterior: extremo más anterior del paladar
ENP — Espina Nasal Posterior: extremo posterior del paladar óseo
Co — Condylion: punto más postero-superior del cóndilo mandibular

Cls — CLIVUS SUPERIOR:
• Punto en el clivus (superficie posterior de la silla turca / cuerpo del esfenoides)
• Se ubica aproximadamente 10mm POR DEBAJO del centro de la silla turca (S)
• Es la parte superior del plano inclinado posterior de la base craneal

Cli — CLIVUS INFERIOR:
• Punto inferior del clivus, aproximadamente 10mm POR ENCIMA del Basion
• El Basion es la punta más inferior y anterior del hueso occipital (donde termina el clivus)

═══════════════════════════════════════
VALIDACIÓN ANTES DE RESPONDER:
═══════════════════════════════════════
1. ¿Or.y > Po.y + 15? → Si no, corrige Or hacia abajo
2. ¿N está en sutura ósea frente-nariz (NO en el rulero metálico)? → Si está en el rulero, corrígelo
3. ¿S está bien posterior en la base del cráneo? → Si está muy anterior, muévelo hacia atrás
4. ¿Cls está entre S y Cli, en la cara posterior de la silla turca? → Verificar secuencia S→Cls→Cli

Responde ÚNICAMENTE con JSON válido:
{{
  "S":   {{"x": 0, "y": 0}},
  "N":   {{"x": 0, "y": 0}},
  "A":   {{"x": 0, "y": 0}},
  "B":   {{"x": 0, "y": 0}},
  "Me":  {{"x": 0, "y": 0}},
  "Go":  {{"x": 0, "y": 0}},
  "ENA": {{"x": 0, "y": 0}},
  "ENP": {{"x": 0, "y": 0}},
  "Po":  {{"x": 0, "y": 0}},
  "Or":  {{"x": 0, "y": 0}},
  "Co":  {{"x": 0, "y": 0}},
  "Cls": {{"x": 0, "y": 0}},
  "Cli": {{"x": 0, "y": 0}}
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

        puntos = json.loads(texto)
        return {"success": True, "puntos": puntos}

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
        categoria           = determinar_categoria(grupo)

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
                "APNI": factores["APNI"],
                "ODI":  factores["ODI"],
                "clasif_ABS": clasif_ABS(factores["ABS"]),
            },
            "medidas_lineales": factores["lineales"],
            "indicadores_petrovic": {
                "T1": T1, "T2": T2, "T3": T3,
                "ML_NSLc": ML_NSLc, "NL_NSLc": NL_NSLc,
            },
            "diagnostico": {
                "grupo": grupo, "categoria": categoria,
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
    return {"status": "ok", "version": "2.3",
            "grupos_rotacionales": 33,
            "factores_bimler": 8,
            "casos_validados": 3}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
