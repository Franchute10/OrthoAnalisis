import math
import os
import json
import urllib.request
import urllib.error
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="OrthoAnalysis - Motor Cefalométrico v2.2")

# =================================================================
# MOTOR MATEMÁTICO — v2.1 (validado con 3 casos reales)
# =================================================================

def calcular_angulo_3_puntos(p1, vertice, p2):
    v1 = (p1[0] - vertice[0], vertice[1] - p1[1])
    v2 = (p2[0] - vertice[0], vertice[1] - p2[1])
    ang_v1 = math.atan2(v1[1], v1[0])
    ang_v2 = math.atan2(v2[1], v2[0])
    angulo = math.degrees(ang_v1 - ang_v2)
    if angulo > 180:   angulo -= 360
    elif angulo < -180: angulo += 360
    return round(angulo, 2)

def calcular_angulo_entre_lineas(p1, p2, p3, p4):
    v1 = (p2[0] - p1[0], p1[1] - p2[1])
    v2 = (p4[0] - p3[0], p3[1] - p4[1])
    ang_v1 = math.atan2(v1[1], v1[0])
    ang_v2 = math.atan2(v2[1], v2[0])
    angulo = math.degrees(abs(ang_v1 - ang_v2))
    if angulo > 180: angulo = 360 - angulo
    if angulo > 90:  angulo = 180 - angulo
    return round(angulo, 2)

def calcular_factores_bimler(pts):
    SNA = abs(calcular_angulo_3_puntos(pts["S"], pts["N"], pts["A"]))
    SNB = abs(calcular_angulo_3_puntos(pts["S"], pts["N"], pts["B"]))
    ANB = round(SNA - SNB, 2)
    F3  = calcular_angulo_entre_lineas(pts["Me"],  pts["Go"],  pts["Po"], pts["Or"])
    F4  = calcular_angulo_entre_lineas(pts["ENA"], pts["ENP"], pts["Po"], pts["Or"])
    F7  = calcular_angulo_entre_lineas(pts["N"],   pts["S"],   pts["Po"], pts["Or"])
    ML_NSL = calcular_angulo_entre_lineas(pts["Me"], pts["Go"], pts["S"], pts["N"])
    NL_NSL = round(F4 + F7, 2)
    return {"SNA": SNA, "SNB": SNB, "ANB": ANB,
            "F3": F3, "F4": F4, "F7": F7,
            "ML_NSL": ML_NSL, "NL_NSL": NL_NSL}

def calcular_indicadores_T(f):
    ML_NSLc = round(192 - (2 * f["SNB"]), 2)
    NL_NSLc = round(0.198 * f["SNA"] - 4.39, 2)
    T1 = round(ML_NSLc - f["ML_NSL"], 2)
    T2 = round(NL_NSLc - f["NL_NSL"], 2)
    T3 = f["ANB"]
    return T1, T2, T3, ML_NSLc, NL_NSLc

def arbol_decision(T1, T2, T3):
    if T1 > 9:    rot = "A"
    elif T1 >= 0: rot = "R"
    else:         rot = "P"
    if T3 > 5:    sag = "D"
    elif T3 >= 0: sag = "N"
    else:         sag = "M"
    if T2 > 3:    vert = "OB"
    elif T2 < -1: vert = "DB"
    else:         vert = "N"
    num   = "2" if T1 > 13 else "1"
    grupo = f"{rot}{num} {sag}N" if vert == "N" else f"{rot}{num} {sag}{vert}"
    return grupo

def determinar_categoria(grupo):
    tabla = {
        "R1 NOB": 1, "R2 NOB": 1,
        "R2 DOB": 2, "R3 MOB": 2, "R1 DOB": 2,
        "R2 DN":  3, "R1 NN":  3, "R3 MN":  3, "R1 DN": 3,
        "R1 NDB": 4, "R2 DDB": 4, "R3 MDB": 4,
        "P1 NDB": 4, "P1 NN":  4, "P1 DDB": 4,
        "A1 NDB": 5, "A1 DDB": 5, "A1 NN":  5,
        "A2 DDB": 6, "A3 MDB": 6,
    }
    g = grupo.replace(" ", "")
    for key, cat in tabla.items():
        if key.replace(" ", "") == g:
            return cat
    return "—"

# =================================================================
# ENDPOINT: SUGERIR PUNTOS CON IA
# =================================================================
@app.post("/api/sugerir-puntos")
async def sugerir_puntos(request: Request):
    """
    Recibe imagen base64 de la radiografía lateral y devuelve
    coordenadas sugeridas para los 11 puntos cefalométricos.
    """
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

        prompt = f"""Eres un especialista en cefalometría de Bimler-Lavergne-Petrovic. Analiza esta telerradiografía lateral de cráneo e identifica con MÁXIMA PRECISIÓN los 11 puntos cefalométricos.

Dimensiones de imagen: {img_w}px ancho × {img_h}px alto. Eje Y crece hacia ABAJO.

═══════════════════════════════════════
INSTRUCCIONES CRÍTICAS PUNTO POR PUNTO
═══════════════════════════════════════

S — SELLA TURCA:
• Es el punto en el CENTRO GEOMÉTRICO de la fosa pituitaria (silla turca)
• La fosa pituitaria es una concavidad ósea en la base del cráneo, detrás del quiasma óptico
• S está DENTRO del hueso, no en el borde. Está en la mitad de esa concavidad
• ERROR COMÚN: marcarlo demasiado anterior (hacia la cara). Debe estar bien posterior, en la base del cráneo
• Referencia: S está aproximadamente sobre la vertical que pasa por el conducto auditivo externo

N — NASION:
• Es la intersección de la sutura frontonasal con el plano sagital medio
• Se ubica en la CONCAVIDAD más profunda del perfil óseo entre la frente y la nariz
• N está donde termina el hueso frontal y empiezan los huesos nasales — en la depresión/concavidad
• ERROR COMÚN: marcarlo demasiado anterior (en la punta más saliente). Debe estar en la CONCAVIDAD, ligeramente más posterior
• En perfil lateral, N es el punto más posterior-inferior de la unión frente-nariz, no el más anterior

Or — ORBITARIO:
• Es el punto MÁS INFERIOR del reborde orbitario óseo inferior
• CRÍTICO: Or debe estar significativamente MÁS BAJO (mayor Y) que Po
• La línea Frankfurt (Po→Or) debe tener ~7-10° de inclinación respecto a S-N
• Si Or y Po tienen casi la misma coordenada Y, la posición es INCORRECTA
• Or está en el borde inferior de la cavidad orbitaria, que en la radiografía se ve como una línea curva densa. El punto más bajo de esa curva es Or
• Típicamente Or está 15-30px MÁS ABAJO que Po en la imagen

Po — PORION:
• Punto más SUPERIOR del conducto auditivo externo óseo
• Es el punto más alto del agujero/canal del oído

A — PUNTO A (Subespinal):
• Punto de MÁXIMA CONCAVIDAD del perfil anterior del maxilar superior
• Entre la espina nasal anterior y el borde alveolar superior
• Es la parte más hundida (posterior) del contorno del maxilar, no el borde dental

B — PUNTO B (Supramental):
• Punto de MÁXIMA CONCAVIDAD del perfil anterior de la mandíbula
• Entre el pogonion y el borde alveolar inferior
• Es la parte más hundida (posterior) del contorno mandibular

Me — MENTÓN:
• Punto más INFERIOR de la sínfisis mentoniana
• El punto más bajo de la mandíbula en la línea media

Go — GONION:
• Vértice del ángulo mandibular postero-inferior
• Se obtiene como la bisectriz del ángulo formado por la rama ascendente y el cuerpo mandibular

ENA — Espina Nasal Anterior:
• Punta más anterior y prominente de la espina nasal anterior
• Estructura ósea puntiaguda en el extremo anterior del paladar

ENP — Espina Nasal Posterior:
• Punta posterior del paladar duro
• Extremo posterior de los huesos palatinos

Co — CONDYLION:
• Punto más postero-superior de la cabeza del cóndilo mandibular

═══════════════════════════════════════
VALIDACIÓN ANTES DE RESPONDER:
═══════════════════════════════════════
1. ¿Or.y > Po.y + 10px? (Or debe ser claramente más bajo que Po) → Si no, corrige Or
2. ¿N está en la concavidad frente-nariz, no en la punta más anterior? → Si está muy anterior, muévelo posterior
3. ¿S está en el centro de la fosa pituitaria, bien posterior en la base del cráneo? → Si está muy anterior, muévelo posterior

Responde ÚNICAMENTE con JSON válido, sin texto adicional:
{{
  "S":   {{"x": 123, "y": 456}},
  "N":   {{"x": 123, "y": 456}},
  "A":   {{"x": 123, "y": 456}},
  "B":   {{"x": 123, "y": 456}},
  "Me":  {{"x": 123, "y": 456}},
  "Go":  {{"x": 123, "y": 456}},
  "ENA": {{"x": 123, "y": 456}},
  "ENP": {{"x": 123, "y": 456}},
  "Po":  {{"x": 123, "y": 456}},
  "Or":  {{"x": 123, "y": 456}},
  "Co":  {{"x": 123, "y": 456}}
}}"""

        # Llamada a Claude Vision usando urllib (sin dependencias extra)
        payload = json.dumps({
            "model": "claude-opus-4-6",
            "max_tokens": 800,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }]
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=60) as response:
            data  = json.loads(response.read().decode("utf-8"))
        texto = data["content"][0]["text"].strip()

        # Limpiar markdown si viene con ```json
        if "```" in texto:
            texto = texto.split("```")[1]
            if texto.startswith("json"):
                texto = texto[4:]

        puntos = json.loads(texto)
        return {"success": True, "puntos": puntos}

    except json.JSONDecodeError as e:
        return {"success": False, "detail": f"Respuesta IA no válida: {str(e)}"}
    except Exception as e:
        return {"success": False, "detail": str(e)}


# =================================================================
# ENDPOINTS PRINCIPALES
# =================================================================
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
            if isinstance(coords, dict):
                pts[nombre] = (coords["x"], coords["y"])
            elif isinstance(coords, list):
                pts[nombre] = (coords[0], coords[1])
            else:
                pts[nombre] = tuple(coords)

        requeridos = ["S","N","A","B","Me","Go","ENA","ENP","Po","Or","Co"]
        for p in requeridos:
            if p not in pts:
                return {"success": False, "detail": f"Falta el punto: {p}"}

        factores            = calcular_factores_bimler(pts)
        T1, T2, T3, ML_NSLc, NL_NSLc = calcular_indicadores_T(factores)
        grupo               = arbol_decision(T1, T2, T3)
        categoria           = determinar_categoria(grupo)

        rot_letra  = grupo[0]
        num_letra  = grupo[1] if len(grupo) > 1 and grupo[1].isdigit() else "1"
        sag_letra  = "D" if " D" in grupo else "M" if " M" in grupo else "N"
        vert_letra = "OB" if "OB" in grupo else "DB" if "DB" in grupo else "N"

        rot_map  = {"A": "Anterior",  "R": "Neutra", "P": "Paralelo/Posterior"}
        sag_map  = {"D": "Distoclusión (Clase II)", "N": "Normal (Clase I)", "M": "Mesioclusión (Clase III)"}
        vert_map = {"OB": "Mordida Abierta", "DB": "Mordida Profunda", "N": "Normal"}
        basal_map = {"1": "Mandíbula = Maxila", "2": "Mandíbula < Maxila", "3": "Mandíbula > Maxila"}

        return {
            "success": True,
            "factores_bimler": {
                "SNA": factores["SNA"], "SNB": factores["SNB"], "ANB": factores["ANB"],
                "F3":  factores["F3"],  "F4":  factores["F4"],  "F7":  factores["F7"],
                "ML_NSL": factores["ML_NSL"], "NL_NSL": factores["NL_NSL"],
            },
            "indicadores_petrovic": {
                "T1": T1, "T2": T2, "T3": T3,
                "ML_NSLc": ML_NSLc, "NL_NSLc": NL_NSLc,
            },
            "diagnostico": {
                "grupo": grupo, "categoria": categoria,
                "rotacion": rot_letra, "desc_rotacion": rot_map.get(rot_letra, "—"),
                "basal":    num_letra, "desc_basal":    basal_map.get(num_letra, "—"),
                "sagital":  sag_letra, "desc_sagital":  sag_map.get(sag_letra, "—"),
                "vertical": vert_letra, "desc_vertical": vert_map.get(vert_letra, "—"),
            }
        }
    except Exception as e:
        return {"success": False, "detail": str(e)}

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.2", "casos_validados": 3,
            "features": ["ai_point_suggestion"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
