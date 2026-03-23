import math
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="OrthoAnalysis - Motor Cefalométrico v2.1")

# =================================================================
# MOTOR MATEMÁTICO — v2.1 (bugs corregidos + validados con 3 casos)
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
    # SNA y SNB siempre positivos
    SNA = abs(calcular_angulo_3_puntos(pts["S"], pts["N"], pts["A"]))
    SNB = abs(calcular_angulo_3_puntos(pts["S"], pts["N"], pts["B"]))
    ANB = round(SNA - SNB, 2)

    F3 = calcular_angulo_entre_lineas(pts["Me"],  pts["Go"],  pts["Po"], pts["Or"])
    F4 = calcular_angulo_entre_lineas(pts["ENA"], pts["ENP"], pts["Po"], pts["Or"])
    F7 = calcular_angulo_entre_lineas(pts["N"],   pts["S"],   pts["Po"], pts["Or"])

    # ML/NSL calculado directo desde los puntos (más robusto que F3+F7)
    ML_NSL  = calcular_angulo_entre_lineas(pts["Me"], pts["Go"], pts["S"], pts["N"])
    # NL/NSL = F4 + F7 (validado: Mia=11.22✓ Piero=12.65✓)
    NL_NSL  = round(F4 + F7, 2)

    return {
        "SNA": SNA, "SNB": SNB, "ANB": ANB,
        "F3": F3, "F4": F4, "F7": F7,
        "ML_NSL": ML_NSL, "NL_NSL": NL_NSL,
    }

def calcular_indicadores_T(f):
    # Fórmulas reales OrthoTP — validadas con 3 casos
    ML_NSLc = round(192 - (2 * f["SNB"]), 2)
    NL_NSLc = round(0.198 * f["SNA"] - 4.39, 2)
    T1 = round(ML_NSLc - f["ML_NSL"], 2)
    T2 = round(NL_NSLc - f["NL_NSL"], 2)
    T3 = f["ANB"]
    return T1, T2, T3, ML_NSLc, NL_NSLc

def arbol_decision(T1, T2, T3):
    # Umbral P/R en 0° — validado con 3 casos
    if T1 > 9:    rot = "A"
    elif T1 >= 0: rot = "R"
    else:         rot = "P"

    if T3 > 5:    sag = "D"
    elif T3 >= 0: sag = "N"
    else:         sag = "M"

    if T2 > 3:    vert = "OB"
    elif T2 < -1: vert = "DB"
    else:         vert = "N"

    num = "2" if T1 > 13 else "1"
    grupo = f"{rot}{num} {sag}N" if vert == "N" else f"{rot}{num} {sag}{vert}"
    return grupo

def determinar_categoria(grupo):
    # FIX: eliminada clave duplicada "A2 DDB" (antes → cat5 era sobreescrita por cat6)
    tabla = {
        "R1 NOB": 1, "R2 NOB": 1,
        "R2 DOB": 2, "R3 MOB": 2, "R1 DOB": 2,
        "R2 DN":  3, "R1 NN":  3, "R3 MN":  3, "R1 DN": 3,
        "R1 NDB": 4, "R2 DDB": 4, "R3 MDB": 4,
        "P1 NDB": 4, "P1 NN":  4, "P1 DDB": 4,
        "A1 NDB": 5, "A1 DDB": 5, "A1 NN":  5,  # FIX: era "A2 DDB":5 (duplicado)
        "A2 DDB": 6, "A3 MDB": 6,
    }
    g = grupo.replace(" ", "")
    for key, cat in tabla.items():
        if key.replace(" ", "") == g:  # FIX: comparación exacta (antes era substring → falsos positivos)
            return cat
    return "—"

# =================================================================
# ENDPOINTS
# =================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/analizar")
async def analizar(request: Request):
    try:
        body = await request.json()

        pts = {}
        for nombre, coords in body.items():
            if isinstance(coords, dict):
                pts[nombre] = (coords["x"], coords["y"])
            elif isinstance(coords, list):
                pts[nombre] = (coords[0], coords[1])
            else:
                pts[nombre] = tuple(coords)

        requeridos = ["S", "N", "A", "B", "Me", "Go", "ENA", "ENP", "Po", "Or", "Co"]
        for p in requeridos:
            if p not in pts:
                return {"success": False, "detail": f"Falta el punto: {p}"}

        factores = calcular_factores_bimler(pts)
        T1, T2, T3, ML_NSLc, NL_NSLc = calcular_indicadores_T(factores)
        grupo = arbol_decision(T1, T2, T3)
        categoria = determinar_categoria(grupo)

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
                "SNA": factores["SNA"],
                "SNB": factores["SNB"],
                "ANB": factores["ANB"],
                "F3":  factores["F3"],
                "F4":  factores["F4"],
                "F7":  factores["F7"],
                "ML_NSL":  factores["ML_NSL"],
                "NL_NSL":  factores["NL_NSL"],
            },
            "indicadores_petrovic": {
                "T1": T1, "T2": T2, "T3": T3,
                "ML_NSLc": ML_NSLc, "NL_NSLc": NL_NSLc,
            },
            "diagnostico": {
                "grupo":         grupo,
                "categoria":     categoria,
                "rotacion":      rot_letra,
                "desc_rotacion": rot_map.get(rot_letra, "—"),
                "basal":         num_letra,
                "desc_basal":    basal_map.get(num_letra, "—"),
                "sagital":       sag_letra,
                "desc_sagital":  sag_map.get(sag_letra, "—"),
                "vertical":      vert_letra,
                "desc_vertical": vert_map.get(vert_letra, "—"),
            }
        }

    except Exception as e:
        return {"success": False, "detail": str(e)}

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.1", "casos_validados": 3}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
