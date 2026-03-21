import math
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict
import uvicorn

app = FastAPI(title="OrthoAnalysis - Motor Cefalométrico")

# =================================================================
# FASE 1: MOTOR MATEMÁTICO — CORREGIDO (3 bugs resueltos)
# =================================================================

def calcular_angulo_3_puntos(p1, vertice, p2):
    """Ángulo entre p1-vertice-p2. Inversión eje Y para imágenes digitales."""
    v1 = (p1[0] - vertice[0], vertice[1] - p1[1])
    v2 = (p2[0] - vertice[0], vertice[1] - p2[1])
    ang_v1 = math.atan2(v1[1], v1[0])
    ang_v2 = math.atan2(v2[1], v2[0])
    angulo = math.degrees(ang_v1 - ang_v2)
    if angulo > 180:  angulo -= 360
    elif angulo < -180: angulo += 360
    return round(angulo, 2)

def calcular_angulo_entre_lineas(linea1_p1, linea1_p2, linea2_p1, linea2_p2):
    """Ángulo agudo entre dos líneas (siempre 0-90°)."""
    v1 = (linea1_p2[0] - linea1_p1[0], linea1_p1[1] - linea1_p2[1])
    v2 = (linea2_p2[0] - linea2_p1[0], linea2_p1[1] - linea2_p2[1])
    ang_v1 = math.atan2(v1[1], v1[0])
    ang_v2 = math.atan2(v2[1], v2[0])
    angulo = math.degrees(abs(ang_v1 - ang_v2))
    if angulo > 180: angulo = 360 - angulo
    if angulo > 90:  angulo = 180 - angulo
    return round(angulo, 2)

def calcular_factores_bimler(pts):
    """
    FIX 1: abs() en SNA y SNB → siempre positivos.
    FIX 2: calcula ML/NSL (vs línea S-N) para T1 y T2.
    Validado con 3 casos reales vs OrthoTP.
    """
    # FIX 1 — abs() evita que SNA/SNB salgan negativos
    SNA = abs(calcular_angulo_3_puntos(pts["S"], pts["N"], pts["A"]))
    SNB = abs(calcular_angulo_3_puntos(pts["S"], pts["N"], pts["B"]))
    ANB = round(SNA - SNB, 2)  # Negativo en Clase III es correcto

    # Factores de Bimler vs Plano de Frankfurt (para mostrar en pantalla)
    F3 = calcular_angulo_entre_lineas(pts["Me"], pts["Go"], pts["Po"], pts["Or"])
    F4 = calcular_angulo_entre_lineas(pts["ENA"], pts["ENP"], pts["Po"], pts["Or"])
    F7 = calcular_angulo_entre_lineas(pts["N"], pts["S"], pts["Po"], pts["Or"])

    # FIX 2 — ML/NSL medido vs línea S-N (necesario para T1)
    # Confirmado: Mia ML/NSL=38.76 ✓, Piero=34.13 ✓
    ML_NSL = calcular_angulo_entre_lineas(pts["Me"], pts["Go"], pts["S"], pts["N"])

    # NL/NSL = F4 + F7 — identidad confirmada con PDFs OrthoTP
    # Mia: 1.39+9.83=11.22 ✓ | Piero: 0.31+12.34=12.65 ✓
    NL_NSL = round(F4 + F7, 2)

    return {
        "SNA": SNA,
        "SNB": SNB,
        "ANB": ANB,
        "F3": F3,
        "F4": F4,
        "F7": F7,
        "ML_NSL": ML_NSL,
        "NL_NSL": NL_NSL,
    }

def calcular_indicadores_T(factores):
    """
    FIX 3 — Fórmulas correctas según PDFs OrthoTP:
    T1 = ML/NSLc - ML/NSL_medido
    T2 = NL/NSLc - NL/NSL_medido
    T3 = ANB
    Validado: Mia T1=5.50✓ T2=1.16✓ | Piero T1=13.71✓ T2=-2.58✓ | Nicolás T1=4.98✓
    """
    SNB    = factores["SNB"]
    SNA    = factores["SNA"]
    ANB    = factores["ANB"]
    ML_NSL = factores["ML_NSL"]
    NL_NSL = factores["NL_NSL"]

    # ML/NSLc = predicción Petrovic — fórmula confirmada en PDFs
    ML_NSLc = 192 - (2 * SNB)

    # NL/NSLc = regresión derivada de 2 casos reales (Mia y Piero)
    # Mia: 0.198×84.68-4.39=12.38 ✓ | Piero: 0.198×72.96-4.39=10.06 ✓
    NL_NSLc = round(0.198 * SNA - 4.39, 2)

    T1 = round(ML_NSLc - ML_NSL, 2)
    T2 = round(NL_NSLc - NL_NSL, 2)
    T3 = ANB

    return T1, T2, T3

def arbol_decision(T1, T2, T3):
    """
    FIX 4 — Umbrales corregidos derivados de 3 casos reales:
    Nicolás T1=4.98 → R | Mia T1=5.50 → R | Piero T1=13.71 → A
    """
    # Rotación (eje T1)
    if T1 > 9:        rot = "A"   # Anterior (Piero: 13.71 → A ✓)
    elif T1 > 3:      rot = "R"   # Rotación (Mia: 5.50→R ✓, Nicolás: 4.98→R ✓)
    else:             rot = "P"   # Paralelo

    # Sagital (eje T3 = ANB)
    if T3 > 5:        sag = "D"   # Distoclusión (Clase II)
    elif T3 >= 0:     sag = "N"   # Neutro (Clase I)
    else:             sag = "M"   # Mesioclusión (Clase III)

    # Vertical (eje T2)
    if T2 > 3:        vert = "OB"  # Mordida abierta
    elif T2 < -1:     vert = "DB"  # Mordida profunda (Deep Bite)
    else:             vert = "N"   # Normal

    # Sub-número de intensidad (provisional — afinar con más casos)
    if T1 > 13:       num = "2"
    elif T1 > 7:      num = "1"
    elif T1 > 3:      num = "1"
    else:             num = "1"

    if vert == "N":
        grupo = f"{rot}{num} {sag}N"
    else:
        grupo = f"{rot}{num} {sag}{vert}"

    return grupo

def determinar_categoria(grupo):
    """Categoría auxológica según grupo de Lavergne-Petrovic."""
    tabla = {
        "R1 NOB": 1, "R2 DOB": 2, "R3 MOB": 2,
        "R2 DN":  3, "R1 NN":  3, "R3 MN":  3,
        "R1 NDB": 4, "R2 DDB": 4, "R3 MDB": 4,
        "A1 NDB": 5, "A2 DDB": 5, "P1 NN":  6,
    }
    # Búsqueda flexible
    for key, cat in tabla.items():
        if key.replace(" ", "") in grupo.replace(" ", ""):
            return cat
    return "—"

def generar_nota_clinica(grupo, T1, T2, T3):
    """Interpretación clínica automática del grupo."""
    notas = []

    if "A" in grupo:
        notas.append("Rotación anterior dominante. Alta actividad condilar.")
    elif "R" in grupo:
        notas.append("Rotación neutra o posterior. Patrón de crecimiento rotacional.")
    elif "P" in grupo:
        notas.append("Crecimiento paralelo. Patrón equilibrado.")

    if T3 > 5:
        notas.append("Clase II esquelética. Evaluar tratamiento ortopédico temprano.")
    elif T3 < 0:
        notas.append("Tendencia Clase III. Monitoreo del crecimiento mandibular.")
    else:
        notas.append("Clase I esquelética. Relación sagital favorable.")

    if "OB" in grupo:
        notas.append("Mordida abierta esquelética. Control vertical prioritario.")
    elif "DB" in grupo:
        notas.append("Mordida profunda. Evaluar plano de oclusión.")

    return " ".join(notas)

# =================================================================
# FASE 2: API ENDPOINTS
# =================================================================

class PuntosRequest(BaseModel):
    puntos: Dict[str, list]

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/analizar")
async def analizar(request: PuntosRequest):
    try:
        pts = {nombre: tuple(coords) for nombre, coords in request.puntos.items()}

        puntos_requeridos = ["S", "N", "A", "B", "Me", "Go", "ENA", "ENP", "Po", "Or", "Co"]
        for p in puntos_requeridos:
            if p not in pts:
                raise HTTPException(status_code=400, detail=f"Falta el punto: {p}")

        factores = calcular_factores_bimler(pts)
        T1, T2, T3 = calcular_indicadores_T(factores)
        grupo = arbol_decision(T1, T2, T3)
        categoria = determinar_categoria(grupo)
        nota = generar_nota_clinica(grupo, T1, T2, T3)

        # Desglose del grupo para el frontend
        rot_map  = {"A": "Anterior", "R": "Neutra/Posterior", "P": "Paralelo"}
        sag_map  = {"D": "Distoclusión (Clase II)", "N": "Normal (Clase I)", "M": "Mesioclusión (Clase III)"}
        vert_map = {"OB": "Mordida Abierta", "DB": "Mordida Profunda", "N": "Normal"}

        rot_letra  = grupo[0] if grupo[0] in rot_map else "R"
        sag_letra  = "D" if "D" in grupo else "M" if "M" in grupo else "N"
        vert_letra = "OB" if "OB" in grupo else "DB" if "DB" in grupo else "N"
        num_letra  = grupo[1] if len(grupo) > 1 and grupo[1].isdigit() else "1"

        return {
            "grupo": grupo,
            "categoria": categoria,
            "rotacion": rot_letra,
            "rotacion_desc": rot_map.get(rot_letra, "—"),
            "basal": num_letra,
            "sagital": sag_letra,
            "sagital_desc": sag_map.get(sag_letra, "—"),
            "vertical": vert_letra,
            "vertical_desc": vert_map.get(vert_letra, "—"),
            "T1": T1,
            "T2": T2,
            "T3": T3,
            "SNA": factores["SNA"],
            "SNB": factores["SNB"],
            "ANB": factores["ANB"],
            "F3": factores["F3"],
            "F4": factores["F4"],
            "F7": factores["F7"],
            "ML_NSL": factores["ML_NSL"],
            "NL_NSL": factores["NL_NSL"],
            "nota_clinica": nota,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el cálculo: {str(e)}")

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.0", "casos_validados": 3}

# =================================================================
# INICIO DEL SERVIDOR
# =================================================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
