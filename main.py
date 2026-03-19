from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import math
import os

app = FastAPI(title="OrthoAnalysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# MODELOS DE DATOS
# ============================================================
class Punto(BaseModel):
    x: float
    y: float

class PuntosRequest(BaseModel):
    S: Punto   # Sella
    N: Punto   # Nasion
    A: Punto   # Punto A
    B: Punto   # Punto B
    Me: Punto  # Menton
    Go: Punto  # Gonion
    ENA: Punto # Espina Nasal Anterior
    ENP: Punto # Espina Nasal Posterior
    Po: Punto  # Porion
    Or: Punto  # Orbitario
    Co: Punto  # Condylion

# ============================================================
# MOTOR GEOMÉTRICO (INVERSIÓN DE EJE Y PARA IMÁGENES)
# ============================================================
def angulo_3_puntos(p1: Punto, vertice: Punto, p2: Punto) -> float:
    """
    Calcula el ángulo interno en el vértice formado por p1-vertice-p2.
    CRÍTICO: Invertimos el eje Y porque en imágenes Y crece hacia abajo.
    """
    v1_x = p1.x - vertice.x
    v1_y = vertice.y - p1.y  # ← inversión Y

    v2_x = p2.x - vertice.x
    v2_y = vertice.y - p2.y  # ← inversión Y

    angulo = math.degrees(math.atan2(v1_y, v1_x) - math.atan2(v2_y, v2_x))

    if angulo > 180:
        angulo -= 360
    elif angulo < -180:
        angulo += 360

    return round(angulo, 2)

def angulo_entre_lineas(l1_p1: Punto, l1_p2: Punto, l2_p1: Punto, l2_p2: Punto) -> float:
    """
    Calcula el ángulo de intersección entre dos planos (líneas).
    Siempre retorna el ángulo agudo (< 90°).
    Inversión de Y aplicada.
    """
    v1_x = l1_p2.x - l1_p1.x
    v1_y = l1_p1.y - l1_p2.y  # ← inversión Y

    v2_x = l2_p2.x - l2_p1.x
    v2_y = l2_p1.y - l2_p2.y  # ← inversión Y

    angulo = math.degrees(abs(math.atan2(v1_y, v1_x) - math.atan2(v2_y, v2_x)))

    if angulo > 180:
        angulo = 360 - angulo
    if angulo > 90:
        angulo = 180 - angulo

    return round(angulo, 2)

# ============================================================
# MOTOR DE FACTORES DE BIMLER
# ============================================================
def calcular_bimler(pts: PuntosRequest) -> dict:
    SNA = angulo_3_puntos(pts.S, pts.N, pts.A)
    SNB = angulo_3_puntos(pts.S, pts.N, pts.B)
    ANB = angulo_3_puntos(pts.A, pts.N, pts.B)

    # Factores medidos respecto al Plano de Frankfurt (Po - Or)
    F3 = angulo_entre_lineas(pts.Me, pts.Go, pts.Po, pts.Or)
    F4 = angulo_entre_lineas(pts.ENA, pts.ENP, pts.Po, pts.Or)
    F7 = angulo_entre_lineas(pts.N, pts.S, pts.Po, pts.Or)

    return {
        "SNA": SNA,
        "SNB": SNB,
        "ANB": ANB,
        "F3": F3,
        "F4": F4,
        "F7": F7
    }

# ============================================================
# MOTOR DE LAVERGNE-PETROVIC
# ============================================================
def calcular_indicadores_T(factores: dict) -> dict:
    """
    Fórmulas extraídas de la clase magistral APOFI:
    T1 = 192 - 2(SNB) - F3 + F7
    T2 = ((F3 + F7) / 2) - 7 - (F7 + F4)
    T3 = ANB
    """
    SNB = factores["SNB"]
    F3  = factores["F3"]
    F4  = factores["F4"]
    F7  = factores["F7"]
    ANB = factores["ANB"]

    T1 = round(192 - (2 * SNB) - F3 + F7, 2)
    T2 = round(((F3 + F7) / 2) - 7 - (F7 + F4), 2)
    T3 = round(ANB, 2)

    return {"T1": T1, "T2": T2, "T3": T3}

def arbol_decision(T1: float, T2: float, T3: float) -> dict:
    """
    Árbol de decisión completo de Lavergne-Petrovic.
    Basado en el PDF clínico de OrthoTP.
    """
    grupo = "Indeterminado"
    rotacion = ""
    basal = ""
    sagital = ""
    vertical = ""

    # ── RAMA T1 > 6 (Rotación Posterior predominante) ──
    if T1 > 6:
        if T2 > 3:
            if T3 <= 1:
                grupo, rotacion, basal, sagital, vertical = "R3 MOB", "R", "3", "M", "OB"
            elif T3 <= 5:
                grupo, rotacion, basal, sagital, vertical = "R1 NOB", "R", "1", "N", "OB"
            else:
                grupo, rotacion, basal, sagital, vertical = "R2 DOB", "R", "2", "D", "OB"
        elif T2 >= 0:  # 0 <= T2 <= 3
            if T3 <= 0:
                grupo, rotacion, basal, sagital, vertical = "A3 MN", "A", "3", "M", "N"
            elif T3 <= 4:
                grupo, rotacion, basal, sagital, vertical = "A1 NN", "A", "1", "N", "N"
            elif T3 <= 7:
                grupo, rotacion, basal, sagital, vertical = "A1 DN", "A", "1", "D", "N"
            else:
                grupo, rotacion, basal, sagital, vertical = "A2 DN", "A", "2", "D", "N"
        else:  # T2 < 0
            if T3 <= -1.5:
                grupo, rotacion, basal, sagital, vertical = "A3 MDB", "A", "3", "M", "DB"
            elif T3 <= 3:
                grupo, rotacion, basal, sagital, vertical = "A1 NDB", "A", "1", "N", "DB"
            elif T3 <= 6:
                grupo, rotacion, basal, sagital, vertical = "A1 DDB", "A", "1", "D", "DB"
            else:
                grupo, rotacion, basal, sagital, vertical = "A2 DDB", "A", "2", "D", "DB"

    # ── RAMA 0 <= T1 <= 6 (Rotación Neutra) ──
    elif T1 >= 0:
        if T2 > 3:
            if T3 <= 1:
                grupo, rotacion, basal, sagital, vertical = "R3 MOB", "R", "3", "M", "OB"
            elif T3 <= 5:
                grupo, rotacion, basal, sagital, vertical = "R1 NOB", "R", "1", "N", "OB"
            else:
                grupo, rotacion, basal, sagital, vertical = "R2 DOB", "R", "2", "D", "OB"
        elif T2 >= 0:  # 0 <= T2 <= 4
            if T3 <= 0:
                grupo, rotacion, basal, sagital, vertical = "R3 MN", "R", "3", "M", "N"
            elif T3 <= 4:
                grupo, rotacion, basal, sagital, vertical = "R1 NN", "R", "1", "N", "N"
            else:
                grupo, rotacion, basal, sagital, vertical = "R2 DN", "R", "2", "D", "N"
        else:  # T2 < 0
            if T3 <= -1:
                grupo, rotacion, basal, sagital, vertical = "R3 MDB", "R", "3", "M", "DB"
            elif T3 <= 3:
                grupo, rotacion, basal, sagital, vertical = "R1 NDB", "R", "1", "N", "DB"
            else:
                grupo, rotacion, basal, sagital, vertical = "R2 DDB", "R", "2", "D", "DB"

    # ── RAMA T1 < 0 (Rotación Anterior) ──
    else:
        if T2 > 3:
            if T3 >= 5.5:
                grupo, rotacion, basal, sagital, vertical = "P2 DOB", "P", "2", "D", "OB"
            elif T3 >= 1:
                grupo, rotacion, basal, sagital, vertical = "P1 NOB", "P", "1", "N", "OB"
            elif T3 >= -6:
                grupo, rotacion, basal, sagital, vertical = "P1 MOB", "P", "1", "M", "OB"
            else:
                grupo, rotacion, basal, sagital, vertical = "P3 MOB", "P", "3", "M", "OB"
        elif T2 >= 0:
            if T3 >= 4:
                grupo, rotacion, basal, sagital, vertical = "P2 DN", "P", "2", "D", "N"
            elif T3 >= 0:
                grupo, rotacion, basal, sagital, vertical = "P1 NN", "P", "1", "N", "N"
            elif T3 >= -7:
                grupo, rotacion, basal, sagital, vertical = "P1 MN", "P", "1", "M", "N"
            else:
                grupo, rotacion, basal, sagital, vertical = "P3 MN", "P", "3", "M", "N"
        else:
            if T3 >= 3:
                grupo, rotacion, basal, sagital, vertical = "P2 DDB", "P", "2", "D", "DB"
            elif T3 >= -1:
                grupo, rotacion, basal, sagital, vertical = "P1 NDB", "P", "1", "N", "DB"
            elif T3 >= -8:
                grupo, rotacion, basal, sagital, vertical = "P1 MDB", "P", "1", "M", "DB"
            else:
                grupo, rotacion, basal, sagital, vertical = "P3 MDB", "P", "3", "M", "DB"

    # Categoría de crecimiento (1-6)
    categoria = determinar_categoria(rotacion, basal, sagital)

    desc_rotacion = {"P": "Posterior", "R": "Neutro", "A": "Anterior"}
    desc_basal = {"1": "Mandíbula = Maxila", "2": "Mandíbula < Maxila", "3": "Mandíbula > Maxila"}
    desc_sagital = {"D": "Distal (Clase II)", "N": "Normal (Clase I)", "M": "Mesial (Clase III)"}
    desc_vertical = {"OB": "Mordida Abierta", "N": "Normal", "DB": "Mordida Profunda"}

    return {
        "grupo": grupo,
        "categoria": categoria,
        "rotacion": rotacion,
        "desc_rotacion": desc_rotacion.get(rotacion, ""),
        "basal": basal,
        "desc_basal": desc_basal.get(basal, ""),
        "sagital": sagital,
        "desc_sagital": desc_sagital.get(sagital, ""),
        "vertical": vertical,
        "desc_vertical": desc_vertical.get(vertical, ""),
    }

def determinar_categoria(rotacion: str, basal: str, sagital: str) -> int:
    """Categorías de potencial de crecimiento 1-6 según Petrovic."""
    if rotacion == "P" and (basal == "2" or sagital == "D"):
        return 1
    elif rotacion == "P":
        return 2
    elif rotacion == "R" and basal == "1" and sagital == "N":
        return 4
    elif rotacion == "R":
        return 3
    elif rotacion == "A" and sagital == "M" and basal == "3":
        return 6
    elif rotacion == "A":
        return 5
    return 3

# ============================================================
# ENDPOINTS DE LA API
# ============================================================
@app.post("/api/analizar")
async def analizar(request: PuntosRequest):
    try:
        factores = calcular_bimler(request)
        indicadores = calcular_indicadores_T(factores)
        diagnostico = arbol_decision(
            indicadores["T1"],
            indicadores["T2"],
            indicadores["T3"]
        )

        return {
            "success": True,
            "factores_bimler": factores,
            "indicadores_petrovic": indicadores,
            "diagnostico": diagnostico
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)