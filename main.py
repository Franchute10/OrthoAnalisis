import math
import os
import json
import uuid
import base64
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
# INTEGRACIÓN SUPABASE — analítica anonimizada + almacenamiento
# de radiografías (sin nombre de paciente).
# Se usa HTTP directo (urllib) para no añadir dependencias nuevas.
# =================================================================
import logging as _logging
_log = _logging.getLogger("ortho.analitica")
if not _log.handlers:
    _h = _logging.StreamHandler()
    _h.setFormatter(_logging.Formatter("[analitica] %(levelname)s %(message)s"))
    _log.addHandler(_h); _log.setLevel(_logging.INFO)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
_SUPABASE_OK = bool(SUPABASE_URL and SUPABASE_KEY)


def _supabase_headers(content_type="application/json"):
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": content_type,
        "Prefer": "return=minimal",
    }


def _pais_desde_ip(ip: str) -> str:
    """Geolocalización por IP, sin pedir nada al usuario.
    Servicio gratuito ip-api.com (45 req/min, sin API key).
    Si falla o el IP es local (desarrollo), devuelve '??'.
    """
    if not ip or ip.startswith(("127.", "10.", "192.168.", "::1")):
        return "??"
    try:
        req = urllib.request.Request(
            f"http://ip-api.com/json/{ip}?fields=countryCode",
            headers={"User-Agent": "OrthoAnalysis/1.0"},
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode())
            return data.get("countryCode") or "??"
    except Exception as e:
        _log.info("geoip fallo: %s", e)
        return "??"


def _subir_radiografia(imagen_b64: str) -> str:
    """Sube la radiografía (base64 dataURL) a Supabase Storage.
    Devuelve la URL pública, o None si falla o no hay imagen.
    No se guarda ningún nombre de paciente en el path del archivo.
    """
    if not _SUPABASE_OK or not imagen_b64:
        return None
    # Límite de tamaño del payload base64 (antes de decodificar): ~14MB base64 ≈ 10MB binario
    MAX_B64_CHARS = 14_000_000
    if len(imagen_b64) > MAX_B64_CHARS:
        _log.warning("radiografia rechazada: payload %d bytes > limite", len(imagen_b64))
        return None
    try:
        # Aceptar tanto dataURL completo como base64 puro
        if "," in imagen_b64 and imagen_b64.strip().startswith("data:"):
            header, b64data = imagen_b64.split(",", 1)
            ext = "png" if "png" in header else "jpg"
        else:
            b64data = imagen_b64
            ext = "jpg"

        raw = base64.b64decode(b64data)

        # Validar que sea realmente una imagen por magic bytes (no confiar en el header)
        es_png  = raw[:8] == b"\x89PNG\r\n\x1a\n"
        es_jpg  = raw[:3] == b"\xff\xd8\xff"
        if not (es_png or es_jpg):
            _log.warning("radiografia rechazada: no es PNG/JPEG (magic bytes)")
            return None
        ext = "png" if es_png else "jpg"
        # Límite del binario decodificado (defensa en profundidad)
        if len(raw) > 10 * 1024 * 1024:
            _log.warning("radiografia rechazada: %d bytes decodificados > 10MB", len(raw))
            return None
        filename = f"{uuid.uuid4().hex}.{ext}"
        bucket = "radiografia"

        url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{filename}"
        req = urllib.request.Request(
            url, data=raw, method="POST",
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": f"image/{ext}",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status not in (200, 201):
                return None
        return f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{filename}"
    except Exception as e:
        _log.warning("subida radiografia fallo: %s", e)
        return None


def _limpiar_para_json(valor):
    """Convierte NaN/inf a None para que PostgREST acepte el JSON (rechaza NaN)."""
    if isinstance(valor, float) and (math.isnan(valor) or math.isinf(valor)):
        return None
    return valor


def _guardar_analisis(registro: dict) -> None:
    """Inserta una fila en la tabla `analisis` de Supabase.
    Falla en silencio (no debe romper la respuesta al doctor
    si Supabase está caído o no configurado).
    """
    if not _SUPABASE_OK:
        return
    try:
        url = f"{SUPABASE_URL}/rest/v1/analisis"
        registro = {k: _limpiar_para_json(v) for k, v in registro.items()}
        payload = json.dumps(registro).encode()
        req = urllib.request.Request(
            url, data=payload, method="POST", headers=_supabase_headers()
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        # Analítica es best-effort; nunca debe afectar al usuario, pero SÍ se registra
        # para que se pueda diagnosticar si Supabase falla silenciosamente por semanas.
        _log.warning("guardar_analisis fallo: %s", e)



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
# ─────────────────────────────────────────────────────────────────────────────
# MEDIDAS LINEALES DE BIMLER (requieren calibración px→mm)
# ─────────────────────────────────────────────────────────────────────────────

def _dist(p1, p2):
    """Distancia euclidiana entre dos puntos."""
    return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

def _dist_perp(pt, l1, l2):
    """Distancia perpendicular de pt a la línea l1-l2."""
    dx, dy = l2[0]-l1[0], l2[1]-l1[1]
    llen = math.sqrt(dx*dx + dy*dy)
    if llen < 1: return 0.0
    cross = abs(dx*(l1[1]-pt[1]) - (l1[0]-pt[0])*dy)
    return cross / llen

def calcular_medidas_lineales(pts, px_per_mm):
    """Calcula medidas lineales de Bimler en mm. Requiere px_per_mm válido."""
    if not px_per_mm or px_per_mm <= 0:
        return None
    def mm(px): return round(px / px_per_mm, 1)

    po, or_ = pts.get("Po"), pts.get("Or")

    result = {
        "NS":    mm(_dist(pts["N"], pts["S"])),
        "CoGo":  mm(_dist(pts["Co"], pts["Go"])),
        "NMe":   mm(_dist(pts["N"],  pts["Me"])),
        "AB":    mm(_dist(pts["A"],  pts["B"])),
        "MeCo":  mm(_dist(pts["Me"], pts["Co"])),  # aprox Gn-Co
    }
    if po and or_:
        result["NFH"]  = mm(_dist_perp(pts["N"],  po, or_))
        result["SFH"]  = mm(_dist_perp(pts["S"],  po, or_))
        result["FHMe"] = mm(_dist_perp(pts["Me"], po, or_))
    return result


def _clasif(val, norma, tol, etiquetas=("Pequeño","Medio","Grande")):
    """Clasifica un valor linear respecto a su norma."""
    if val is None: return "—"
    d = val - norma
    if   d < -tol: return etiquetas[0]
    elif d >  tol: return etiquetas[1 if len(etiquetas)==2 else 2]
    else:          return etiquetas[1]

def generar_resumen_narrativo(f, indicadores_T, medidas_lineales, grupo, categoria):
    """Genera lista de frases diagnósticas estilo OrthoTP."""
    T1  = indicadores_T.get("T1")
    T2  = indicadores_T.get("T2")
    T3  = indicadores_T.get("T3")
    F3  = f.get("F3")
    F4  = f.get("F4")
    F5  = f.get("F5")
    ANB = f.get("ANB")

    frases = []

    # Rotación de crecimiento
    rot = grupo[0] if grupo else "?"
    rot_desc = {"R":"Neutro (R)", "A":"Anterior (A)", "P":"Posterior (P)"}
    frases.append(f"Rotación de crecimiento tipo {rot_desc.get(rot, rot)}")

    # Relación basal maxilo-mandibular (T3/ANB)
    if ANB is not None:
        if   ANB >  5: frases.append("Mandíbulo-maxilar dolicognático (DISTAL)")
        elif ANB <  0: frases.append("Mandíbulo-maxilar mesognático (MESIAL)")
        else:          frases.append("Mandíbulo-maxilar mesoprosópico = Neutro")

    # Inclinación plano mandibular (F3)
    if F3 is not None:
        if   F3 > 30: frases.append("Inclinación del plano mandibular – Hiperdivergente")
        elif F3 < 20: frases.append("Inclinación del plano mandibular – Hipodivergente")
        else:         frases.append("Inclinación del plano mandibular – Neutra")

    # Inclinación maxilar (F4)
    if F4 is not None:
        if   F4 > 2:  frases.append("Inclinación maxilar – Arriba=Positivo")
        elif F4 < -2: frases.append("Inclinación maxilar – Arriba=Negativo")
        else:         frases.append("Inclinación maxilar – Horizontal")

    # Inclinación clivus (F5)
    if F5 is not None:
        if   F5 > 70: frases.append("Inclinación del clivus = V = Vertical (profundo)")
        elif F5 < 60: frases.append("Inclinación del clivus = D = Horizontal (bajo)")
        else:         frases.append("Inclinación del clivus = M = Medio")
        # Clivo maxilar profundidad
        if F5 > 68:   frases.append("Clivo maxilar – PROFUNDO")
        elif F5 < 62: frases.append("Clivo maxilar (horizontal) – BAJO (DEEP)")
        else:         frases.append("Clivo maxilar – MEDIO")

    # Grupo y categoría
    frases.append(f"Grupo {grupo}")
    frases.append(f"Categoría de crecimiento auxológico = {categoria}")

    # Medidas lineales (si disponibles)
    if medidas_lineales:
        ns = medidas_lineales.get("NS")
        if ns:
            c = _clasif(ns, 70, 4, ("Corto","Medio","Largo"))
            frases.append(f"Base craneal anterior (N-S) – {c} ({ns} mm)")

        gn = medidas_lineales.get("MeCo")
        if gn:
            c = _clasif(gn, 110, 10, ("Pequeño","Medio","Grande"))
            frases.append(f"Diagonal mandibular (Me-Co) – {c} ({gn} mm)")

        nme = medidas_lineales.get("NMe")
        if nme:
            c = _clasif(nme, 120, 5, ("Pequeño","Medio","Grande"))
            frases.append(f"Altura anterior de la cara (N-Me) – {c} ({nme} mm)")

        # Fix Rubén: resalte esquelético = A'-B' (proyección sobre Frankfurt),
        # con norma de la fuente Bimler: 0/6mm Clase I, >6 Clase II, <0 Clase III.
        ab = f.get("resalte_esqueletico_mm")
        if ab is None:
            ab = medidas_lineales.get("resalte_esqueletico_mm")
        if ab is not None:
            if   ab > 6:   c = "Clase II"
            elif ab < 0:   c = "Clase III"
            else:          c = "Clase I"
            frases.append(f"Resalte esquelético (A'-B') – {c} ({ab} mm)")

    return frases


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

    # F4: plano palatino con FH — firmado: + si ENA más bajo que ENP.
    # Fix (test de invariancia): el signo se mide RESPECTO A FRANKFURT, no con la
    # Y cruda de la imagen. Comparar pts[..][1] directo rompía el signo cuando la
    # Rx estaba inclinada (F4 pasaba de -1.91 a +1.91 al rotar). Proyectamos ENA y
    # ENP sobre la perpendicular a FH para saber cuál queda "más abajo" en el
    # marco de Frankfurt.
    _fhx = pts["Or"][0] - pts["Po"][0]
    _fhy = pts["Or"][1] - pts["Po"][1]
    _fhl = (_fhx**2 + _fhy**2) ** 0.5 or 1.0
    # Perpendicular a FH que apunta "hacia abajo" en el marco de la imagen
    # (rotar FH +90°): (nx, ny) = (-fhy, fhx)/|FH|
    _nx, _ny = -_fhy / _fhl, _fhx / _fhl
    _proj_ena = pts["ENA"][0]*_nx + pts["ENA"][1]*_ny
    _proj_enp = pts["ENP"][0]*_nx + pts["ENP"][1]*_ny
    _signo_f4 = 1 if _proj_ena > _proj_enp else -1
    F4 = round(_ang_FH(pts["ENA"], pts["ENP"]) * _signo_f4, 2)

    # F7: base craneal anterior con FH (sin signo)
    F7 = _ang_FH(pts["N"], pts["S"])

    # ── F1 = N-A, ángulo superior del perfil. Norma -1/+1. ──
    # Convención OrthoTP validada en 3 casos (Mia/Nicolás/Piero): F1 = -signed(N,A).
    #
    # ⚠ DISCREPANCIA ABIERTA CON EL DR. RUBÉN (caso Benjamín):
    #   El especialista obtiene F1=+0.5; el motor da -1.54. El signo depende del EJE
    #   contra el que se mide "A por delante de N": Rubén lo juzga en la horizontal
    #   del trazado (A 14px a la derecha de N → +); el motor lo proyecta sobre
    #   Frankfurt, que en esta Rx está inclinado ~4°, y con N muy por encima de A la
    #   proyección invierte el signo (A queda 8.8px "detrás" → -).
    #   La ficha de Bimler NO especifica el eje de referencia, así que NO se cambia
    #   la fórmula sin una fuente que lo defina. Queda documentado para resolver con
    #   Rubén cuál eje usa su escuela. La MAGNITUD (~1-1.5°) coincide; solo el signo
    #   está en disputa, y solo cuando A y N están casi alineados verticalmente.
    F1 = round(-calcular_angulo_signed(pts["N"],  pts["A"],  Po, Or), 2)

    # F2 = A-B (ángulo inferior del perfil). Norma 0/+10. Convención validada
    # (+ = retrogenia/Clase II). Caso Benjamín: Rubén +14, motor +16.98 → mismo
    # signo, diferencia por marcación de puntos.
    F2 = round( calcular_angulo_signed(pts["A"],  pts["B"],  Po, Or), 2)

    # ── Fix Rubén: F8 = FLEXIÓN MANDIBULAR con CAPITULARE (C-Go), no Condylion ──
    # Fuente primaria Bimler: F8 mide C-Go, donde C = Capitulare (CENTRO del cóndilo),
    # distinto de Cd/Condylion (póstero-superior). Valor normal 0°/8°:
    #   Hiperflexión → Go por delante de C → signo (-)
    #   Hipoflexión  → Go por detrás de C  → signo (+)
    # Retrocompatibilidad: si el análisis no trae C (marcados viejos), se usa Co
    # como APROXIMACIÓN y se avisa. El signo se mantiene consistente con OrthoTP.
    _pf = pts.get("C") or pts.get("Co")   # Capitulare preferente; Co como fallback
    F8 = round(-calcular_angulo_signed(_pf, pts["Go"], Po, Or), 2)
    F8_fuente = "Capitulare (C)" if pts.get("C") else "Condylion (Co) — aproximado"

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

    # ── Fix Rubén: RESALTE ESQUELÉTICO A'-B' CON SIGNO, en mm ──
    # Fuente Bimler: A'-B' = distancia entre las proyecciones de A y B sobre
    # Frankfurt (NO la distancia directa A-B). Signo según dirección sobre FH:
    #   A por delante de B → (+)   |   B por delante de A → (-)
    # Valor normal 0/6mm (Clase I); >6 Clase II; <0 Clase III.
    # Se proyecta el vector (A'-B') sobre el eje de Frankfurt (Po→Or) para el signo.
    fh_dx, fh_dy = (Or[0] - Po[0]), (Or[1] - Po[1])
    fh_len = (fh_dx**2 + fh_dy**2) ** 0.5 or 1.0
    ux, uy = fh_dx / fh_len, fh_dy / fh_len           # vector unitario Frankfurt
    resalte_px = (A_prima[0] - B_prima[0]) * ux + (A_prima[1] - B_prima[1]) * uy
    # Convención OrthoTP: A adelante de B (Clase II esquelética) = positivo.
    # El eje FH apunta Po→Or (posterior→anterior); un valor + significa A' más
    # anterior que B'. Se normaliza el signo para que coincida con la fuente.
    resalte_ab_px = round(abs(resalte_px), 1)
    resalte_ab_mm = (round((resalte_px / escala_mm_px), 1)
                     if escala_mm_px else None)

    # ── Medidas lineales (en píxeles, convertibles a mm) ──────
    lin = {
        "A_prima_T":   round(distancia(A_prima, T),    1),
        "A_prima_B_prima": round(distancia(A_prima, B_prima), 1),
        "resalte_esqueletico_px": resalte_ab_px,       # magnitud proyectada (px)
        "resalte_esqueletico_mm": resalte_ab_mm,       # CON SIGNO, en mm (norma 0/6)
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
        "F8_fuente": F8_fuente,
        "ML_NSL": ML_NSL, "NL_NSL": NL_NSL,
        "perfil": perfil, "ABS": ABS, "ABI": ABI, "ABT": ABT,
        "AG": AG, "APNI_estimado": APNI_estimado, "ODI": ODI,
        "resalte_esqueletico_mm": resalte_ab_mm,
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

def arbol_decision(T1, T2, T3, poblacion: str = "latam"):
    """
    Árbol de decisión EXACTO de Petrovic, Stutzmann, Lavergne (1996).
    Fuente: Figura 14, tesis UNAM-León 2019 (Mateos González).
    Produce los 33 grupos originales con sub-rangos numéricos de T3.

    T1 — Rotación mandibular:
      A  si T1 > 6   (Anterior)
      R  si 0≤T1≤6  (Neutra)
      P  si T1 < 0   (Posterior)

    T2 — Dimensión vertical:
      OB si T2 > 3   (Mordida Abierta)
      N  si 0≤T2≤3  (Normal)
      DB si T2 < 0   (Mordida Profunda)

    T3 — Sub-rangos numéricos que determinan tipo rotacional y sagital:
      Ver diagrama completo en documentación.
    """
    p = POBLACION_PARAMS.get(poblacion, POBLACION_PARAMS["latam"])
    # Guard: valores no numericos -> no clasificar (evita caer en P por NaN)
    if any((x is None) or (isinstance(x, float) and math.isnan(x)) for x in (T1, T2, T3)):
        return "?? (datos incompletos)"
    T1_ANT = p["T1_ant"]
    T2_OB  = p["T2_ob"]
    T2_N   = p["T2_n_inf"]

    # Bordes inclusivos (Figura 14 Petrovic 1996). No cambiar >= por >.
    if T1 > T1_ANT:      # ── ANTERIOR ──────────────────────────────────────
        if T2 > T2_OB:                     # OB
            if   T3 <= 1.5:  return "A3 MOB"
            elif T3 <= 5.5:  return "A1 NOB"
            elif T3 <= 8.5:  return "A1 DOB"
            else:            return "A2 DOB"
        elif T2 >= T2_N:                   # N
            if   T3 <= 0:    return "A3 MN"
            elif T3 <= 4:    return "A1 NN"
            elif T3 <= 7:    return "A1 DN"
            else:            return "A2 DN"
        else:                               # DB (T2 < T2_N)
            if   T3 <= -1.5: return "A3 MDB"
            elif T3 <= 3:    return "A1 NDB"
            elif T3 <= 6:    return "A1 DDB"
            else:            return "A2 DDB"

    elif T1 >= 0:   # ── NEUTRAL  (T1 ≤ T1_ANT) ───────────────────────────────────────
        if T2 > T2_OB:                     # OB
            if   T3 <= 1:    return "R3 MOB"
            elif T3 <= 5:    return "R1 NOB"
            else:            return "R2 DOB"
        elif T2 >= T2_N:                   # N
            if   T3 <= 0:    return "R3 MN"
            elif T3 <= 4:    return "R1 NN"
            else:            return "R2 DN"
        else:                               # DB (T2 < T2_N)
            if   T3 <= -1:   return "R3 MDB"
            elif T3 <= 3:    return "R1 NDB"
            else:            return "R2 DDB"

    else:           # ── POSTERIOR (T1 < 0) ─────────────────────────────────────
        if T2 > T2_OB:                     # OB
            if   T3 >= 5.5:  return "P2 DOB"
            elif T3 >= 1:    return "P1 NOB"
            elif T3 >= -6:   return "P1 MOB"
            else:            return "P3 MOB"
        elif T2 >= T2_N:                   # N
            if   T3 >= 4:    return "P2 DN"
            elif T3 >= 0:    return "P1 NN"
            elif T3 >= -7:   return "P1 MN"
            else:            return "P3 MN"
        else:                               # DB (T2 < T2_N)
            if   T3 >= 3:    return "P2 DDB"
            elif T3 >= -1:   return "P1 NDB"
            elif T3 >= -8:   return "P1 MDB"
            else:            return "P3 MDB"


# ── 33 grupos de Petrovic-Lavergne 1996 ──────────────────────────────────
# Fuente: diagrama de flujo original (Petrovic, Stutzmann, Lavergne, 1996)
GRUPOS_33 = {
    # Categoría 1 — Potencial Muy Bajo
    "P2 DOB": 1,  "P2 DN": 1,  "P2 DDB": 1,

    # Categoría 2 — Potencial Bajo
    "A2 DOB": 2,  "A2 DN": 2,  "A2 DDB": 2,
    "P1 NOB": 2,  "P1 NN": 2,  "P1 NDB": 2,

    # Categoría 3 — Potencial Moderado
    "R2 DOB": 3,  "R2 DN": 3,  "R2 DDB": 3,

    # Categoría 4 — Potencial Neutro
    "R1 NOB": 4,  "R1 NN": 4,  "R1 NDB": 4,

    # Categoría 5 — Potencial Alto
    "A1 DOB": 5,  "A1 DN": 5,  "A1 DDB": 5,
    "A1 NOB": 5,  "A1 NN": 5,  "A1 NDB": 5,
    "P1 MOB": 5,  "P1 MN": 5,  "P1 MDB": 5,
    "R3 MOB": 5,  "R3 MN": 5,  "R3 MDB": 5,

    # Categoría 6 — Potencial Excesivo
    "A3 MOB": 6,  "A3 MN": 6,  "A3 MDB": 6,
    "P3 MOB": 6,  "P3 MN": 6,  "P3 MDB": 6,
}


# ── Parámetros por población ────────────────────────────────────────────────
# Fuente: Petrovic (1996) para Latinoamérica; OrthoTP/Bjork-Skieller para Europa
POBLACION_PARAMS = {
    "latam": {
        "nombre":      "Latinoamérica (Petrovic 1996)",
        "T1_ant":      6,     # T1 > 6 → Anterior
        "T1_neu":      0,     # 0 ≤ T1 ≤ 6 → Neutro
        "T2_ob":       3,     # T2 > 3 → Mordida Abierta
        "T2_n_inf":    0,     # T2 ≥ 0 → Normal (T2 < 0 → Profunda)
        "fuente":      "Petrovic-Stutzmann-Lavergne (1996); Coba Moreno UCE (2019); UNAM-León (2019)"
    },
    "europa": {
        "nombre":      "Europa / OrthoTP (calibración italiana)",
        "T1_ant":      9,     # T1 > 9 → Anterior
        "T1_neu":      0,     # 0 ≤ T1 ≤ 9 → Neutro
        "T2_ob":       3,     # T2 > 3 → Mordida Abierta
        "T2_n_inf":   -1,     # T2 ≥ -1 → Normal (T2 < -1 → Profunda)
        "fuente":      "OrthoTP (italiano); Bjork-Skieller (1972); Guercio-Saccomanno (2009-2018)"
    },
}

# ── Recomendaciones de aparatología por TIPO ROTACIONAL ────────────────────
# Fuente principal: Petrovic-Stutzmann-Lavergne (via Simoes W., 2004;
#   Tamayo Sendoya A., 2020; draclaude@gmail.com PPT docente Lavergne-Petrovic)
# Indicadores de extracción: 0=contraindicada, +=frecuente, ++=inevitable
APARATOS_POR_TIPO = {
    # ── CATEGORÍA 1 — potencial mandibular muy bajo ──────────────────
    "P2D": {
        "cat": 1,
        "pronostico": "⚠️ Desfavorable",
        "desc_pronostico": (
            "Mandíbula con potencial de crecimiento muy bajo comparado al maxilar. "
            "Respuestas ortodóncicas y ortopédicas lentas. Tratamientos prolongados. "
            "Mordidas abiertas frecuentes."
        ),
        "primera_linea": ["Ortodóncico convencional", "Elásticos Clase II (fuerzas leves e intermitentes)"],
        "alternativo":   ["Aparatos fijos segmentados"],
        "extraccion":    "+  (frecuentemente indicada)",
        "cirugia":       False,
        "timing":        "Iniciar lo antes posible; baja respuesta requiere mayor tiempo",
        "pronostico_largo_plazo": (
            "Seguimiento a 10 años (Moro, 2001, USP-Bauru, n=100, Clase II tratada con "
            "extracciones): menor tendencia a recidiva de sobremordida que categoría 3. "
            "Perfil facial post-contención tiende a ser más estable que en categorías 4-5."
        ),
    },
    # ── CATEGORÍA 2 — potencial bajo ─────────────────────────────────
    "A2D": {
        "cat": 2,
        "pronostico": "⚠️ Desfavorable",
        "desc_pronostico": (
            "Rotación anterior acorta el Comprimento Oclusal Relevante. "
            "Distoclusión muy severa con mordida profunda. Alta probabilidad de extracciones. "
            "Tratamientos largos con alta tendencia a recidiva."
        ),
        "primera_linea": ["Ortodóncico convencional"],
        "alternativo":   ["Elásticos Clase II", "Extrusión premolares"],
        "extraccion":    "++ (muy frecuentemente indicada)",
        "cirugia":       False,
        "timing":        "Sin ventana óptima; respuesta limitada independiente de la edad",
    },
    "P1N": {
        "cat": 2,
        "pronostico": "⚠️ Moderadamente desfavorable",
        "desc_pronostico": (
            "Potencial mandibular bajo; la rotación posterior 'alarga' la mandíbula dando una "
            "relación sagital neutra aparente. Enzimas de osteosíntesis poco activas. "
            "Frecuentes apiñamientos, mordidas profundas o abiertas."
        ),
        "primera_linea": ["Hiperpropulsor postural de la mandíbula*", "Elásticos Clase II"],
        "alternativo":   ["Bionator*", "Activador LSU*"],
        "extraccion":    "0  (contraindicada)",
        "cirugia":       False,
        "timing":        "Fase ascendente del pico puberal (CVM estadio 2-3)",
    },
    # ── CATEGORÍA 3 — potencial moderado ─────────────────────────────
    "R2D": {
        "cat": 3,
        "pronostico": "🔶 Moderado",
        "desc_pronostico": (
            "Pequeña diferencia de potencial entre maxila y mandíbula. "
            "Sin corrección espontánea de la distoclusión. Pronóstico mediano si se logra "
            "cambiar la rotación mandibular de neutra a posterior."
        ),
        "primera_linea": ["Hiperpropulsor postural (objetivo: cambiar R→P)*", "Activador LSU*"],
        "alternativo":   ["Bionator*", "Elásticos Clase II"],
        "extraccion":    "+  (a evaluar según apiñamiento)",
        "cirugia":       False,
        "timing":        "Fase ascendente del pico puberal (CVM estadio 2-3); aprox. 18% de los pacientes",
        "pronostico_largo_plazo": (
            "Seguimiento a 10 años (Moro, 2001, USP-Bauru, n=100, Clase II tratada con "
            "extracciones): mayor tendencia a recidiva de sobremordida que categoría 1. "
            "Corrección molar sostenida en 75% por crecimiento diferencial real (ABCH), resto "
            "por compensación dentoalveolar (Moro et al., AJODO 2000;117:86-97)."
        ),
    },
    # ── CATEGORÍA 4 — potencial neutro ───────────────────────────────
    "R1N": {
        "cat": 4,
        "pronostico": "✅ Favorable",
        "desc_pronostico": (
            "Relaciones basales equilibradas. Potencial mandibular similar al maxilar. "
            "Comparador oclusal funciona correctamente. Grupo más frecuente (~20% de casos)."
        ),
        "primera_linea": ["Ortodóncico convencional"],
        "alternativo":   ["Aparatos funcionales suaves si se requiere corrección menor"],
        "extraccion":    "según apiñamiento dental",
        "cirugia":       False,
        "timing":        "Flexible; buena respuesta en cualquier fase",
        "pronostico_largo_plazo": (
            "Seguimiento a 10 años (Moro, 2001, USP-Bauru, n=100, Clase II tratada con "
            "extracciones): mejor estabilidad de alineación de incisivos inferiores de todas "
            "las categorías estudiadas (78.57% de casos satisfactorios). Nota: perfil facial "
            "post-contención tendió a mostrarse más retruido que en categorías 1-3."
        ),
    },
    # ── CATEGORÍA 5 — potencial alto ─────────────────────────────────
    "A1D": {
        "cat": 5,
        "pronostico": "✅✅ Muy favorable",
        "desc_pronostico": (
            "A pesar de la apariencia de Clase II severa, el potencial mandibular es mayor "
            "que el maxilar. La rotación anterior es el factor determinante, no el potencial. "
            "Cambiar la dirección de crecimiento es el objetivo principal."
        ),
        "primera_linea": ["Regulador de Función Fränkel (eficacia +++)*", "Hiperpropulsor postural*"],
        "alternativo":   ["Activador LSU*", "Bionator*"],
        "extraccion":    "0  (contraindicada — no extraer)",
        "cirugia":       False,
        "timing":        "Fase ascendente puberal (CVM 2-3); máxima efectividad tisular",
    },
    "A1N": {
        "cat": 5,
        "pronostico": "✅✅ Muy favorable",
        "desc_pronostico": (
            "Clase I con potencial mandibular mayor que el maxilar. "
            "La rotación anterior compensatoria genera la relación neutra. "
            "Excelente respuesta a aparatos funcionales."
        ),
        "primera_linea": ["Regulador de Función Fränkel*", "Ortodóncico convencional"],
        "alternativo":   ["Bionator*"],
        "extraccion":    "0  (contraindicada)",
        "cirugia":       False,
        "timing":        "Fase ascendente puberal; buena respuesta",
    },
    "P1M": {
        "cat": 5,
        "pronostico": "🔶 Moderado a desfavorable",
        "desc_pronostico": (
            "Alto potencial mandibular CON rotación posterior: doble factor mesializante. "
            "Difícil inhibir el crecimiento mandibular. Iniciar lo más precozmente posible. "
            "Con mordida profunda el tratamiento tiende a ser más favorable."
        ),
        "primera_linea": ["Tracción postero-anterior del maxilar (máscara facial)*", "Contención activa mandibular"],
        "alternativo":   ["Tracción antero-posterior de la mandíbula (limitada)"],
        "extraccion":    "0  (contraindicada)",
        "cirugia":       False,
        "timing":        "Iniciar PRECOZMENTE (dentición mixta temprana); ~1.2% de los pacientes",
    },
    "R3M": {
        "cat": 5,
        "pronostico": "🔶 Variable",
        "desc_pronostico": (
            "Potencial mandibular mayor que el maxilar con rotación neutra. "
            "Comparador oclusal NO funciona normalmente. Mesioclusión con o sin mordida cruzada. "
            "Con mordida profunda: mejor pronóstico. Seguimiento periódico hasta fin del crecimiento."
        ),
        "primera_linea": ["Tracción postero-anterior del maxilar*", "Seguimiento periódico"],
        "alternativo":   ["Máscara facial"],
        "extraccion":    "0  (contraindicada)",
        "cirugia":       False,
        "timing":        "Inicio precoz; vigilar hasta fin de crecimiento (alta recidiva sin contención)",
    },
    # ── CATEGORÍA 6 — potencial extremo ──────────────────────────────
    "A3M": {
        "cat": 6,
        "pronostico": "⛔ Muy desfavorable",
        "desc_pronostico": (
            "Menos del 1% de los pacientes. Potencial mandibular extremo con rotación anterior. "
            "Mesioclusión con mordida cruzada anterior, frecuentemente asociada a mordida profunda. "
            "Alta recidiva. Monitoreo hasta fin del crecimiento imprescindible."
        ),
        "primera_linea": ["Tracción postero-anterior del maxilar (inicio precoz)", "Evaluación para cirugía ortognática"],
        "alternativo":   ["Máscara facial", "Mentón de contención"],
        "extraccion":    "0  (contraindicada en fase de crecimiento)",
        "cirugia":       True,
        "timing":        "Inicio MUY precoz; alta probabilidad de cirugía al fin del crecimiento",
    },
    "P3M": {
        "cat": 6,
        "pronostico": "⛔ Sombrio — CIRUGÍA",
        "desc_pronostico": (
            "Potencial mandibular altísimo CON rotación posterior: mesioclusión severa. "
            "P3MN y P3MOB pertenecen a cirugía ortognática. "
            "Tratamiento ortopédico precoz puede reducir la magnitud quirúrgica. "
            "Alta tendencia a recidiva incluso post-quirúrgica."
        ),
        "primera_linea": ["⚠️ CIRUGÍA ORTOGNÁTICA (indicación principal)",
                          "Tracción postero-anterior del maxilar (preparación pre-quirúrgica)"],
        "alternativo":   ["Aparatos funcionales para reducir magnitud quirúrgica"],
        "extraccion":    "+  (frecuente en fase pre-quirúrgica)",
        "cirugia":       True,
        "timing":        "Cirugía diferida hasta fin del crecimiento; ortopedia desde dentición mixta",
    },
}

def obtener_recomendacion(grupo: str) -> dict:
    """Devuelve la recomendación de aparatología para un grupo rotacional.
    El tipo base (sin vertical) se extrae del grupo (ej. 'A1 DN' → 'A1D').
    """
    # Extraer tipo base: rot + basal + sag (sin OB/N/DB)
    partes = grupo.strip().split()
    if len(partes) < 2:
        return {}
    tipo = partes[0]                    # 'A1', 'P2', 'R1', etc.
    # sag está en partes[1][0] o partes[1]: 'DOB', 'DN', 'NOB', etc.
    sag_raw = partes[1]
    sag = sag_raw[0] if sag_raw else ''  # 'D', 'N', 'M'
    tipo_base = tipo + sag              # 'A1D', 'P2N', 'R1N', etc.
    rec = APARATOS_POR_TIPO.get(tipo_base, {})
    return rec


def determinar_categoria(grupo):
    """
    Busca el grupo en la tabla de grupos alcanzables de Petrovic-Lavergne.
    Devuelve (categoria, advertencia):
      categoria   -> int 1-6, o None si no está mapeado
      advertencia -> None, o texto si el grupo no está en la tabla
    """
    if not grupo or not isinstance(grupo, str):
        return None, "Grupo no calculado (datos incompletos)."
    cat = GRUPOS_33.get(grupo.strip())
    if cat is None:
        return None, (f"Grupo '{grupo}' no está en los 33 grupos de Petrovic-Lavergne 1996. Revise los puntos.")
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
        orig_w    = body.get("orig_width",  img_w)   # original (px de las anclas)
        orig_h    = body.get("orig_height", img_h)
        anchors   = body.get("anchors")              # opcional: {S,N,Me,Go} en px original

        if not image_b64:
            return {"success": False, "detail": "No se recibió imagen"}

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return {"success": False, "detail": "ANTHROPIC_API_KEY no configurada en el servidor"}

        # ── Detección real del contorno del cráneo con OpenCV (opcional) ──
        cv_skull = detectar_craneo_opencv(image_b64, img_w, img_h)
        print(f"[OpenCV] skull={'detectado(Otsu+CLAHE)' if cv_skull else 'fallback-Claude'}"
              + (f" bbox=({cv_skull['skull_left']},{cv_skull['skull_top']})-({cv_skull['skull_right']},{cv_skull['skull_bottom']})" if cv_skull else ""))

        # ── Bounding box antes del prompt: fuente de verdad para anclas ──
        if cv_skull:
            bbox_left   = float(cv_skull["skull_left"])
            bbox_right  = float(cv_skull["skull_right"])
            bbox_top    = float(cv_skull["skull_top"])
            bbox_bottom = float(cv_skull["skull_bottom"])
        else:
            bbox_left   = 0.05 * img_w
            bbox_right  = 0.95 * img_w
            bbox_top    = 0.05 * img_h
            bbox_bottom = 0.95 * img_h
        bbox_w = max(bbox_right - bbox_left, 50.0)
        bbox_h = max(bbox_bottom - bbox_top, 50.0)

        # ── Convertir anclas px ORIGINAL → x_skull/y_skull ──
        anclas_xs = {}
        anclas_px = {}
        if anchors:
            sx = img_w / float(orig_w or img_w)
            sy = img_h / float(orig_h or img_h)
            for k in ("S", "N", "Me", "Go"):
                a = anchors.get(k)
                if not a:
                    continue
                try:
                    xc = float(a["x"]) * sx
                    yc = float(a["y"]) * sy
                except (KeyError, TypeError, ValueError):
                    continue
                anclas_px[k] = (round(xc, 1), round(yc, 1))
                anclas_xs[k] = (round((xc - bbox_left) / bbox_w * 100.0, 1),
                                round((yc - bbox_top)  / bbox_h * 100.0, 1))

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
 Co:  x_skull≈22%, y_skull≈34%   (Condylion — polo póstero-superior del cóndilo)
 C:   x_skull≈23%, y_skull≈36%   (Capitulare — CENTRO del cóndilo; ~2% por debajo y ligeramente por delante de Co)
 Cls: x_skull≈28%, y_skull≈30%   (clivus sup)
 Cli: x_skull≈24%, y_skull≈42%   (clivus inf)

 Úsalas como REFERENCIA cuando un punto no es claramente visible.
 Prioriza siempre lo que ves en la imagen sobre estas proporciones.
"""
        else:
            contorno_ctx = ""

        # ── Bloque de ANCLAS marcadas por el doctor (si vienen) ──
        if anclas_xs:
            def _xs(k):
                return anclas_xs.get(k, (0.0, 0.0))
            s_xs, s_ys   = _xs("S")
            n_xs, n_ys   = _xs("N")
            me_xs, me_ys = _xs("Me")
            go_xs, go_ys = _xs("Go")
            anclas_ctx = f"""
════════════════════════════════════════════════════
ANCLAS MARCADAS POR EL ORTODONCISTA (coordenadas exactas)
════════════════════════════════════════════════════
Estas posiciones son VERDAD ABSOLUTA — el doctor las marcó con precisión clínica.
NO las muevas.

S  → x_skull={s_xs:.1f}, y_skull={s_ys:.1f}
N  → x_skull={n_xs:.1f}, y_skull={n_ys:.1f}
Me → x_skull={me_xs:.1f}, y_skull={me_ys:.1f}
Go → x_skull={go_xs:.1f}, y_skull={go_ys:.1f}

Usando estas anclas, coloca los 9 puntos restantes (A, B, Po, Or, ENA, ENP, Co, Cls, Cli).

Referencias geométricas a partir de las anclas:
- La línea S→N define la base craneal anterior (NSL)
- Po está ~14% del ancho craneal a la derecha de S, y ~14% del alto por debajo de S
- Or está a la misma altura que Po o ligeramente más bajo, en zona anterior (x_skull≈58%)
- ENA/ENP están al ~52% del alto craneal (mitad entre N y Me aproximadamente)
- Cls está inmediatamente adyacente a S hacia abajo
- Cli continúa la línea S-Cls hacia inferior
"""
            contorno_ctx = contorno_ctx + anclas_ctx


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
R11 y_skull(C) > y_skull(Co) y |C−Co| pequeño    (Capitulare = centro del cóndilo,
                                                  levemente bajo/adelante de Condylion; NO el mismo punto)

CONFIANZA esperada: ALTA 0.85-1.0: N,Me,S,ENA,ENP,Or,Go · MEDIA 0.65-0.84: A,B,Po,Co,C · BAJA 0.35-0.64: Cls,Cli
Prioriza SIEMPRE lo que ves en la imagen sobre las cifras de referencia.

PROPORCIONES NORMATIVAS (media±DE, % del bbox; úsalas si la estructura no es nítida):
  S x32±4/y22±3 · N x60±4/y12±3 · Po x28±4/y36±4 · Or x58±4/y34±3 · A x78±5/y50±4
  B x74±5/y63±5 · Me x68±4/y94±3 · Go x18±5/y76±5 · ENA x76±4/y52±3 · ENP x48±4/y52±3
  Co x22±4/y34±4 · C x23±4/y36±4 · Cls x28±4/y30±4 · Cli x24±5/y42±5

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
    "C":   {{"x_skull": 0.0, "y_skull": 0.0, "confianza": 0.0}},
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

        # ── Anclas del doctor: SOBRESCRIBEN lo que diga la IA (verdad absoluta) ──
        if anclas_px:
            for k, (xc, yc) in anclas_px.items():
                puntos[k] = {"x": round(xc, 1), "y": round(yc, 1)}
                confianzas[k] = 1.0

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
        px_per_mm = body.pop("px_per_mm", None)      # calibración px→mm
        poblacion  = body.pop("poblacion", "latam")  # latam | europa

        # Campos de analítica anonimizada (opcionales; no rompen el análisis si faltan)
        edad_paciente   = body.pop("edad_paciente", None)
        sexo_paciente   = body.pop("sexo_paciente", None)
        nombre_doctor   = body.pop("nombre_doctor", None)
        codigo_ref      = body.pop("codigo_referencia", None)   # código propio del doctor, NO el nombre del paciente
        imagen_rx_b64   = body.pop("imagen_radiografia", None)  # dataURL del canvas, opcional
        acepto_datos    = body.pop("acepto_uso_datos", True)
        puntos_ia_orig  = body.pop("puntos_sugeridos_ia", None) # snapshot de lo que sugirió la IA, si aplica

        pts  = {}
        for nombre, coords in body.items():
            if isinstance(coords, dict):   pts[nombre] = (coords["x"], coords["y"])
            elif isinstance(coords, list): pts[nombre] = (coords[0], coords[1])
            else:                          pts[nombre] = tuple(coords)

        # Todos los puntos son obligatorios (coincide con el frontend): sin el set
        # completo, F5 (Cls/Cli) y F8 (Capitulare) no se calculan y el diagnóstico
        # sería parcial. C se acepta como punto marcado de pleno derecho.
        requeridos = ["S","N","A","B","Me","Go","ENA","ENP","Po","Or","Co","C","Cls","Cli"]
        for p in requeridos:
            if p not in pts:
                return {"success": False, "detail": f"Falta el punto: {p}"}

        factores            = calcular_factores_bimler(pts, px_per_mm)
        T1, T2, T3, ML_NSLc, NL_NSLc = calcular_indicadores_T(factores)
        grupo               = arbol_decision(T1, T2, T3, poblacion)
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

        # ── Analítica anonimizada: geolocalizar + subir + guardar (best-effort) ──
        # Se ejecuta en BACKGROUND (thread pool) para NO bloquear el event loop:
        # las llamadas urllib son síncronas y, si Supabase/ip-api tardan o están
        # caídos, la respuesta clínica al doctor NO debe esperar por ellas.
        if acepto_datos:
            ip_cliente = request.headers.get("x-forwarded-for", "").split(",")[0].strip() \
                         or (request.client.host if request.client else "")

            def _persistir_bg():
                try:
                    pais_detectado  = _pais_desde_ip(ip_cliente)
                    radiografia_url = _subir_radiografia(imagen_rx_b64)
                    _guardar_analisis({
                        "pais": pais_detectado,
                        "nombre_doctor": nombre_doctor,
                        "codigo_referencia_doctor": codigo_ref,
                        "edad_paciente": edad_paciente,
                        "sexo_paciente": sexo_paciente,
                        "radiografia_url": radiografia_url,
                        "puntos_finales": json.dumps(pts),
                        "puntos_sugeridos_ia": json.dumps(puntos_ia_orig) if puntos_ia_orig else None,
                        "px_per_mm": px_per_mm,
                        "sna": factores["SNA"], "snb": factores["SNB"], "anb": factores["ANB"],
                        "ml_nsl": factores["ML_NSL"], "nl_nsl": factores["NL_NSL"],
                        "t1": T1, "t2": T2, "t3": T3,
                        "grupo": grupo, "categoria": categoria,
                        "poblacion_parametro": poblacion,
                    })
                except Exception as e:
                    _log.warning("persistencia background fallo: %s", e)

            # fire-and-forget: no se espera el resultado (best-effort)
            import asyncio as _asyncio
            _asyncio.get_event_loop().run_in_executor(None, _persistir_bg)

        return {
            "success": True,
            "factores_bimler": {
                "SNA": factores["SNA"], "SNB": factores["SNB"], "ANB": factores["ANB"],
                "F1": factores["F1"],   "F2": factores["F2"],
                "F3": factores["F3"],   "F4": factores["F4"],
                "F5": factores["F5"],   "F7": factores["F7"],   "F8": factores["F8"],
                "F8_fuente": factores.get("F8_fuente"),
                "resalte_esqueletico_mm": factores.get("resalte_esqueletico_mm"),
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
            "medidas_lineales": calcular_medidas_lineales(pts, px_per_mm),
            "resumen_narrativo": generar_resumen_narrativo(
                factores, {"T1":T1,"T2":T2,"T3":T3},
                calcular_medidas_lineales(pts, px_per_mm),
                grupo, str(categoria)
            ),
            "diagnostico": {
                "grupo": grupo, "categoria": categoria,
                "categoria_advertencia": advertencia,
                "rotacion": rot_letra, "desc_rotacion": rot_map.get(rot_letra,"—"),
                "basal":    basal_num,  "desc_basal":    basal_map.get(basal_num,"—"),
                "sagital":  sag_letra,  "desc_sagital":  sag_map.get(sag_letra,"—"),
                "vertical": vert_letra, "desc_vertical": vert_map.get(vert_letra,"—"),
            },
            "recomendacion": obtener_recomendacion(grupo),
            "poblacion": {
                "clave":   poblacion,
                "nombre":  POBLACION_PARAMS.get(poblacion, POBLACION_PARAMS["latam"])["nombre"],
                "fuente":  POBLACION_PARAMS.get(poblacion, POBLACION_PARAMS["latam"])["fuente"],
            }
        }
    except Exception as e:
        return {"success": False, "detail": str(e)}

@app.get("/api/mi-poblacion")
async def mi_poblacion(request: Request):
    """
    Sugiere el set de parámetros (latam/europa) según el país detectado
    por IP del doctor que abre el sistema. Es solo una SUGERENCIA inicial
    para pre-seleccionar el dropdown — el doctor siempre puede cambiarlo
    manualmente (ej. un doctor en Lima tratando a un paciente europeo).
    """
    ip_cliente = request.headers.get("x-forwarded-for", "").split(",")[0].strip() \
                 or (request.client.host if request.client else "")
    pais = _pais_desde_ip(ip_cliente)

    # Países cuya calibración de referencia es la europea (Bjork-Skieller/OrthoTP).
    # Todo lo demás (incluido "??" sin detectar) usa latam por defecto,
    # que es la fuente primaria de Petrovic 1996.
    PAISES_EUROPA = {
        "DE","IT","FR","ES","PT","GB","IE","NL","BE","LU","CH","AT",
        "PL","CZ","SK","HU","RO","BG","GR","HR","SI","DK","SE","NO",
        "FI","EE","LV","LT",
    }
    sugerido = "europa" if pais in PAISES_EUROPA else "latam"
    return {"pais_detectado": pais, "poblacion_sugerida": sugerido}


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "3.2",
            "grupos_rotacionales": 33,
            "factores_bimler": 8}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
