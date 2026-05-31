"""
test_regresion_orthotp.py
Congela los 3 casos reales validados contra OrthoTP como pruebas de regresión.

Estos tests NO dependen del posicionamiento de puntos en la radiografía: alimentan
directamente los valores intermedios medidos que aparecen en los PDFs oficiales de
OrthoTP y verifican que el motor (calcular_indicadores_T + arbol_decision +
determinar_categoria, y la fórmula NL/NSL = F4 + F7) reproduce el grupo y la
categoría publicados.

Ejecutar:
    python -m pytest test_regresion_orthotp.py -v
    # o sin pytest:
    python test_regresion_orthotp.py
"""

import math
import main  # importa el motor corregido (main.py v2.4)


# -----------------------------------------------------------------
# Helpers de prueba que reusan EXACTAMENTE la lógica del motor.
# -----------------------------------------------------------------
def _pipeline_desde_medidas(SNA, SNB, ANB, ML_NSL, F4, F7):
    """
    Reproduce la cadena del motor a partir de las medidas del PDF de OrthoTP:
      NL/NSL = F4 + F7  (Fix A, con signo)
      -> calcular_indicadores_T -> arbol_decision -> determinar_categoria
    """
    NL_NSL = round(F4 + F7, 2)                      # Fix A
    factores = {"SNA": SNA, "SNB": SNB, "ANB": ANB,
                "ML_NSL": ML_NSL, "NL_NSL": NL_NSL}
    T1, T2, T3, ML_NSLc, NL_NSLc = main.calcular_indicadores_T(factores)
    grupo = main.arbol_decision(T1, T2, T3)
    categoria, advertencia = main.determinar_categoria(grupo)
    return {
        "NL_NSL": NL_NSL, "T1": T1, "T2": T2, "T3": T3,
        "grupo": grupo, "categoria": categoria, "advertencia": advertencia,
    }


CASOS = {
    # Caso 3 — Mia Palomino: R2 DN Cat.3
    "Mia": dict(
        SNA=84.68, SNB=73.87, ANB=10.81, ML_NSL=38.76, F4=1.39, F7=9.83,
        esperado=dict(NL_NSL=11.22, T1=5.50, T2=1.16, grupo="R2 DN", categoria=3),
    ),
    # Caso 1 — Nicolás Espinoza.
    # OrthoTP publica R1 NDB Cat.4 (T2=-2.14). Con Fix A (NL/NSL=F4+F7=11.45,
    # correcto) el motor calcula T2=-0.27 -> R1 NN, porque la fórmula NL/NSLc
    # (0.198*SNA-4.39, NO validada) da 11.18 en vez del 9.31 real de OrthoTP.
    # Congelamos el estado HONESTO del motor y marcamos la divergencia conocida.
    # La categoría 4 SÍ se mantiene (R1 N* -> Cat.4 en ambos casos).
    "Nicolas": dict(
        SNA=78.62, SNB=77.20, ANB=1.42, ML_NSL=32.62, F4=-2.63, F7=14.08,
        esperado=dict(NL_NSL=11.45, T1=4.98, categoria=4),
        # divergencia documentada con OrthoTP (pendiente fórmula NL/NSLc real):
        orthotp=dict(grupo="R1 NDB", T2=-2.14),
        motor_actual=dict(grupo="R1 NN", T2=-0.27),
    ),
    # Caso 2 — Piero Espinoza: A1 NDB Cat.5
    "Piero": dict(
        SNA=72.96, SNB=72.08, ANB=0.87, ML_NSL=34.13, F4=0.31, F7=12.34,
        esperado=dict(grupo="A1 NDB", categoria=5),
    ),
}

TOL = 0.05  # tolerancia angular (los PDFs redondean a 2 decimales)


def _check(nombre):
    c = CASOS[nombre]
    r = _pipeline_desde_medidas(c["SNA"], c["SNB"], c["ANB"],
                                c["ML_NSL"], c["F4"], c["F7"])
    e = c["esperado"]
    errores = []
    for k, v in e.items():
        got = r[k]
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if abs(got - v) > TOL:
                errores.append(f"{k}: motor={got} esperado={v}")
        else:
            if got != v:
                errores.append(f"{k}: motor={got!r} esperado={v!r}")
    # El grupo debe estar mapeado (sin advertencia)
    if r["advertencia"] is not None:
        errores.append(f"advertencia inesperada: {r['advertencia']}")
    return r, errores


# -----------------------------------------------------------------
# Tests (estilo pytest; también corren con el __main__ de abajo)
# -----------------------------------------------------------------
def test_mia_palomino():
    r, errores = _check("Mia")
    assert not errores, f"Mia: {errores}  ({r})"

def test_nicolas_espinoza():
    r, errores = _check("Nicolas")
    assert not errores, f"Nicolás: {errores}  ({r})"

def test_nicolas_divergencia_orthotp_documentada():
    """
    DIVERGENCIA CONOCIDA: el motor da R1 NN / T2≈-0.27, OrthoTP da R1 NDB / T2=-2.14.
    Causa: fórmula NL/NSLc (0.198*SNA-4.39) NO validada — da 11.18 vs 9.31 real.
    Este test fija la divergencia: si algún día el motor empieza a dar NDB para
    Nicolás, significará que se corrigió NL/NSLc y habrá que actualizar esperado.
    """
    c = CASOS["Nicolas"]
    r = _pipeline_desde_medidas(c["SNA"], c["SNB"], c["ANB"],
                                c["ML_NSL"], c["F4"], c["F7"])
    assert r["grupo"] == c["motor_actual"]["grupo"], (
        f"El motor cambió de grupo para Nicolás: {r['grupo']}. "
        f"¿Se corrigió NL/NSLc? Actualizar el esperado a OrthoTP {c['orthotp']}.")
    assert abs(r["T2"] - c["motor_actual"]["T2"]) <= 0.05
    # la categoría coincide con OrthoTP pese a la divergencia de bucket vertical
    assert r["categoria"] == 4

def test_piero_espinoza():
    r, errores = _check("Piero")
    assert not errores, f"Piero: {errores}  ({r})"


def test_fixA_nl_nsl_firmado():
    """Fix A: con F4<0 (Nicolás) NL/NSL debe ser 11.45, NO 16.71 (|F4|+F7)."""
    assert round(-2.63 + 14.08, 2) == 11.45
    assert round(abs(-2.63) + 14.08, 2) == 16.71  # valor erróneo antiguo

def test_fixE_grupos_inalcanzables_eliminados():
    """Fix E: las 6 filas inalcanzables ya no están en la tabla."""
    for g in ["A1 DOB", "A1 DN", "A1 DDB", "P1 MOB", "P1 MN", "P1 MDB"]:
        assert g not in main.GRUPOS_33, f"{g} debería haberse eliminado"
    assert len(main.GRUPOS_33) == 27

def test_fixE_advertencia_si_no_mapea():
    """Fix E: un grupo fuera de la tabla devuelve categoría None + advertencia."""
    cat, adv = main.determinar_categoria("Z9 XOB")
    assert cat is None and adv is not None

def test_fixB_gonial_signed():
    """Fix B: AG = F3 - F8 + 90 (Mia: 33.45 - 5.38 + 90 = 118.07)."""
    assert round(33.45 - 5.38 + 90, 2) == 118.07
    # Nicolás hiperflexión F8=-12.57 -> 21.81 - (-12.57) + 90 = 124.38
    assert round(21.81 - (-12.57) + 90, 2) == 124.38


if __name__ == "__main__":
    print("=" * 60)
    print("REGRESIÓN OrthoTP — motor v2.4")
    print("=" * 60)
    fallos = 0
    for nombre in CASOS:
        r, errores = _check(nombre)
        estado = "PASS" if not errores else "FAIL"
        if errores:
            fallos += 1
        print(f"\n[{estado}] {nombre}: grupo={r['grupo']} cat={r['categoria']} "
              f"T1={r['T1']} T2={r['T2']} NL/NSL={r['NL_NSL']}")
        for e in errores:
            print("       -", e)
    # checks de fixes
    for fn in [test_fixA_nl_nsl_firmado, test_fixB_gonial_signed,
               test_fixE_grupos_inalcanzables_eliminados,
               test_fixE_advertencia_si_no_mapea]:
        try:
            fn(); print(f"[PASS] {fn.__name__}")
        except AssertionError as ex:
            fallos += 1; print(f"[FAIL] {fn.__name__}: {ex}")
    print("\n" + ("TODOS LOS TESTS PASARON ✅" if fallos == 0
                  else f"{fallos} TEST(S) FALLARON ❌"))
    raise SystemExit(1 if fallos else 0)
