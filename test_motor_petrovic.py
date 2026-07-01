"""
Suite de regresión del motor OrthoAnalysis contra la fuente PRIMARIA:
Figura 14 y Figura 15 de Petrovic-Stutzmann-Lavergne (1996),
transcritas de la tesis UNAM-León 2019 (Mateos González, pp.37-38)
y validadas de forma cruzada con la tesis UCE-Ecuador 2019 (Coba Moreno).

Ejecutar:  python3 test_motor_petrovic.py       (modo standalone)
       o:  pytest test_motor_petrovic.py -q
"""
import importlib.util, math, sys

def _load(path="main.py"):
    spec = importlib.util.spec_from_file_location("mp", path)
    m = importlib.util.module_from_spec(spec); sys.modules["mp"]=m
    spec.loader.exec_module(m); return m

# Permite pasar la ruta del main.py como argumento
MAIN = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".py") else "/tmp/m.py"
m = _load(MAIN)

# ─────────────────────────────────────────────────────────────────────
# Árbol de referencia EXACTO — Figura 14 (umbral latam T1>6, T2 con 0≤T2≤3)
# ─────────────────────────────────────────────────────────────────────
def figura14(T1, T2, T3):
    rot = 'A' if T1 > 6 else ('R' if T1 >= 0 else 'P')
    v   = 'OB' if T2 > 3 else ('N' if T2 >= 0 else 'DB')
    if rot == 'A':
        if v=='OB': return 'A3 MOB' if T3<=1.5 else 'A1 NOB' if T3<=5.5 else 'A1 DOB' if T3<=8.5 else 'A2 DOB'
        if v=='N':  return 'A3 MN'  if T3<=0   else 'A1 NN'  if T3<=4   else 'A1 DN'  if T3<=7   else 'A2 DN'
        return         'A3 MDB' if T3<=-1.5 else 'A1 NDB' if T3<=3 else 'A1 DDB' if T3<=6 else 'A2 DDB'
    if rot == 'R':
        if v=='OB': return 'R3 MOB' if T3<=1 else 'R1 NOB' if T3<=5 else 'R2 DOB'
        if v=='N':  return 'R3 MN'  if T3<=0 else 'R1 NN'  if T3<=4 else 'R2 DN'
        return         'R3 MDB' if T3<=-1 else 'R1 NDB' if T3<=3 else 'R2 DDB'
    if v=='OB': return 'P2 DOB' if T3>=5.5 else 'P1 NOB' if T3>=1 else 'P1 MOB' if T3>=-6 else 'P3 MOB'
    if v=='N':  return 'P2 DN'  if T3>=4   else 'P1 NN'  if T3>=0 else 'P1 MN'  if T3>=-7 else 'P3 MN'
    return         'P2 DDB' if T3>=3 else 'P1 NDB' if T3>=-1 else 'P1 MDB' if T3>=-8 else 'P3 MDB'

# Árbol europa (T1>9, T2 con -1≤T2≤3) — para verificar el selector de población
def figura14_eu(T1, T2, T3):
    rot = 'A' if T1 > 9 else ('R' if T1 >= 0 else 'P')
    v   = 'OB' if T2 > 3 else ('N' if T2 >= -1 else 'DB')
    # sub-rangos T3 idénticos a latam (la población solo cambia T1 y el borde inferior de T2)
    if rot == 'A':
        if v=='OB': return 'A3 MOB' if T3<=1.5 else 'A1 NOB' if T3<=5.5 else 'A1 DOB' if T3<=8.5 else 'A2 DOB'
        if v=='N':  return 'A3 MN'  if T3<=0   else 'A1 NN'  if T3<=4   else 'A1 DN'  if T3<=7   else 'A2 DN'
        return         'A3 MDB' if T3<=-1.5 else 'A1 NDB' if T3<=3 else 'A1 DDB' if T3<=6 else 'A2 DDB'
    if rot == 'R':
        if v=='OB': return 'R3 MOB' if T3<=1 else 'R1 NOB' if T3<=5 else 'R2 DOB'
        if v=='N':  return 'R3 MN'  if T3<=0 else 'R1 NN'  if T3<=4 else 'R2 DN'
        return         'R3 MDB' if T3<=-1 else 'R1 NDB' if T3<=3 else 'R2 DDB'
    if v=='OB': return 'P2 DOB' if T3>=5.5 else 'P1 NOB' if T3>=1 else 'P1 MOB' if T3>=-6 else 'P3 MOB'
    if v=='N':  return 'P2 DN'  if T3>=4   else 'P1 NN'  if T3>=0 else 'P1 MN'  if T3>=-7 else 'P3 MN'
    return         'P2 DDB' if T3>=3 else 'P1 NDB' if T3>=-1 else 'P1 MDB' if T3>=-8 else 'P3 MDB'

# Figura 15 — categoría auxológica (fuente Guercio, confirmada por UCE-Ecuador)
FIG15 = {1:{'P2D'}, 2:{'A2D','P1N'}, 3:{'R2D'}, 4:{'R1N'},
         5:{'A1D','A1N','P1M','R3M'}, 6:{'A3M','P3M'}}
def cat_fig15(grupo):
    tb = grupo.split()[0][:2] + grupo.split()[1][0]
    for c,s in FIG15.items():
        if tb in s: return c
    return None

def _frange(a,b,step):
    x=a
    while x<=b+1e-9:
        yield round(x,2); x+=step

# ── TEST 1: árbol latam == Figura 14 (barrido exhaustivo) ──
def test_arbol_latam_vs_figura14():
    bad=0
    for T1 in _frange(-15,20,0.5):
        for T2 in _frange(-8,10,0.5):
            for T3 in _frange(-12,12,0.5):
                if m.arbol_decision(T1,T2,T3,"latam") != figura14(T1,T2,T3): bad+=1
    assert bad==0, f"{bad} discrepancias latam vs Figura14"

# ── TEST 2: árbol europa == Figura 14 con T1>9 y T2>=-1 ──
def test_arbol_europa():
    bad=0
    for T1 in _frange(-15,20,0.5):
        for T2 in _frange(-8,10,0.5):
            for T3 in _frange(-12,12,0.5):
                if m.arbol_decision(T1,T2,T3,"europa") != figura14_eu(T1,T2,T3): bad+=1
    assert bad==0, f"{bad} discrepancias europa"

# ── TEST 3: categorías == Figura 15 para los 33 grupos ──
def test_categorias_vs_figura15():
    for g in m.GRUPOS_33:
        assert m.determinar_categoria(g)[0] == cat_fig15(g), f"cat mal en {g}"

# ── TEST 4: los 33 grupos alcanzables, 0 huérfanos (latam y europa) ──
def test_33_grupos_alcanzables():
    for pob in ("latam","europa"):
        reach=set()
        for T1 in _frange(-20,25,0.5):
            for T2 in _frange(-12,12,0.5):
                for T3 in _frange(-15,15,0.5):
                    reach.add(m.arbol_decision(T1,T2,T3,pob))
        assert set(m.GRUPOS_33)-reach==set(), f"[{pob}] grupos no alcanzables: {set(m.GRUPOS_33)-reach}"
        assert reach-set(m.GRUPOS_33)==set(), f"[{pob}] grupos huérfanos: {reach-set(m.GRUPOS_33)}"

# ── TEST 5: fórmulas T1/T2/T3 (fuente: tesis UNAM p.36, UCE p.29) ──
def test_formulas():
    f={"SNA":80.0,"SNB":78.0,"ANB":2.0,"ML_NSL":34.0,"NL_NSL":9.0}
    T1,T2,T3,MLc,NLc = m.calcular_indicadores_T(f)
    assert MLc == round(192-2*78.0,2)            # 36.0
    assert NLc == round(34.0/2-7,2)              # 10.0
    assert T1 == round(MLc-34.0,2)               # 2.0
    assert T2 == round(NLc-9.0,2)                # 1.0
    assert T3 == 2.0                             # ANB

# ── TEST 6: casos reales conocidos (OrthoTP + tesis) ──
def test_casos_reales():
    casos = [
        # (SNA,SNB,ML_NSL,NL_NSL, grupo_esperado_latam)
        (78.62,77.20,32.62,11.46, "R1 NDB"),   # Nicolás
        (77.03,74.83,38.37,13.60, "R1 NDB"),   # caso 2 tabla
    ]
    for SNA,SNB,ML,NL,eg in casos:
        f={"SNA":SNA,"SNB":SNB,"ANB":round(SNA-SNB,2),"ML_NSL":ML,"NL_NSL":NL}
        T1,T2,T3,_,_ = m.calcular_indicadores_T(f)
        assert m.arbol_decision(T1,T2,T3,"latam")==eg, f"{eg} != {m.arbol_decision(T1,T2,T3,'latam')}"

# ── TEST 7: caso ejemplo bibliográfico Tamayo (T1=13,T2=0.5,T3=6.5 -> A1 DN) ──
def test_caso_tamayo():
    assert m.arbol_decision(13,0.5,6.5,"latam")=="A1 DN"
    assert m.arbol_decision(13,0.5,6.5,"europa")=="A1 DN"

# ── TEST 8: selector de población cambia el resultado en 6<T1<=9 ──
def test_selector_poblacion():
    for T1 in (6.5,7,8,9):
        assert m.arbol_decision(T1,1,2,"latam")[0]=="A"
        assert m.arbol_decision(T1,1,2,"europa")[0]=="R"

# ── TEST 9: robustez None/NaN (no debe caer silenciosamente en 'P') ──
def test_robustez_nan():
    import math as _mm
    g = m.arbol_decision(float('nan'),1,2,"latam")
    assert not g.startswith("P"), f"NaN cayó en {g} (debería avisar datos incompletos)"

if __name__=="__main__":
    tests=[v for k,v in sorted(globals().items()) if k.startswith("test_")]
    ok=0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}"); ok+=1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(tests)} tests OK")
