"""
OrthoAnalysis — Script de ingesta del Knowledge Base
=====================================================
Extrae texto de TXTs, PDFs digitales y DOCXs (en realidad texto plano),
trocea en fragmentos, genera embeddings con Claude y los guarda en Supabase.

USO:
  1. Poner este archivo en la raíz del repo (junto a main.py)
  2. Asegurarse de tener las variables de entorno:
       ANTHROPIC_API_KEY
       SUPABASE_URL
       SUPABASE_SERVICE_KEY
  3. Ejecutar: python ingestar_knowledge_base.py

SOLO SE EJECUTA UNA VEZ por documento. Si el documento ya fue ingestado,
lo salta automáticamente.
"""

import os
import re
import time
import hashlib
from pathlib import Path
from pypdf import PdfReader
import anthropic
from supabase import create_client

# ── Configuración ────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY")
SUPABASE_URL       = os.environ.get("SUPABASE_URL")
SUPABASE_KEY       = os.environ.get("SUPABASE_SERVICE_KEY")

CARPETA_DOCS       = Path(".")          # carpeta donde están los archivos
TAMANO_FRAGMENTO   = 400               # palabras por fragmento
SUPERPOSICION      = 50                # palabras de superposición entre fragmentos
PAUSA_ENTRE_CALLS  = 0.5              # segundos entre llamadas a la API (evitar rate limit)

# Mapa de archivos → metadatos (tema, fuente legible)
ARCHIVOS_CONFIG = {
    # TXTs
    "663132977-Analisis-Cefalometrico-de-Bimler-y-Sassouni.txt": {
        "fuente": "Análisis Cefalométrico de Bimler y Sassouni — R1 Ortodoncia",
        "tema": "bimler",
        "acceso": "global"
    },
    "549077346-petrovic-clasificacion.txt": {
        "fuente": "Identificación grupo auxológico Petrovic-Lavergne — UCE Ecuador",
        "tema": "tipos_rotacionales",
        "acceso": "global"
    },
    "631023927-petrovic.txt": {
        "fuente": "Petrovic — KIRU 2022, Vol. 19(1): 36-45",
        "tema": "tipos_rotacionales",
        "acceso": "global"
    },
    "581357172-IDENTIFICACION-DE-TIPOS-ROTACIONALES-Y-CATEGORIAS-AUXOLOGICAS-COMO-HERRAMIENTA-DIAGNOSTICA-EN-LA-PREDICCION-DEL-POTENCIAL-DE-CRECIMIENTO-MANDIBULAR.txt": {
        "fuente": "Identificación de Tipos Rotacionales y Categorías Auxológicas — 2014",
        "tema": "tipos_rotacionales",
        "acceso": "global"
    },
    "270275239-Identificacion-de-Tipos-Rotacionales-y-Categorias-Auxologicas-Como-Herramienta-Diagnostica-en-La-Prediccion-Del-Potencial-de-Crecimiento-Mandibular.txt": {
        "fuente": "Identificación de Tipos Rotacionales y Categorías Auxológicas — Vol 52 N°3 2014",
        "tema": "tipos_rotacionales",
        "acceso": "global"
    },
    "384846322-La-mandibula-su-rotacion-pdf.txt": {
        "fuente": "La Mandíbula: Su Rotación — Lavergne-Petrovic",
        "tema": "rotacion_mandibular",
        "acceso": "global"
    },
    "459980079-5-CEFALOMETRIA-DE-PETROVIC-Y-SCHWARTZ-pptx.txt": {
        "fuente": "Cefalometría de Petrovic y Schwartz — Material docente",
        "tema": "tipos_rotacionales",
        "acceso": "global"
    },
    "507099513-Cefalometri-a-de-petrovic.txt": {
        "fuente": "Cefalometría de Petrovic — Material académico",
        "tema": "tipos_rotacionales",
        "acceso": "global"
    },
    "295281609-analisis-cefalometrico-basico.txt": {
        "fuente": "Análisis Cefalométrico Básico — Prof. Martha Torres, UCV",
        "tema": "cefalometria",
        "acceso": "global"
    },
    "641499029-BJORK-PREDICCION-MANDIBULAR.txt": {
        "fuente": "Björk — Predicción del Crecimiento Mandibular",
        "tema": "rotacion_mandibular",
        "acceso": "global"
    },
    "641857580-Untitled.txt": {
        "fuente": "Material académico — Ortopedia funcional",
        "tema": "cefalometria",
        "acceso": "global"
    },
    "585223363-AULA-PETROVIC-CEDEFACE.txt": {
        "fuente": "Aula Petrovic — CEDEFACE",
        "tema": "tipos_rotacionales",
        "acceso": "global"
    },
    "181569359-Teoria-de-Petrovic-1.txt": {
        "fuente": "Teoría de Petrovic — Material docente",
        "tema": "tipos_rotacionales",
        "acceso": "global"
    },
    "640479084-Crecimiento-mandibular-post-natal.txt": {
        "fuente": "Crecimiento Mandibular Postnatal",
        "tema": "rotacion_mandibular",
        "acceso": "global"
    },
    "444359971-Sebenta-de-Ortodontia.txt": {
        "fuente": "Sebenta de Ortodontia — Universidade Católica (portugués)",
        "tema": "cefalometria",
        "acceso": "global"
    },
    # PDFs con texto digital
    "480465385CefalometriaBimler.pdf": {
        "fuente": "Cefalometría de Bimler — Universidad Antonio Nariño, 2020",
        "tema": "bimler",
        "acceso": "global"
    },
    "463586775APARATOLOGIADESIMOES.pdf": {
        "fuente": "Aparatología de Simões — Ortopedia funcional",
        "tema": "aparatologia",
        "acceso": "global"
    },
    "LUIZA_DINIZ.pdf": {
        "fuente": "Importância da Ortopedia Funcional dos Maxilares — Luíza Diniz 2020 (portugués)",
        "tema": "ortopedia_funcional",
        "acceso": "global"
    },
    # DOCXs (en realidad texto plano)
    "405807022-CEFALOMETRIA-DE-BIMLER-1-docx.docx": {
        "fuente": "Cefalometría de Bimler — Material académico",
        "tema": "bimler",
        "acceso": "global"
    },
    "270275239-Identificacion-de-Tipos-Rotacionales-y-Categorias-Auxologicas-Como-Herramienta-Diagnostica-en-La-Prediccion-Del-Potencial-de-Crecimiento-Mandibular.docx": {
        "fuente": "Identificación de Tipos Rotacionales — Material académico",
        "tema": "tipos_rotacionales",
        "acceso": "global"
    },
}

# ── Funciones de extracción de texto ─────────────────────────────────────────

def extraer_texto_txt(ruta: Path) -> str:
    with open(ruta, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def extraer_texto_pdf(ruta: Path) -> str:
    reader = PdfReader(str(ruta))
    texto = ""
    for page in reader.pages:
        texto += (page.extract_text() or "") + "\n"
    return texto

def extraer_texto_docx_plano(ruta: Path) -> str:
    """
    Estos DOCXs son en realidad texto plano o markdown disfrazado.
    Los leemos directamente como texto.
    """
    with open(ruta, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def extraer_texto(ruta: Path) -> str:
    ext = ruta.suffix.lower()
    if ext == '.txt':
        return extraer_texto_txt(ruta)
    elif ext == '.pdf':
        return extraer_texto_pdf(ruta)
    elif ext == '.docx':
        return extraer_texto_docx_plano(ruta)
    return ""

# ── Fragmentación ─────────────────────────────────────────────────────────────

def limpiar_texto(texto: str) -> str:
    """Limpia saltos de línea múltiples y espacios extras."""
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    texto = re.sub(r' {2,}', ' ', texto)
    texto = re.sub(r'\f', '\n\n', texto)  # form feeds → salto de párrafo
    return texto.strip()

def trocear(texto: str, tamano: int = TAMANO_FRAGMENTO, superposicion: int = SUPERPOSICION) -> list[str]:
    """
    Divide el texto en fragmentos de N palabras con superposición.
    La superposición evita cortar ideas a la mitad entre fragmentos.
    """
    palabras = texto.split()
    fragmentos = []
    i = 0
    while i < len(palabras):
        fin = min(i + tamano, len(palabras))
        fragmento = " ".join(palabras[i:fin])
        if len(fragmento.strip()) > 100:  # ignorar fragmentos muy cortos
            fragmentos.append(fragmento)
        i += tamano - superposicion
    return fragmentos

# ── Embeddings con Claude ─────────────────────────────────────────────────────

def generar_embedding(cliente_anthropic, texto: str) -> list[float]:
    """
    Genera embedding usando el modelo de embeddings de Anthropic.
    Reintentos automáticos si hay rate limit.
    """
    for intento in range(3):
        try:
            response = cliente_anthropic.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1,
                messages=[{"role": "user", "content": texto}],
                system="Respond with just: OK"
            )
            # Nota: Anthropic lanzará su API de embeddings dedicada.
            # Por ahora usamos text-embedding-3 via compatibilidad,
            # o una alternativa: usar el modelo de OpenAI para embeddings
            # y Claude solo para el chatbot.
            # Esta función se actualiza cuando Anthropic lance embeddings.
            raise NotImplementedError("Ver nota en el código")
        except Exception as e:
            if intento < 2:
                time.sleep(2 ** intento)
            else:
                raise e

def generar_embedding_openai_compatible(texto: str) -> list[float]:
    """
    Alternativa práctica: usar text-embedding-3-small de OpenAI para embeddings.
    Es más barato ($0.02/1M tokens) y funciona perfectamente con pgvector.
    El chatbot sigue usando Claude — solo los embeddings usan OpenAI.
    Requiere: pip install openai, variable OPENAI_API_KEY
    """
    import openai
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.embeddings.create(
        input=texto,
        model="text-embedding-3-small"  # 1536 dimensiones, compatible con nuestra tabla
    )
    return response.data[0].embedding

# ── Guardar en Supabase ───────────────────────────────────────────────────────

def documento_ya_ingestado(supabase_client, nombre_archivo: str) -> bool:
    """Verifica si el archivo ya fue procesado — evita duplicados."""
    result = supabase_client.table("knowledge_base") \
        .select("id") \
        .eq("documento", nombre_archivo) \
        .limit(1) \
        .execute()
    return len(result.data) > 0

def guardar_fragmento(supabase_client, texto: str, embedding: list,
                      fuente: str, documento: str, pagina: int,
                      tema: str, acceso: str):
    """Guarda un fragmento con su embedding en Supabase."""
    supabase_client.table("knowledge_base").insert({
        "texto": texto,
        "fuente": fuente,
        "documento": documento,
        "pagina": pagina,
        "tema": tema,
        "acceso": acceso,
        "embedding": embedding
    }).execute()

# ── Proceso principal ─────────────────────────────────────────────────────────

def ingestar_archivo(archivo: str, config: dict,
                     supabase_client, fn_embedding):
    """Procesa un archivo completo: extrae → trocea → embedding → guarda."""

    ruta = CARPETA_DOCS / archivo

    if not ruta.exists():
        print(f"  ⚠️  No encontrado: {archivo}")
        return 0

    if documento_ya_ingestado(supabase_client, archivo):
        print(f"  ⏭️  Ya ingestado: {archivo}")
        return 0

    print(f"  📄 Extrayendo texto...")
    texto_crudo = extraer_texto(ruta)
    texto = limpiar_texto(texto_crudo)

    if len(texto) < 200:
        print(f"  ⚠️  Texto muy corto ({len(texto)} chars) — saltando")
        return 0

    print(f"  ✂️  Troceando ({len(texto)} chars)...")
    fragmentos = trocear(texto)
    print(f"  📦 {len(fragmentos)} fragmentos generados")

    guardados = 0
    for i, fragmento in enumerate(fragmentos):
        try:
            print(f"  🔢 Embedding {i+1}/{len(fragmentos)}...", end="\r")
            embedding = fn_embedding(fragmento)

            guardar_fragmento(
                supabase_client,
                texto=fragmento,
                embedding=embedding,
                fuente=config["fuente"],
                documento=archivo,
                pagina=i + 1,  # número de fragmento como página aproximada
                tema=config["tema"],
                acceso=config.get("acceso", "global")
            )
            guardados += 1
            time.sleep(PAUSA_ENTRE_CALLS)

        except Exception as e:
            print(f"\n  ❌ Error en fragmento {i+1}: {e}")
            continue

    print(f"\n  ✅ {guardados}/{len(fragmentos)} fragmentos guardados")
    return guardados

def main():
    print("=" * 60)
    print("OrthoAnalysis — Ingesta Knowledge Base")
    print("=" * 60)

    # Verificar variables de entorno
    errores = []
    if not ANTHROPIC_API_KEY:
        errores.append("Falta ANTHROPIC_API_KEY")
    if not SUPABASE_URL:
        errores.append("Falta SUPABASE_URL")
    if not SUPABASE_KEY:
        errores.append("Falta SUPABASE_SERVICE_KEY")

    # Verificar si hay OPENAI_API_KEY para embeddings
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        errores.append("Falta OPENAI_API_KEY (necesaria para embeddings)")

    if errores:
        print("\n❌ Variables de entorno faltantes:")
        for e in errores:
            print(f"   - {e}")
        print("\nAgrega estas variables en Railway → tu servicio → Variables")
        return

    # Conectar servicios
    print("\n🔌 Conectando a Supabase...")
    supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✅ Supabase conectado")

    # Función de embedding (OpenAI por ahora)
    fn_embedding = generar_embedding_openai_compatible

    # Procesar cada archivo
    total_fragmentos = 0
    print(f"\n📚 Procesando {len(ARCHIVOS_CONFIG)} archivos...\n")

    for archivo, config in ARCHIVOS_CONFIG.items():
        print(f"\n{'─' * 50}")
        print(f"📖 {archivo[:55]}")
        print(f"   Tema: {config['tema']} | Fuente: {config['fuente'][:50]}")

        n = ingestar_archivo(archivo, config, supabase_client, fn_embedding)
        total_fragmentos += n

    print(f"\n{'=' * 60}")
    print(f"✅ INGESTA COMPLETA")
    print(f"   Total fragmentos guardados: {total_fragmentos}")
    print(f"   Tabla: knowledge_base en Supabase")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
