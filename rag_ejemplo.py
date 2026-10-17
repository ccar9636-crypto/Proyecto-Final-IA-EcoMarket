"""
Taller Práctico #2 - Sistema RAG para EcoMarket
Utiliza LangChain, ChromaDB, Embeddings multilingües y el LLM de Groq (Taller 1)
para responder consultas de atención al cliente usando únicamente el contexto recuperado.

Arquitectura (chaining):
    Consulta -> Embedding -> ChromaDB (recuperación) -> Prompt aumentado -> LLM Groq -> Respuesta
"""

import os
import json
from typing import List

import requests
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

# Modelo de embeddings multilingüe explícito (soporte robusto para español)
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# Mensaje exacto cuando el contexto recuperado no alcanza para responder
MENSAJE_SIN_INFORMACION = "No tengo esa información."

_EMBEDDINGS = None


def obtener_embeddings():
    """Instancia (una sola vez) el modelo de embeddings multilingüe."""
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
    return _EMBEDDINGS


def cargar_documentos() -> List[Document]:
    """Carga las fuentes de datos de la carpeta docs/."""
    documentos = []
    
    # 1. Documento no estructurado (Política unificada de devoluciones)
    if os.path.exists("docs/politica_devoluciones.txt"):
        loader = TextLoader("docs/politica_devoluciones.txt", encoding="utf-8")
        docs_politica = loader.load()
        for d in docs_politica:
            d.metadata["fuente"] = "Politica de Devoluciones"
        documentos.extend(docs_politica)

    # 2. Documento estructurado (Inventario JSON)
    if os.path.exists("docs/inventario_productos.json"):
        with open("docs/inventario_productos.json", "r", encoding="utf-8") as f:
            inventario = json.load(f)
            for item in inventario:
                devolucion = "Admite devolucion" if item.get("admite_devolucion", True) else "NO admite devolucion"
                contenido = (
                    f"Producto ID: {item['id']}. Nombre: {item['nombre']}. Categoria: {item['categoria']}. "
                    f"Precio: ${item['precio']} USD. Stock: {item['stock']} unidades. "
                    f"{devolucion}. Descripcion: {item['descripcion']}"
                )
                documentos.append(Document(page_content=contenido, metadata={"fuente": "Inventario", "id": item["id"]}))

    # 3. Documento semi-estructurado (Preguntas Frecuentes JSON)
    if os.path.exists("docs/preguntas_frecuentes.json"):
        with open("docs/preguntas_frecuentes.json", "r", encoding="utf-8") as f:
            faqs = json.load(f)
            for faq in faqs:
                contenido = f"Pregunta Frecuente: {faq['pregunta']} Respuesta: {faq['respuesta']}"
                documentos.append(Document(page_content=contenido, metadata={"fuente": "Preguntas Frecuentes"}))

    # 4. Pedidos del Taller 1 (mock_orders.json)
    if os.path.exists("docs/mock_orders.json"):
        with open("docs/mock_orders.json", "r", encoding="utf-8") as f:
            pedidos = json.load(f)
            for p in pedidos:
                prod_txt = f" Producto ID: {p.get('producto_id', '')}." if p.get('producto_id') else ""
                contenido = (
                    f"Pedido {p['tracking_number']}. Cliente: {p['cliente']}.{prod_txt} Estado: {p['estado']}. "
                    f"Fecha estimada de entrega: {p['fecha_estimada']}. Enlace de rastreo: {p['url_rastreo']}"
                )
                documentos.append(Document(page_content=contenido, metadata={"fuente": "Pedidos", "id": p["tracking_number"]}))

    return documentos


def crear_sistema_rag(chunk_size: int = 300, chunk_overlap: int = 50, collection_name: str = "ecomarket_knowledge"):
    print("[1/3] Cargando documentos de EcoMarket...")
    documentos = cargar_documentos()
    print(f"   Total de documentos base cargados: {len(documentos)}")

    print(f"\n[2/3] Aplicando Chunking (RecursiveCharacterTextSplitter, chunk_size={chunk_size}, overlap={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = text_splitter.split_documents(documentos)
    print(f"   Total de fragmentos (chunks) generados: {len(chunks)}")

    print(f"\n[3/3] Creando vectores ({EMBEDDING_MODEL}) e indexando en ChromaDB...")
    embeddings = obtener_embeddings()
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name
    )
    print("   Base de datos vectorial indexada exitosamente.")

    return vectorstore


def consultar_rag(vectorstore, pregunta: str, k: int = 3):
    """Etapa de recuperación (Retrieval): devuelve los k fragmentos más similares."""
    print(f"\n[CONSULTA] Consulta del cliente: '{pregunta}'")
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    resultados = retriever.invoke(pregunta)
    
    print("\n[RESULTADOS] Fragmentos recuperados por similitud semántica (RAG):")
    for i, doc in enumerate(resultados, 1):
        fuente = doc.metadata.get("fuente", "Desconocida")
        print(f"\n--- Resultado #{i} (Fuente: {fuente}) ---")
        print(doc.page_content)
    return resultados


def construir_prompt(pregunta: str, resultados: List[Document]) -> str:
    """Prompt aumentado: restringe al LLM a responder solo con el contexto recuperado."""
    contexto = "\n\n".join(f"[{d.metadata.get('fuente', 'Desconocida')}] {d.page_content}" for d in resultados)
    return (
        "Actúa como un agente de servicio al cliente amable, profesional y empático de la empresa EcoMarket.\n"
        "Responde la pregunta del cliente utilizando ÚNICAMENTE la información del CONTEXTO proporcionado. "
        "No uses conocimiento externo ni inventes datos.\n"
        f"Si el contexto no contiene la información necesaria para responder, responde exactamente: \"{MENSAJE_SIN_INFORMACION}\"\n"
        "Si el pedido está 'Retrasado', ofrece una disculpa sincera e incluye la fecha estimada y el enlace de rastreo.\n"
        "Si el producto no admite devolución, explícalo con empatía y ofrece bonos o notas de crédito como alternativa.\n"
        "Sé conciso: máximo dos párrafos cortos.\n\n"
        f"[CONTEXTO]\n{contexto}\n\n"
        f"[PREGUNTA DEL CLIENTE]\n{pregunta}"
    )


def consultar_llm(prompt: str) -> str:
    """Consulta el LLM de Groq (Llama 3) con fallback entre modelos y auto-descubrimiento."""
    raw_api_key = os.getenv("GROQ_API_KEY")
    if not raw_api_key:
        return "[AVISO] GROQ_API_KEY no configurada. Configure la variable de entorno para ver la respuesta generada por el LLM."

    # Limpiar comillas accidentales, espacios y saltos de línea
    api_key = raw_api_key.strip().strip("'").strip('"')

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    candidate_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ]

    ultimo_error = ""

    for model_name in candidate_models:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 500,
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.ok:
                return response.json()["choices"][0]["message"]["content"].strip()
            else:
                try:
                    err_json = response.json()
                    err_msg = err_json.get("error", {}).get("message", response.text)
                except Exception:
                    err_msg = response.text
                ultimo_error = f"Código {response.status_code}: {err_msg}"
                if response.status_code == 401:
                    return f"Error de autenticación con Groq (401): {err_msg}. Verifique que su GROQ_API_KEY sea válida y esté activa."
        except Exception as e:
            ultimo_error = str(e)
            continue

    # Fallback: Auto-descubrir modelo activo de texto en Groq (como en el Taller 1)
    try:
        models_resp = requests.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
        if models_resp.ok:
            available = [m['id'] for m in models_resp.json().get('data', []) if 'whisper' not in m['id'] and 'guard' not in m['id']]
            if available:
                payload = {
                    "model": available[0],
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 500
                }
                res = requests.post(url, headers=headers, json=payload, timeout=30)
                if res.ok:
                    return res.json()["choices"][0]["message"]["content"].strip()
        else:
            try:
                err_msg = models_resp.json().get("error", {}).get("message", models_resp.text)
            except Exception:
                err_msg = models_resp.text
            return f"Error de autenticación con Groq ({models_resp.status_code}): {err_msg}"
    except Exception as e:
        ultimo_error = str(e)

    return f"Error al consultar Groq: {ultimo_error or 'No se pudo obtener respuesta de ningún modelo.'}"


def responder_rag(vectorstore, pregunta: str, k: int = 3, mostrar_fragmentos: bool = True) -> str:
    """Flujo RAG completo: recuperación + prompt aumentado + generación con LLM."""
    if mostrar_fragmentos:
        resultados = consultar_rag(vectorstore, pregunta, k=k)
    else:
        resultados = vectorstore.as_retriever(search_kwargs={"k": k}).invoke(pregunta)
    prompt = construir_prompt(pregunta, resultados)
    respuesta = consultar_llm(prompt)
    print("\n[RESPUESTA LLM - Groq]")
    print(respuesta)
    return respuesta


if __name__ == "__main__":
    # Inicializar el sistema RAG
    vectorstore = crear_sistema_rag()
    
    # Pruebas dentro de dominio (políticas, productos, pedidos del Taller 1)
    responder_rag(vectorstore, "¿Cuántos días tengo para devolver un producto defectuoso?")
    responder_rag(vectorstore, "¿Tienen cepillos de bambú y cuánto cuestan?")
    responder_rag(vectorstore, "¿Puedo devolver un cepillo de bambú que ya abrí?")
    responder_rag(vectorstore, "¿Cuál es el estado de mi pedido ECO-1003?")
    responder_rag(vectorstore, "¿Los empaques son ecológicos?")

    # Prueba fuera de dominio: el LLM debe indicar que no tiene la información
    responder_rag(vectorstore, "¿Cuál es el precio del Bitcoin hoy?")
