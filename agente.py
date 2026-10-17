"""
Agente de IA de EcoMarket (Proyecto Final).

Arquitectura: un solo agente (LangChain + LLM de Groq con "tool calling") que DECIDE en cada mensaje:
  - consultar_base_conocimiento  -> ruta RAG (Taller 2) para preguntas informativas
  - verificar_elegibilidad_producto -> acción: revisa si se puede devolver
  - generar_etiqueta_devolucion     -> acción: emite la etiqueta
  - ninguna herramienta             -> responde directo (saludos, charla)

Ciclo: Usuario -> LLM -> ¿pide herramienta? -> se ejecuta -> resultado vuelve al LLM -> ... -> respuesta final
"""

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from herramientas import HERRAMIENTAS_ACCION
from rag_ejemplo import MENSAJE_SIN_INFORMACION, crear_sistema_rag

load_dotenv()

import requests

MODELOS_PRIORITARIOS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]
MAX_PASOS = 5  # Tope de vueltas del ciclo, evita bucles infinitos

SYSTEM_PROMPT = f"""Eres el agente de servicio al cliente de EcoMarket: amable, profesional y empático.
Reglas:
1. Para preguntas sobre políticas, productos, precios, FAQ o estado de pedidos, usa consultar_base_conocimiento.
   Responde SOLO con esa información; si no alcanza, responde exactamente: "{MENSAJE_SIN_INFORMACION}"
2. Para devoluciones: primero llama verificar_elegibilidad_producto. Solo si es elegible, llama generar_etiqueta_devolucion.
3. Si el cliente menciona el nombre del producto (ej. "detergente", "cepillo") pero no su código PROD-xxx, usa primero consultar_base_conocimiento para averiguar el ID del producto o del pedido. Si te falta el número de pedido o el motivo, pregúntalo amablemente. Nunca inventes datos.
4. Si no es elegible, explica el motivo con empatía y ofrece bono o nota de crédito.
5. Si una herramienta falla, informa el problema con claridad. Saludos y charla: responde sin herramientas.
Responde en español, máximo dos párrafos cortos."""


def seleccionar_modelo_disponible(api_key: str) -> str:
    """Detecta el mejor modelo con soporte para tool calling disponible en la cuenta."""
    try:
        r = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=5,
        )
        if r.ok:
            disponibles = {m["id"] for m in r.json().get("data", [])}
            for cand in MODELOS_PRIORITARIOS:
                if cand in disponibles:
                    return cand
            # Si ninguno de los preferidos está, seleccionar el primer modelo de texto no-audio
            for m in r.json().get("data", []):
                mid = m["id"]
                if "whisper" not in mid and "guard" not in mid and "orpheus" not in mid:
                    return mid
    except Exception:
        pass
    return MODELOS_PRIORITARIOS[0]


class AgenteEcoMarket:
    def __init__(self):
        self.vectorstore = crear_sistema_rag()  # Reutiliza el RAG del Taller 2

        @tool
        def consultar_base_conocimiento(pregunta: str) -> str:
            """Busca en políticas, inventario (con ids PROD-xxx), preguntas frecuentes y pedidos de EcoMarket."""
            docs = self.vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(pregunta)
            return "\n\n".join(f"[{d.metadata.get('fuente')}] {d.page_content}" for d in docs)

        self.herramientas = {t.name: t for t in [consultar_base_conocimiento, *HERRAMIENTAS_ACCION]}
        api_key = os.getenv("GROQ_API_KEY", "").strip().strip("'\"")
        modelo_elegido = seleccionar_modelo_disponible(api_key)
        self.modelo_nombre = modelo_elegido
        llm = ChatGroq(model=modelo_elegido, temperature=0, api_key=api_key)
        self.llm = llm.bind_tools(list(self.herramientas.values()))
        self.historial = [SystemMessage(content=SYSTEM_PROMPT)]

    def responder(self, texto: str):
        """Devuelve (respuesta, lista_de_herramientas_usadas)."""
        self.historial.append(HumanMessage(content=texto))
        usadas = []
        try:
            for _ in range(MAX_PASOS):
                ia = self.llm.invoke(self.historial)
                self.historial.append(ia)
                if not ia.tool_calls:  # El LLM decidió responder directamente
                    return ia.content, usadas
                for llamada in ia.tool_calls:
                    usadas.append(f"{llamada['name']}({llamada['args']})")
                    try:
                        salida = str(self.herramientas[llamada["name"]].invoke(llamada["args"]))
                    except Exception as e:  # Un fallo de herramienta no tumba el agente
                        salida = f"Error en la herramienta: {e}"
                    self.historial.append(ToolMessage(content=salida, tool_call_id=llamada["id"]))
            return "Lo siento, no pude completar la solicitud. Intenta reformularla.", usadas
        except Exception as e:
            return f"Ocurrió un error al contactar al modelo: {e}", usadas


if __name__ == "__main__":
    agente = AgenteEcoMarket()
    for prueba in ["Hola", "¿Cuántos días tengo para devolver un producto?",
                   "Quiero devolver el detergente del pedido ECO-1001, llegó defectuoso",
                   "Quiero devolver el cepillo del pedido ECO-1005, por arrepentimiento"]:
        r, t = agente.responder(prueba)
        print(f"\n> {prueba}\n  herramientas: {t}\n  {r}")
