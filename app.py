"""Interfaz web (Streamlit) del agente de EcoMarket. Ejecutar: uv run streamlit run app.py"""

import streamlit as st

from agente import AgenteEcoMarket, langsmith_activo
from rag_ejemplo import crear_sistema_rag

st.set_page_config(page_title="EcoMarket - Agente de IA", page_icon="♻️")
st.title("EcoMarket - Asistente de Atención al Cliente")
st.caption("Pregunta por políticas, productos o pedidos, o solicita una devolución.")


@st.cache_resource  # Se crea una sola vez para todos los usuarios (carga embeddings y base vectorial)
def obtener_vectorstore():
    return crear_sistema_rag()


# Cada sesión del navegador tiene su propio agente, así la memoria del chat no se mezcla entre usuarios
if "agente" not in st.session_state:
    try:
        st.session_state.agente = AgenteEcoMarket(obtener_vectorstore())
    except ValueError as e:  # Falta la clave de Groq
        st.error(f"⚠️ {e}")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ No se pudo iniciar el agente: {e}")
        st.stop()
agente = st.session_state.agente
if "chat" not in st.session_state:
    st.session_state.chat = []

with st.sidebar:
    st.header("⚙️ Configuración")
    st.write(f"**Modelo activo:** `{agente.modelo_nombre}`")
    st.write(f"**Trazas LangSmith:** {'✅ activas' if langsmith_activo() else '⚪ desactivadas'}")
    if st.button("🔄 Reiniciar conversación"):
        st.session_state.chat = []
        agente.reiniciar()
        st.rerun()

    st.divider()
    st.header("Ejemplos para probar")
    st.markdown(
        "- ¿Cuántos días tengo para devolver un producto?\n"
        "- Quiero devolver el detergente del pedido ECO-1001, llegó defectuoso\n"
        "- Quiero devolver el cepillo del pedido ECO-1005\n"
        "- Devolver la botella del pedido ECO-1008\n"
        "- ¿Cuál es el estado de mi pedido ECO-1003?"
    )

for rol, texto, tools in st.session_state.chat:
    with st.chat_message(rol):
        st.write(texto)
        if tools:
            with st.expander("🔧 Herramientas usadas"):
                st.write(tools)

if pregunta := st.chat_input("Escribe tu mensaje..."):
    with st.chat_message("user"):
        st.write(pregunta)
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            respuesta, tools = agente.responder(pregunta)
        st.write(respuesta)
        if tools:
            with st.expander("🔧 Herramientas usadas"):
                st.write(tools)
    st.session_state.chat += [("user", pregunta, None), ("assistant", respuesta, tools)]
