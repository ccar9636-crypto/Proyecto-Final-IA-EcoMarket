"""Interfaz web (Streamlit) del agente de EcoMarket. Ejecutar: uv run streamlit run app.py"""

import streamlit as st

from agente import AgenteEcoMarket

st.set_page_config(page_title="EcoMarket - Agente de IA", page_icon="Ⓜ️")
st.title("EcoMarket - Asistente de Atención al Cliente")
st.caption("Pregunta por políticas, productos o pedidos, o solicita una devolución.")


@st.cache_resource  # Se crea una sola vez (carga embeddings y base vectorial)
def obtener_agente():
    return AgenteEcoMarket()


agente = obtener_agente()
if "chat" not in st.session_state:
    st.session_state.chat = []

with st.sidebar:
    st.header("⚙️ Configuración")
    modelo_nombre = getattr(agente, "modelo_nombre", "Auto (Groq)")
    st.write(f"**Modelo activo:** `{modelo_nombre}`")
    if st.button("🔄 Reiniciar conversación"):
        st.session_state.chat = []
        st.cache_resource.clear()
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
