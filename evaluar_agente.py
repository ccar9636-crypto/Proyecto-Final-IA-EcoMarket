"""
Evaluación del comportamiento del agente (Fase 2).
Ejecuta prompts de prueba y verifica si el agente decide bien cuándo usar herramientas y cuándo responder directo.
Guarda los resultados reales en resultados_agente.md. Ejecutar: uv run evaluar_agente.py
"""

from agente import AgenteEcoMarket

# (mensaje, herramientas que DEBEN usarse, herramientas que NO deben usarse)
CASOS = [
    ("Hola", [], ["consultar_base_conocimiento", "verificar_elegibilidad_producto", "generar_etiqueta_devolucion"]),
    ("¿Cuántos días tengo para devolver un producto?", ["consultar_base_conocimiento"], ["generar_etiqueta_devolucion"]),
    ("Quiero devolver el detergente del pedido ECO-1001, llegó defectuoso",
     ["verificar_elegibilidad_producto", "generar_etiqueta_devolucion"], []),
    ("Quiero devolver el cepillo del pedido ECO-1005 por arrepentimiento",
     ["verificar_elegibilidad_producto"], ["generar_etiqueta_devolucion"]),
    ("Quiero devolver la botella del pedido ECO-1008, no me gustó",
     ["verificar_elegibilidad_producto"], ["generar_etiqueta_devolucion"]),
    ("Quiero devolver algo", [], ["verificar_elegibilidad_producto", "generar_etiqueta_devolucion"]),
    ("¿Cuál es el precio del Bitcoin hoy?", [], ["verificar_elegibilidad_producto", "generar_etiqueta_devolucion"]),
]


def nombres(usadas):
    return [u.split("(", 1)[0] for u in usadas]


def evaluar():
    agente = AgenteEcoMarket()
    filas, aciertos = [], 0
    for mensaje, deben, no_deben in CASOS:
        agente.reiniciar()  # Cada caso empieza con una conversación nueva
        respuesta, usadas = agente.responder(mensaje)
        n = nombres(usadas)
        ok = all(h in n for h in deben) and not any(h in n for h in no_deben)
        aciertos += ok
        print(f"\n> {mensaje}\n  herramientas: {usadas}\n  {'OK' if ok else 'FALLA'}: {respuesta}")
        filas.append((mensaje, " → ".join(n) or "Ninguna", respuesta.replace("\n", " ").replace("|", "/"), ok))

    with open("resultados_agente.md", "w", encoding="utf-8") as f:
        f.write(f"# Resultados de la evaluación del agente\n\nModelo: `{agente.modelo_nombre}` · "
                f"Aciertos: **{aciertos}/{len(CASOS)}**\n\n")
        f.write("| Mensaje | Herramientas usadas | Respuesta del agente | ¿Correcto? |\n| :--- | :--- | :--- | :---: |\n")
        for mensaje, herr, resp, ok in filas:
            f.write(f"| {mensaje} | {herr} | {resp} | {'✅' if ok else '❌'} |\n")
    print(f"\nAciertos: {aciertos}/{len(CASOS)}. Resultados guardados en resultados_agente.md")


if __name__ == "__main__":
    evaluar()
