"""
Evaluador de Métricas de Desempeño del Sistema RAG - EcoMarket

Este módulo independiente evalúa el componente de recuperación (Retrieval)
midiendo métricas cuantitativas clave como Hit Rate@K y el nivel de similitud.
Además, genera y guarda un panel gráfico profesional 'metricas_rag.png'.
No modifica el script principal 'rag_ejemplo.py'.
"""

import time
import matplotlib.pyplot as plt
from rag_ejemplo import crear_sistema_rag


def generar_grafico_metricas(casos_prueba, latencias, hit_rate_global, latencia_promedio, k):
    """Genera y guarda una imagen PNG profesional con los gráficos de evaluación."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Gráfico de Latencia por Consulta
    indices = [f"P{i}" for i in range(1, len(casos_prueba) + 1)]
    bars = ax1.bar(indices, latencias, color='#2b5c8f', alpha=0.85, edgecolor='black')
    ax1.axhline(latencia_promedio, color='red', linestyle='--', linewidth=1.5, label=f'Promedio ({latencia_promedio:.1f} ms)')
    ax1.set_title('Latencia de Búsqueda por Consulta (ms)', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('Caso de Prueba', fontsize=10)
    ax1.set_ylabel('Tiempo (ms)', fontsize=10)
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', linestyle=':', alpha=0.6)

    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}ms',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    # 2. Gráfico de Indicador Hit Rate Global
    colores_pie = ['#28a745', '#dc3545'] if hit_rate_global < 100 else ['#28a745']
    labels = ['Exitoso', 'Fallo'] if hit_rate_global < 100 else ['Exitoso (100%)']
    sizes = [hit_rate_global, 100 - hit_rate_global] if hit_rate_global < 100 else [100]

    ax2.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colores_pie, startangle=90,
            wedgeprops={'edgecolor': 'black', 'linewidth': 1})
    ax2.set_title(f'Tasa de Recuperación Hit Rate @ {k}', fontsize=12, fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig('metricas_rag.png', dpi=300)
    plt.close()
    print("  [GRAFICO] Imagen 'metricas_rag.png' generada y guardada exitosamente.")


def evaluar_sistema_rag():
    print("[EVALUACION] Inicializando el sistema RAG para medicion...")
    vectorstore = crear_sistema_rag()
    
    # Dataset de prueba con preguntas y palabras clave esperadas
    dataset_prueba = [
        {
            "pregunta": "¿Cuántos días tengo para devolver un producto defectuoso?",
            "palabras_clave": ["30 días", "devolución", "defectuoso"],
            "fuente_esperada": "Politica de Devoluciones"
        },
        {
            "pregunta": "¿Tienen cepillos de bambú y cuánto cuestan?",
            "palabras_clave": ["Cepillo de Dientes de Bambú", "4.99"],
            "fuente_esperada": "Inventario"
        },
        {
            "pregunta": "¿Los empaques son ecológicos y biodegradables?",
            "palabras_clave": ["100% biodegradable", "180 días"],
            "fuente_esperada": "Preguntas Frecuentes"
        },
        {
            "pregunta": "¿Cuáles son los métodos de pago aceptados?",
            "palabras_clave": ["Visa", "Mastercard", "PayPal", "PSE"],
            "fuente_esperada": "Preguntas Frecuentes"
        },
        {
            "pregunta": "¿Cuál es el estado de mi pedido ECO-1003?",
            "palabras_clave": ["ECO-1003", "Retrasado"],
            "fuente_esperada": "Pedidos"
        },
        {
            "pregunta": "¿Puedo devolver un producto de higiene personal que ya abrí?",
            "palabras_clave": ["NO admiten devolución", "higiene personal"],
            "fuente_esperada": "Politica de Devoluciones"
        }
    ]

    k = 3  # Número de fragmentos a recuperar por consulta
    aciertos_hit_rate = 0
    tiempos_respuesta = []

    print("\n" + "="*60)
    print(" REPORTE DE EVALUACION DE DESEMPEÑO (RETRIEVAL METRICS)")
    print("="*60)

    for i, test in enumerate(dataset_prueba, 1):
        inicio = time.time()
        
        resultados_con_score = vectorstore.similarity_search_with_score(test["pregunta"], k=k)
        
        latencia = (time.time() - inicio) * 1000  # ms
        tiempos_respuesta.append(latencia)

        acierto = False
        fuentes_recuperadas = []
        
        for doc, score in resultados_con_score:
            fuentes_recuperadas.append(doc.metadata.get("fuente", "Desconocida"))
            contenido = doc.page_content.lower()
            if any(kw.lower() in contenido for kw in test["palabras_clave"]):
                acierto = True

        if acierto:
            aciertos_hit_rate += 1

        print(f"\nCaso de Prueba #{i}: '{test['pregunta']}'")
        print(f"  - Latencia de busqueda: {latencia:.2f} ms")
        print(f"  - Fuentes recuperadas: {fuentes_recuperadas}")
        print(f"  - Hit Rate @ {k}: {'EXITO (1.0)' if acierto else 'FALLO (0.0)'}")

    # Métricas Globales
    hit_rate_global = (aciertos_hit_rate / len(dataset_prueba)) * 100
    latencia_promedio = sum(tiempos_respuesta) / len(tiempos_respuesta)

    print("\n" + "="*60)
    print(" RESUMEN DE METRICAS GLOBALES")
    print("="*60)
    print(f" Total de pruebas ejecutadas: {len(dataset_prueba)}")
    print(f" Hit Rate @ {k} (Recuperacion Exitosa): {hit_rate_global:.1f}%")
    print(f" Latencia Promedio de Busqueda: {latencia_promedio:.2f} ms")
    print("="*60 + "\n")

    # Generación del gráfico
    generar_grafico_metricas(dataset_prueba, tiempos_respuesta, hit_rate_global, latencia_promedio, k)


if __name__ == "__main__":
    evaluar_sistema_rag()
