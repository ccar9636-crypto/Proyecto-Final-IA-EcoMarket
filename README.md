# Taller Práctico #2: Sistema RAG para EcoMarket
**Caso de Estudio:** Optimización de la Atención al Cliente en E-commerce mediante Generación Aumentada por Recuperación (RAG).

**Integrantes:**
- Deibi Bastidas Cerón.
- Camilo Arciniegas Forero.

Este taller es la evolución del [Taller 1 (ecomarket-ai-taller1)](https://github.com/ccar9636-crypto/ecomarket-ai-taller1): se reutiliza el LLM de **Groq (Llama 3)**, la base de pedidos `mock_orders.json` y las reglas de devolución, y se les incorpora una capa de recuperación (RAG) para que el modelo responda solo con información verificable de EcoMarket.

---

## Flujo del Sistema RAG

```text
Documentos (docs/) -> Chunking -> Embedding multilingüe -> ChromaDB      (Indexación, una sola vez)

Pregunta del cliente -> Embedding -> ChromaDB (top-k fragmentos)
                     -> Prompt aumentado (rol + reglas + contexto + pregunta)
                     -> LLM Groq (Llama 3) -> Respuesta                  (Consulta, cada vez)
```

1. **Indexación:** los documentos se cargan, se segmentan en chunks, cada chunk se convierte en un vector con `intfloat/multilingual-e5-large` y se almacena en ChromaDB.
2. **Recuperación (Retrieval):** la pregunta del cliente se convierte en vector con el mismo modelo y ChromaDB devuelve los `k` fragmentos más cercanos.
3. **Generación (Chaining):** los fragmentos se insertan en un prompt que obliga al LLM a responder **únicamente con el contexto recuperado**. Si el contexto no alcanza, debe responder exactamente: `No tengo esa información.`

El modelo de embeddings **busca**; el LLM **redacta**. Son componentes distintos conectados por el prompt.

---

## Fase 1: Selección de Componentes Clave del Sistema RAG

### 1. Modelo de Embeddings
**Selección:** `intfloat/multilingual-e5-large` (código abierto, ejecutado localmente con FastEmbed). Es el modelo fijado en `rag_ejemplo.py`.

| Criterio | `intfloat/multilingual-e5-large` (elegido) | `text-embedding-3-small` (OpenAI) | `BAAI/bge-small-en-v1.5` (default de FastEmbed) |
| :--- | :--- | :--- | :--- |
| Tipo | Open source | Propietario (API) | Open source |
| Idioma español | Excelente (multilingüe, ~100 idiomas) | Muy bueno | Deficiente (optimizado para inglés) |
| Costo | Gratis (solo cómputo local) | ~$0.02 por millón de tokens | Gratis |
| Privacidad | Datos no salen del equipo | Datos enviados a terceros | Datos no salen del equipo |
| Infraestructura | CPU/RAM local (~2 GB de modelo) | Ninguna, requiere API Key | CPU local (muy ligero) |
| Latencia | Media (inferencia local) | Dependiente de red | Baja |

**Justificación:** EcoMarket atiende clientes en español, por lo que descartamos el modelo por defecto de FastEmbed (entrenado para inglés). Entre un modelo propietario y uno abierto elegimos el abierto porque no genera costo por token, evita enviar datos de pedidos y clientes (información personal) a terceros y no requiere una API Key adicional. Su desventaja es el mayor consumo de memoria y la primera descarga (~2 GB).

### 2. Base de Datos Vectorial
**Selección:** `ChromaDB` (prototipado) con ruta de migración a `Qdrant` o `Pinecone` (producción).

| Criterio | ChromaDB (elegido) | Pinecone | Weaviate / Qdrant |
| :--- | :--- | :--- | :--- |
| Despliegue | Embebida en Python, sin servidor | SaaS administrado | Servidor propio o nube |
| Costo | Gratis | Plan gratuito limitado, luego pago continuo | Gratis (self-hosted) o pago (cloud) |
| Escalabilidad | Baja/media (un nodo) | Muy alta | Alta |
| Búsqueda híbrida (vectorial + palabras clave) | Limitada | Disponible | Nativa |
| Integración LangChain | Nativa | Nativa | Nativa |
| Dependencia de terceros | Ninguna | Alta | Media |

**Justificación:** el catálogo de EcoMarket para este prototipo es pequeño; ChromaDB ofrece configuración inmediata y costo cero. Si el catálogo y el volumen de consultas crecen, se migraría a Qdrant/Weaviate para búsqueda híbrida (útil para números de guía como `ECO-1003`, que una búsqueda puramente semántica puede confundir) o a Pinecone si se prefiere un servicio administrado.

### 3. LLM de Generación
**Selección:** Llama 3 vía API de **Groq** (`llama-3.3-70b-versatile` con respaldo `llama-3.1-8b-instant`), el mismo proveedor usado en el Taller 1, con `temperature=0` para respuestas deterministas y apegadas al contexto.

---

## Fase 2: Creación de la Base de Conocimiento de Documentos

### 1. Documentos Seleccionados (carpeta `docs/`)
| Documento | Tipo | Contenido |
| :--- | :--- | :--- |
| `politica_devoluciones.txt` | No estructurado | Política **unificada**: 30 días para productos estándar, sin devolución para higiene personal y perecederos, alternativas (bono/nota de crédito), costos de envío y reembolsos. |
| `inventario_productos.json` | Estructurado | Productos, precios, stock, categoría y si admiten devolución. |
| `preguntas_frecuentes.json` | Semi-estructurado | Preguntas y respuestas sobre empaques, pagos y envíos rurales. |
| `mock_orders.json` | Estructurado (Taller 1) | 10 pedidos con número de guía, estado, fecha estimada y enlace de rastreo. |

**Unificación de la política:** las reglas del Taller 1 (higiene personal y perecederos no se devuelven; los productos sostenibles estándar sí dentro de 30 días; se ofrecen bonos o notas de crédito) se integraron en `politica_devoluciones.txt` y se reflejan en el inventario (el cepillo de bambú figura como *Higiene personal* y `admite_devolucion: false`), de modo que no existan reglas contradictorias.

### 2. Estrategia de Segmentación (Chunking)
| Estrategia | Ventaja | Desventaja | Uso en el proyecto |
| :--- | :--- | :--- | :--- |
| Tamaño fijo | Simple y predecible | Corta frases a la mitad | Descartada |
| **Recursiva por separadores** | Respeta párrafos y frases | Tamaño variable | **Elegida** |
| Semántica (por embeddings) | Cortes por cambio de tema | Mayor costo de cómputo | Mejora futura |

* **Parámetros:** `chunk_size=300`, `chunk_overlap=50`, separadores `["\n\n", "\n", ". ", " "]`.
* **Justificación:** con 300 caracteres cada chunk representa una idea completa (una cláusula de política, un producto o un pedido) y el vector no se diluye en varios temas. El solapamiento de 50 caracteres (~17 %) evita perder contexto en los bordes. Los documentos JSON se convierten a una frase por registro antes de segmentar, por lo que cada producto o pedido queda en un chunk propio.

### 3. Proceso de Indexación
1. **Carga:** se leen los 4 archivos de `docs/` (17 documentos base).
2. **Fragmentación:** `RecursiveCharacterTextSplitter` genera 25 chunks.
3. **Vectorización:** cada chunk pasa por `multilingual-e5-large`.
4. **Almacenamiento:** vectores y metadatos (`fuente`, `id`) se insertan en ChromaDB.

---

## Fase 3: Integración y Ejecución de Código

Archivos principales:
* `rag_ejemplo.py`: sistema RAG completo (recuperación + prompt + LLM Groq).
* `evaluacion_metricas.py`: métricas de recuperación (Hit Rate@K, latencia) y gráfico `metricas_rag.png`.

### Prompt de generación (restringido al contexto)
El prompt asigna el rol de agente de EcoMarket, ordena responder **solo** con el `[CONTEXTO]`, exige la frase `No tengo esa información.` cuando el contexto no alcanza, y conserva las reglas del Taller 1 (disculpa en pedidos retrasados con fecha y enlace; empatía y alternativas cuando un producto no admite devolución).

### Experimento antes / después
Se evaluó el impacto de variar los hiperparámetros de `chunk_size` y `k` sobre el banco de 6 preguntas de prueba (políticas, productos, pagos, pedido `ECO-1003` y regla de higiene):

| Configuración | chunk_size | overlap | k | Hit Rate |
| :--- | :---: | :---: | :---: | :---: |
| Antes | 100 | 20 | 1 | **66.7 %** |
| Después | 300 | 50 | 3 | **100.0 %** |

**Análisis:** con chunks de 100 caracteres y un solo fragmento recuperado, la información queda cortada o el fragmento correcto no ocupa el primer puesto, y fallan 2 de 6 preguntas. Con 300 caracteres el chunk conserva la idea completa y con `k=3` aumenta la probabilidad de incluir el fragmento correcto, alcanzando 100 %.

### Prueba dentro de dominio y fuera de dominio
`rag_ejemplo.py` ejecuta ambas pruebas al correr el script:

| Tipo | Pregunta | Comportamiento esperado |
| :--- | :--- | :--- |
| Dentro de dominio | "¿Cuál es el estado de mi pedido ECO-1003?" | Recupera el pedido (estado *Retrasado*), el LLM se disculpa e informa fecha estimada y enlace de rastreo. |
| Dentro de dominio | "¿Puedo devolver un cepillo de bambú que ya abrí?" | Recupera la regla de higiene; el LLM explica que no procede y ofrece bono o nota de crédito. |
| Fuera de dominio | "¿Cuál es el precio del Bitcoin hoy?" | Ningún fragmento es relevante; el LLM responde `No tengo esa información.` |

> Las respuestas del LLM dependen de la `GROQ_API_KEY` y del modelo activo en Groq; la etapa de recuperación se puede verificar sin clave con `evaluacion_metricas.py`.

### Limitaciones y Suposiciones
1. **Calidad del contexto:** el LLM solo es tan bueno como los fragmentos recuperados; si la recuperación falla, la respuesta será `No tengo esa información.` o incompleta. Por eso se mide el Hit Rate.
2. **Números de guía:** la búsqueda semántica no es exacta para identificadores como `ECO-1003`. Con pocos pedidos funciona, pero en producción convendría búsqueda híbrida (palabras clave + vectores) o consulta directa a la base de datos.
3. **Datos simulados:** pedidos, inventario y políticas son ficticios (`mock_orders.json`); un sistema real requeriría sincronización en tiempo real con la base transaccional.
4. **Prefijos de E5:** el modelo E5 recomienda anteponer `query:` y `passage:` a los textos; la integración de FastEmbed con LangChain no lo hace de forma automática, lo que podría mejorar marginalmente la precisión.
5. **Recursos y dependencias:** `multilingual-e5-large` requiere ~2 GB de descarga y memoria. Se fija `onnxruntime<1.23` porque versiones recientes rechazan la carga de este modelo desde la caché de FastEmbed.
6. **Dependencia externa:** la generación requiere conexión y clave de Groq; está sujeta a cuotas (*rate limits*) y a cambios en los modelos disponibles (se incluye un respaldo entre modelos).
7. **Escalabilidad:** ChromaDB embebida es adecuada para prototipos; para millones de registros o alta concurrencia se migraría a Qdrant o Pinecone.
8. **Privacidad:** los datos de pedidos incluyen nombres de clientes; al enviar contexto a Groq se comparten con un tercero, por lo que en producción se deberían anonimizar.
9. **Evaluación acotada:** el banco de 6 preguntas valida el prototipo, pero no reemplaza una evaluación amplia (por ejemplo con RAGAS: *faithfulness*, *context precision*).

---

## Instrucciones de Instalación y Ejecución

Este proyecto utiliza **`uv`** como gestor de entornos virtuales y dependencias.

### Requisitos Previos
* Python 3.10 o superior.
* `uv` instalado (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
* Clave gratuita de [Groq Cloud](https://console.groq.com).

### Pasos de Ejecución

1. **Sincronizar el entorno e instalar dependencias:**
   ```bash
   uv sync
   ```

2. **Configurar la clave de Groq:**
   ```bash
   export GROQ_API_KEY="tu_groq_api_key_aqui"
   ```

3. **Ejecutar el sistema RAG completo (recuperación + generación):**
   ```bash
   uv run rag_ejemplo.py
   ```

4. **Ejecutar la evaluación de métricas y generar el gráfico `metricas_rag.png`:**
   ```bash
   uv run evaluacion_metricas.py
   ```

> La primera ejecución descarga el modelo `multilingual-e5-large` (~2 GB); las siguientes usan la caché local.