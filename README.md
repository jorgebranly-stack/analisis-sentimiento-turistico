# Análisis de Reseñas de Hoteles en Español

## Descripción del proyecto

Este proyecto fue desarrollado como trabajo final del módulo de **Procesamiento de Lenguaje Natural (PLN)**.

El objetivo es analizar reseñas de hoteles escritas en español para identificar:

* El **sentimiento general** del huésped (positivo, neutral o negativo), usando dos enfoques que se comparan entre sí: un modelo clásico (Naive Bayes) y un modelo neuronal (BETO).
* El **sentimiento por aspecto**: comida, servicio, limpieza y precio.
* Los **temas principales** del corpus mediante *Topic Modeling* (LDA).

Finalmente se construyó una interfaz en **Gradio** donde el usuario pega una reseña y obtiene las estrellas estimadas y el sentimiento de cada aspecto.

---

## Flujo general del proyecto
Carga del dataset (Hugging Face)
↓
Preprocesamiento del texto
↓
Modelado de temas (LDA)
↓
Clasificación de sentimiento
(Naive Bayes vs. BETO)
↓
Sentimiento por aspecto
↓
Interfaz en Gradio
---

## Dataset utilizado

Se utilizó el conjunto **Punta Cana Spanish Reviews**, disponible públicamente en Hugging Face.

**Características**

* Fuente: `beltrewilton/punta-cana-spanish-reviews`
* Idioma: Español
* Total de reseñas: 34.561
* Calificaciones originales: de 1 a 5 estrellas

Las calificaciones se agruparon en tres categorías de sentimiento:

* Negativo (1–2 estrellas)
* Neutral (3 estrellas)
* Positivo (4–5 estrellas)

Como las clases estaban muy desbalanceadas (la mayoría positivas), se aplicó **submuestreo** hasta 1.695 reseñas por clase (5.085 en total), para que los modelos aprendan de forma equilibrada.

El dataset se carga directamente desde Hugging Face con la librería `datasets`, sin necesidad de descargar archivos manualmente.

---

## Estructura del proyecto

```bash
analisis-sentimiento-turistico/
├── src/
│   ├── preprocesamiento.py
│   ├── modelado_temas.py
│   ├── clasificador_clasico.py
│   ├── clasificador_beto.py
│   ├── sentimiento_aspectos.py
│   └── interfaz_gradio.py
├── notebook/
│   └── Proyecto_Turismo_NLP.ipynb
├── docs/
│   ├── ARQUITECTURA.md
│   └── informe.pdf
├── requirements.txt
└── README.md
```
---

## Desarrollo del proyecto

### 1. Preprocesamiento

Antes de entrenar los modelos se prepararon las reseñas. Las tareas principales fueron:

* Conversión del texto a minúsculas.
* Eliminación de URLs, menciones y caracteres especiales.
* Eliminación de palabras vacías (*stopwords*).
* Tokenización y lematización con **spaCy**.
* Conservación de las negaciones, importantes para el análisis de sentimiento.

Se generaron dos versiones del texto: una limpieza exhaustiva para los modelos clásicos (TF-IDF y LDA) y una limpieza ligera para BETO, que aprovecha mejor el contexto completo.

### 2. Modelado de temas

Para descubrir los temas presentes en las reseñas se utilizó el algoritmo **LDA (Latent Dirichlet Allocation)** implementado con **gensim**, trabajando sobre el texto lematizado.

### 3. Clasificación de sentimiento

El proyecto compara dos enfoques sobre el mismo conjunto de prueba:

* **Modelo clásico:** TF-IDF (unigramas y bigramas) con Multinomial Naive Bayes como línea base.
* **Modelo neuronal:** BETO (`dccuchile/bert-base-spanish-wwm-uncased`) afinado para clasificación de sentimiento de tres clases.

### 4. Sentimiento por aspecto

Además del sentimiento general, el sistema detecta el sentimiento de cuatro aspectos: comida, servicio, limpieza y precio. Se localizan las frases que mencionan cada aspecto y se evalúa su sentimiento con un modelo de sentimiento en español.

### 5. Interfaz

La interfaz en **Gradio** ofrece dos modos: análisis de una reseña individual (estrellas estimadas y sentimiento por aspecto) y análisis por lote a partir de un archivo CSV.

---

## Resultados

Ambos modelos se evaluaron sobre el mismo conjunto de prueba (1.017 reseñas balanceadas):

| Modelo | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Naive Bayes (TF-IDF) | 0.734 | 0.735 | 0.734 | 0.734 |
| **BETO (Transformer)** | **0.807** | **0.809** | **0.807** | **0.808** |

El modelo neuronal (BETO) supera al clásico en aproximadamente 7 puntos. La clase **neutral** es la más difícil para ambos modelos, algo esperable por su ambigüedad, pero BETO la mejora notablemente (F1 0.62 → 0.72).

---

## Cómo ejecutar

1. Instalar dependencias:
```bash
   pip install -r requirements.txt
   python -m spacy download es_core_news_sm
```
2. Abrir el notebook en `notebook/` (recomendado en Google Colab con GPU) y ejecutarlo de arriba hacia abajo.
3. La última celda lanza la interfaz de Gradio con un enlace público para probar el sistema.

---

## Autor

Jorge Branly Carrizales Yampara — Proyecto final del módulo de PLN.


