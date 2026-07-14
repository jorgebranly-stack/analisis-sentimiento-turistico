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

Proyecto_final_PLN/
├── README.md                      
├── requirements.txt               
├── src/
│   ├── preprocesamiento.py        
│   ├── modelado_temas.py          
│   ├── clasificador_clasico.py    
│   ├── clasificador_beto.py       
│   ├── sentimiento_aspectos.py    
│   └── interfaz_gradio.py         
├── notebook/
│   └── Proyecto_Turismo_NLP.ipynb 
└── docs/
    └── ARQUITECTURA.md            

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
Jorge Branly Carrizales Yampara


