# Arquitectura del sistema

## Descripcion general

El sistema recibe una reseña de hotel escrita en español y devuelve el sentimiento general del huesped, el sentimiento de cada aspecto (comida, servicio, limpieza y precio) y una estimacion de estrellas.

A lo largo del proyecto se implementaron distintas etapas de procesamiento de lenguaje natural, combinando tecnicas tradicionales y modelos basados en aprendizaje profundo, con el fin de comparar ambos enfoques sobre el mismo conjunto de datos.

## Flujo del sistema

```text
Reseña de hotel
      │
      ▼
Preprocesamiento del texto
      │
      ├───────────────┐
      ▼               ▼
Modelado de temas   Clasificacion de sentimiento
 (LDA)              (Naive Bayes y BETO)
      │               │
      └───────┬───────┘
              ▼
      Sentimiento por aspecto
              │
              ▼
       Interfaz en Gradio
```

---

# Descripcion de cada etapa

## 1. Preprocesamiento

La primera etapa prepara las reseñas antes de usarlas para entrenar los modelos. Las tareas principales son:

* Conversion del texto a minusculas.
* Eliminacion de URLs, menciones y caracteres especiales.
* Tokenizacion.
* Lematizacion mediante spaCy.
* Conservacion de las negaciones, importantes para el analisis de sentimiento.

El objetivo es reducir el ruido de los textos conservando la informacion relevante.

---

## 2. Representacion del texto

No todos los modelos usan el texto de la misma manera, por lo que se preparan dos versiones.

### Texto para Naive Bayes y LDA

Se utiliza un texto mas limpio, sin palabras poco informativas y con lematizacion. Esta representacion facilita que los modelos tradicionales encuentren patrones. Sobre ella se aplica TF-IDF con unigramas y bigramas.

### Texto para BETO

En este caso solo se aplica una limpieza basica. No se eliminan stopwords ni se lematiza, ya que el modelo fue entrenado para comprender el lenguaje en su forma natural y aprovecha mejor el contexto completo, lo que ayuda a detectar sentimientos mixtos.

---

## 3. Modelado de temas

Para identificar los temas principales presentes en las reseñas se utiliza el algoritmo Latent Dirichlet Allocation (LDA), implementado con gensim. Se trabaja con cinco temas, buscando un equilibrio entre interpretacion y coherencia. Los temas descubiertos se relacionan con aspectos como el servicio, la comida y las instalaciones.

---

## 4. Clasificacion de sentimiento

El proyecto compara dos enfoques para clasificar el sentimiento de una reseña en tres clases: negativo, neutral y positivo.

### Modelo clasico

El primer modelo utiliza TF-IDF para representar el texto y Multinomial Naive Bayes como clasificador. Es rapido y sirve como linea base.

### Modelo neuronal

El segundo modelo es BETO (`dccuchile/bert-base-spanish-wwm-uncased`), un modelo tipo BERT pre-entrenado en español y afinado para esta tarea. Aprovecha el mecanismo de atencion para entender la relacion entre las palabras de la frase.

### Comparacion

Ambos modelos se evaluan sobre el mismo conjunto de prueba con las metricas de exactitud, precision, recall y F1. BETO supera al modelo clasico en aproximadamente siete puntos, y mejora especialmente la clase neutral, que es la mas dificil para ambos.

---

## 5. Sentimiento por aspecto

Ademas del sentimiento general, el sistema estima el sentimiento de cuatro aspectos: comida, servicio, limpieza y precio.

Como el dataset no incluye etiquetas por aspecto, el proceso se realiza en dos pasos. Primero se localizan las frases que mencionan cada aspecto mediante palabras clave. Luego esas frases se evaluan con un modelo de sentimiento en español ya entrenado, y se asigna el resultado a cada aspecto.

---

## 6. Interfaz

La etapa final expone el sistema a traves de una interfaz en Gradio con tema oscuro. Ofrece tres modos de uso:

* Reseña individual: el usuario pega una reseña y obtiene las estrellas estimadas y el sentimiento por aspecto.
* Analisis por lote: el usuario sube un archivo CSV y obtiene un resumen con la distribucion de sentimientos y un grafico.
* Metricas del modelo: muestra las estadisticas del corpus y la comparativa entre el modelo clasico y el neuronal.

---

# Decisiones de diseño

* La carga del dataset se hace directamente desde Hugging Face, lo que garantiza que el proyecto sea reproducible sin descargar archivos manualmente.
* Las calificaciones numericas se agrupan en tres clases de sentimiento, mas faciles de predecir que la escala de cinco estrellas.
* El corpus se balancea por submuestreo para que los modelos no se inclinen hacia la clase mayoritaria.
* Se mantienen dos pipelines de preprocesamiento distintos, uno para los modelos clasicos y otro para BETO, segun lo que cada enfoque necesita.
