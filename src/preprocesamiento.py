"""
preprocesamiento.py
--------------------
Pipeline de preprocesamiento de texto en español, reutilizable en todo
el proyecto (LDA, TF-IDF/Naive Bayes y preparacion de texto para BETO).

Ofrece dos caminos:
  - pipeline_clasico: limpieza, lematizacion y filtrado de stopwords,
    pensado para TF-IDF y LDA.
  - pipeline_transformer: limpieza ligera que conserva puntuacion y
    stopwords, pensado para BETO.

Uso tipico:

    from preprocesamiento import Preprocesador

    prep = Preprocesador()
    df["texto_procesado"]   = df["review_text"].apply(prep.pipeline_clasico)
    df["texto_transformer"] = df["review_text"].apply(prep.pipeline_transformer)
"""

import re
import spacy
import nltk
from nltk.corpus import stopwords


class Preprocesador:
    """
    Pipeline de preprocesamiento de texto en español.

    Parametros
    ----------
    modelo_spacy : str
        Modelo de spaCy a cargar.
    conservar_negaciones : bool
        Evita eliminar palabras de negacion aunque sean stopwords. Es
        necesario para el analisis de sentimiento, donde quitar el "no"
        de "no es bueno" invierte el significado de la frase.
    """

    # Negaciones que se conservan siempre
    NEGACIONES = {
        "no", "nunca", "jamas", "tampoco", "ninguno", "nadie",
        "nada", "sin", "apenas",
    }

    def __init__(self, modelo_spacy="es_core_news_sm", conservar_negaciones=True):
        # Solo se habilita lo necesario para tokenizar y lematizar
        self.nlp = spacy.load(modelo_spacy, disable=["ner", "parser"])

        nltk.download("stopwords", quiet=True)
        self.stopwords = set(stopwords.words("spanish"))
        self.conservar_negaciones = conservar_negaciones

    def limpiar(self, texto):
        """Minusculas y eliminacion de URLs, menciones y puntuacion."""
        texto = texto.lower()
        texto = re.sub(r"http\S+|www\.\S+", "", texto)
        texto = re.sub(r"@\w+", "", texto)
        texto = re.sub(r"[^a-záéíóúñü\s]", "", texto)
        texto = re.sub(r"\s+", " ", texto).strip()
        return texto

    def tokenizar_lematizar(self, texto):
        """Tokeniza y lematiza filtrando stopwords, salvo las negaciones."""
        doc = self.nlp(texto)
        tokens = []
        for token in doc:
            lema = token.lemma_.lower()
            if lema not in self.stopwords or (
                self.conservar_negaciones and lema in self.NEGACIONES
            ):
                tokens.append(lema)
        return tokens

    def pipeline_clasico(self, texto):
        """Texto para modelos clasicos. Devuelve lemas separados por espacio."""
        texto_limpio = self.limpiar(texto)
        tokens = self.tokenizar_lematizar(texto_limpio)
        return " ".join(tokens)

    def pipeline_transformer(self, texto):
        """Texto para BETO. Limpieza ligera que conserva el contexto."""
        texto = texto.lower()
        texto = re.sub(r"http\S+|www\.\S+", "", texto)
        texto = re.sub(r"@\w+", "", texto)
        texto = re.sub(r"\s+", " ", texto).strip()
        return texto


if __name__ == "__main__":
    prep = Preprocesador()
    ejemplo = "El hotel NO estaba limpio, pero el desayuno fue excelente"
    print("Clasico    :", prep.pipeline_clasico(ejemplo))
    print("Transformer:", prep.pipeline_transformer(ejemplo))
