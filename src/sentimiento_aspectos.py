"""
sentimiento_aspectos.py
-----------------------
Analisis de sentimiento por aspecto para reseñas de hoteles.
Detecta el sentimiento de cuatro aspectos: comida, servicio, limpieza y precio.

El dataset no trae etiquetas por aspecto, por lo que el sentimiento se estima
en dos pasos: primero se localizan las frases que mencionan cada aspecto
mediante palabras clave, y luego esas frases se evaluan con un modelo de
sentimiento en español ya entrenado.

Uso tipico:

    from sentimiento_aspectos import AnalizadorAspectos

    analizador = AnalizadorAspectos()
    print(analizador.analizar("La comida rica pero el servicio malo y todo sucio"))
"""

import re
import unicodedata
from transformers import pipeline


# Palabras clave que identifican cada aspecto dentro de una reseña
ASPECTOS = {
    "comida": ["comida", "desayuno", "cena", "restaurante", "plato",
               "menu", "buffet", "sabor", "comer"],
    "servicio": ["servicio", "personal", "atencion", "recepcionista",
                 "camarero", "staff", "empleado", "trato"],
    "limpieza": ["limpieza", "limpio", "sucio", "aseo", "habitacion limpia",
                 "baño limpio"],
    "precio": ["precio", "coste", "caro", "barato", "costo", "tarifa", "gasto"],
}


def normalizar(texto):
    """Pasa a minusculas y elimina acentos para comparar palabras clave."""
    texto = texto.lower()
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )


class AnalizadorAspectos:
    """
    Analizador de sentimiento por aspecto.

    Parametros
    ----------
    modelo : str
        Modelo de sentimiento en español usado para evaluar cada fragmento.
    aspectos : dict
        Diccionario de aspecto -> lista de palabras clave.
    """

    def __init__(self, modelo="finiteautomata/beto-sentiment-analysis", aspectos=None):
        self.aspectos = aspectos or ASPECTOS
        self.pipeline = pipeline("sentiment-analysis", model=modelo, top_k=None)

    def analizar(self, texto):
        """Devuelve un diccionario aspecto -> sentimiento (positivo/neutral/negativo)."""
        texto_norm = normalizar(texto)
        # Se corta en oraciones y comas para separar mejor los aspectos
        fragmentos = re.split(r"[.!?,;]", texto_norm)

        resultados = {}
        for aspecto, palabras in self.aspectos.items():
            claves = [normalizar(p) for p in palabras]
            relevantes = [f.strip() for f in fragmentos if any(c in f for c in claves)]

            if not relevantes:
                resultados[aspecto] = "neutral"
                continue

            pred = self.pipeline(". ".join(relevantes)[:512])[0]
            etiqueta = max(pred, key=lambda x: x["score"])["label"]
            if "POS" in etiqueta:
                resultados[aspecto] = "positivo"
            elif "NEG" in etiqueta:
                resultados[aspecto] = "negativo"
            else:
                resultados[aspecto] = "neutral"
        return resultados


if __name__ == "__main__":
    analizador = AnalizadorAspectos()
    ejemplo = "La comida era deliciosa pero el servicio muy malo y la habitacion sucia"
    print(analizador.analizar(ejemplo))
