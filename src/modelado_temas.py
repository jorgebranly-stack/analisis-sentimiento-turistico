"""
modelado_temas.py
------------------
Modelado de temas (LDA) sobre el corpus de reseñas ya preprocesado.

Requiere una columna de texto preprocesado (lemas separados por espacio),
tal como la genera Preprocesador.pipeline_clasico() en preprocesamiento.py.

Uso tipico:

    from modelado_temas import ModeladorTemas

    modelador = ModeladorTemas(num_temas=5)
    modelador.entrenar(df["texto_procesado"])
    modelador.mostrar_temas()

    # Tema dominante de cada reseña, para agregar como columna:
    df["tema_id"] = modelador.temas_por_documento(df["texto_procesado"])
"""

from gensim import corpora
from gensim.models import LdaModel


class ModeladorTemas:
    """
    Envoltorio sobre gensim LdaModel para el modelado de temas.

    Parametros
    ----------
    num_temas : int
        Numero de topicos a descubrir.
    passes : int
        Numero de pasadas de entrenamiento sobre el corpus.
    """

    def __init__(self, num_temas=5, passes=10, random_state=42):
        self.num_temas = num_temas
        self.passes = passes
        self.random_state = random_state
        self.diccionario = None
        self.corpus_bow = None
        self.modelo = None

    def _preparar_corpus(self, textos_procesados):
        """Construye el diccionario y el corpus BOW a partir del texto."""
        tokens_por_doc = [t.split() for t in textos_procesados]

        self.diccionario = corpora.Dictionary(tokens_por_doc)
        # Descarta palabras en menos de 5 documentos o en mas del 50 por ciento
        # del corpus, por ser demasiado genericas para definir un tema
        self.diccionario.filter_extremes(no_below=5, no_above=0.5)

        self.corpus_bow = [self.diccionario.doc2bow(tokens) for tokens in tokens_por_doc]
        return tokens_por_doc

    def entrenar(self, textos_procesados):
        """Entrena el modelo LDA sobre la columna de texto preprocesado."""
        self._preparar_corpus(textos_procesados)

        self.modelo = LdaModel(
            corpus=self.corpus_bow,
            id2word=self.diccionario,
            num_topics=self.num_temas,
            passes=self.passes,
            random_state=self.random_state,
        )
        return self.modelo

    def mostrar_temas(self, num_palabras=10):
        """Imprime las palabras mas representativas de cada tema."""
        if self.modelo is None:
            raise ValueError("Primero llama a entrenar().")
        for idx, tema in self.modelo.print_topics(num_topics=-1, num_words=num_palabras):
            print(f"Tema {idx}: {tema}")

    def temas_por_documento(self, textos_procesados):
        """Devuelve el id del tema dominante de cada texto."""
        if self.modelo is None:
            raise ValueError("Primero llama a entrenar().")
        temas = []
        for texto in textos_procesados:
            bow = self.diccionario.doc2bow(texto.split())
            distribucion = self.modelo.get_document_topics(bow)
            if distribucion:
                tema_id = max(distribucion, key=lambda x: x[1])[0]
            else:
                tema_id = -1
            temas.append(tema_id)
        return temas

    def guardar(self, ruta_modelo="modelo_lda.model", ruta_diccionario="dictionary_lda.dict"):
        """Guarda el modelo LDA y el diccionario en disco."""
        if self.modelo is None:
            raise ValueError("Primero llama a entrenar().")
        self.modelo.save(ruta_modelo)
        self.diccionario.save(ruta_diccionario)


if __name__ == "__main__":
    textos = [
        "hotel limpio habitacion comodo personal amable",
        "comida excelente restaurante buen servicio",
        "precio caro habitacion sucio mal atencion",
    ]
    modelador = ModeladorTemas(num_temas=2, passes=5)
    modelador.entrenar(textos)
    modelador.mostrar_temas(num_palabras=5)
