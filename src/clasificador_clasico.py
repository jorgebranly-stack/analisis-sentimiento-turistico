"""
clasificador_clasico.py
------------------------
Clasificador clasico de sentimiento: TF-IDF + Naive Bayes.
Sirve como linea base para comparar contra el modelo neuronal (BETO).

Uso tipico:

    from clasificador_clasico import ClasificadorClasico

    clf = ClasificadorClasico()
    clf.entrenar(df["texto_procesado"], df["sentimiento"])
    clf.evaluar()
    clf.matriz_confusion()
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


class ClasificadorClasico:
    """
    Envoltorio sobre TfidfVectorizer y MultinomialNB.

    Parametros
    ----------
    max_features : int
        Numero maximo de terminos que conserva el vectorizador TF-IDF.
    ngram_range : tuple
        Rango de n-gramas. (1, 2) usa unigramas y bigramas.
    test_size : float
        Proporcion del conjunto reservada para prueba.
    """

    def __init__(self, max_features=5000, ngram_range=(1, 2), test_size=0.2, random_state=42):
        self.vectorizador = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        self.modelo = MultinomialNB()
        self.test_size = test_size
        self.random_state = random_state
        self.clases = None

    def entrenar(self, textos, etiquetas):
        """Divide, vectoriza con TF-IDF y entrena el clasificador Naive Bayes."""
        X_train, X_test, y_train, y_test = train_test_split(
            textos, etiquetas,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=etiquetas,
        )

        X_train_tfidf = self.vectorizador.fit_transform(X_train)
        X_test_tfidf = self.vectorizador.transform(X_test)

        self.modelo.fit(X_train_tfidf, y_train)

        self.y_test = y_test
        self.y_pred = self.modelo.predict(X_test_tfidf)
        self.clases = sorted(set(etiquetas))
        return self.modelo

    def evaluar(self):
        """Imprime accuracy y el reporte de clasificacion sobre el test."""
        acc = accuracy_score(self.y_test, self.y_pred)
        print(f"Exactitud (accuracy): {acc:.4f}\n")
        print("Reporte de clasificacion:")
        print(classification_report(self.y_test, self.y_pred, zero_division=0))
        return acc

    def matriz_confusion(self, guardar_como=None):
        """Dibuja la matriz de confusion del modelo sobre el test."""
        cm = confusion_matrix(self.y_test, self.y_pred, labels=self.clases)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=self.clases, yticklabels=self.clases)
        plt.xlabel("Prediccion")
        plt.ylabel("Real")
        plt.title("Matriz de confusion - Naive Bayes")
        plt.tight_layout()
        if guardar_como:
            plt.savefig(guardar_como)
        plt.show()

    def predecir(self, texto_procesado):
        """Predice el sentimiento de un solo texto ya preprocesado."""
        vector = self.vectorizador.transform([texto_procesado])
        return self.modelo.predict(vector)[0]

    def guardar(self, ruta_modelo="modelo_nb.pkl", ruta_vectorizador="vectorizer.pkl"):
        """Guarda el modelo y el vectorizador en disco con joblib."""
        import joblib
        joblib.dump(self.modelo, ruta_modelo)
        joblib.dump(self.vectorizador, ruta_vectorizador)


if __name__ == "__main__":
    textos = [
        "hotel limpio personal amable excelente",
        "comida rica buen servicio",
        "sucio caro mal atencion pesimo",
        "habitacion horrible no volveria",
    ]
    etiquetas = [2, 2, 0, 0]  # 0 negativo, 1 neutral, 2 positivo
    clf = ClasificadorClasico(test_size=0.5)
    clf.entrenar(textos, etiquetas)
    clf.evaluar()
