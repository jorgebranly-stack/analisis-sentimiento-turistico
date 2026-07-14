"""
clasificador_beto.py
--------------------
Fine-tuning de BETO (BERT en español) para clasificacion de sentimiento
de tres clases: negativo, neutral y positivo.

Requiere GPU para tiempos de entrenamiento razonables.

Uso tipico:

    from clasificador_beto import ClasificadorBeto

    clf = ClasificadorBeto()
    clf.preparar_datos(df["texto_transformer"], df["sentimiento"])
    clf.entrenar(epochs=2)
    clf.evaluar()
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)


MODELO_BASE = "dccuchile/bert-base-spanish-wwm-uncased"
ID_A_ETIQUETA = {0: "negativo", 1: "neutral", 2: "positivo"}


class DatasetSentimiento(Dataset):
    """Dataset de PyTorch con los textos y etiquetas ya tokenizados."""

    def __init__(self, textos, etiquetas, tokenizer, max_length=256):
        # Fuerza todo a string y reemplaza vacios, para que el tokenizer
        # nunca reciba valores nulos o tipos inesperados
        textos_limpios = [
            str(t) if pd.notna(t) and str(t).strip() != "" else "vacio"
            for t in textos
        ]
        self.encodings = tokenizer(
            textos_limpios,
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        self.labels = list(etiquetas)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def calcular_metricas(eval_pred):
    """Metricas que usa el Trainer en cada evaluacion."""
    logits, labels = eval_pred
    predicciones = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predicciones)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicciones, average="weighted", zero_division=0
    )
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


class ClasificadorBeto:
    """
    Envoltorio de alto nivel para afinar BETO sobre un dataset de
    sentimiento de tres clases.

    Parametros
    ----------
    modelo_base : str
        Nombre del modelo pre-entrenado en Hugging Face.
    max_length : int
        Longitud maxima de tokens por reseña.
    test_size : float
        Proporcion del conjunto reservada para validacion.
    """

    def __init__(self, modelo_base=MODELO_BASE, max_length=256, test_size=0.2, random_state=42):
        self.max_length = max_length
        self.test_size = test_size
        self.random_state = random_state

        self.tokenizer = AutoTokenizer.from_pretrained(modelo_base)
        self.modelo = AutoModelForSequenceClassification.from_pretrained(
            modelo_base, num_labels=3
        )
        self.trainer = None

    def preparar_datos(self, textos, etiquetas):
        """Divide los datos y construye los datasets de entrenamiento y validacion."""
        train_textos, val_textos, train_labels, val_labels = train_test_split(
            list(textos), list(etiquetas),
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=list(etiquetas),
        )
        self.train_dataset = DatasetSentimiento(train_textos, train_labels, self.tokenizer, self.max_length)
        self.val_dataset = DatasetSentimiento(val_textos, val_labels, self.tokenizer, self.max_length)
        self.val_labels = val_labels

    def entrenar(self, epochs=2, batch_size=16):
        """Afina el modelo BETO sobre los datos preparados."""
        args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir="./logs",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_steps=100,
        )
        self.trainer = Trainer(
            model=self.modelo,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=calcular_metricas,
        )
        self.trainer.train()
        return self.trainer

    def evaluar(self):
        """Imprime el reporte de clasificacion sobre el conjunto de validacion."""
        preds_output = self.trainer.predict(self.val_dataset)
        y_pred = preds_output.predictions.argmax(-1)
        print("Reporte de clasificacion - BETO:")
        print(classification_report(
            self.val_labels, y_pred,
            target_names=list(ID_A_ETIQUETA.values()),
            zero_division=0,
        ))
        return y_pred

    def predecir(self, texto):
        """Predice el sentimiento de un solo texto crudo."""
        device = next(self.modelo.parameters()).device
        inputs = self.tokenizer(
            texto, return_tensors="pt", truncation=True, max_length=self.max_length
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        self.modelo.eval()
        with torch.no_grad():
            logits = self.modelo(**inputs).logits
        idx = int(torch.argmax(logits, dim=-1))
        return ID_A_ETIQUETA[idx]

    def guardar(self, ruta="modelo_beto"):
        """Guarda el modelo afinado y su tokenizer."""
        if self.trainer is None:
            raise ValueError("Primero llama a entrenar().")
        self.trainer.save_model(ruta)
        self.tokenizer.save_pretrained(ruta)


if __name__ == "__main__":
    print("Modulo de clasificacion con BETO. Importar y usar la clase ClasificadorBeto.")
