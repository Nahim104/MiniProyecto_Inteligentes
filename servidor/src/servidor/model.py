from pathlib import Path
import json

import numpy as np
import torch
from torch import nn


MODEL = "modelo.pth"

#Buscamos en artifacts el archivo modelo.pth y si no esta, lanzamos un error
def validar_modelo(artifacts_dir: Path) -> Path:
    if not artifacts_dir.is_dir():
        raise FileNotFoundError(
            f"No existe la carpeta de artefactos: {artifacts_dir}"
        )

    # Nombres reales en disco, respetando mayúsculas/minúsculas
    nombres = {p.name for p in artifacts_dir.iterdir() if p.is_file()}

    #Busca el nombre de otros archivos .pth para comprobar si hay un error de nombre
    if MODEL not in nombres:
        otros_pth = sorted(n for n in nombres if n.lower().endswith(".pth"))

        if otros_pth:
            raise FileNotFoundError(
                f"No se encontró '{MODEL}' en {artifacts_dir}. "
                f"Hay archivos .pth con otro nombre: {otros_pth}. "
                f"Renómbralo exactamente a '{MODEL}'."
            )

        #No existe para nada un archivo .pth 
        else:
         raise FileNotFoundError(
            f"No se encontró '{MODEL}' en {artifacts_dir}."
        )

    return artifacts_dir / MODEL


class CancerSubtypeNet(nn.Module):
    """Arquitectura utilizada durante el entrenamiento en Colab."""

    def __init__(self, input_size, classes):
        super().__init__()

        self.hidden = nn.Linear(input_size, 64)
        self.norm = nn.BatchNorm1d(64)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.output = nn.Linear(64, classes)

    def forward(self, X):
        X = self.hidden(X)
        X = self.norm(X)
        X = self.activation(X)
        return self.output(self.dropout(X))


class CancerClassifier:
    """Carga los artefactos y realiza predicciones."""

    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir

        model_path = validar_modelo(artifacts_dir)

        # 1. Cargar metadata
        with open(artifacts_dir / "metadata.json", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.features = self.metadata["features"]
        self.classes = self.metadata["classes"]

        # 2. Cargar scaler
        scaler = np.load(artifacts_dir / "scaler.npz")
        self.mean = scaler["mean"]
        self.scale = scaler["scale"]

        # 3. Cargar pesos del modelo
        state_dict = torch.load(
            artifacts_dir / "modelo.pth",
            map_location="cpu",
            weights_only=True
        )

        self.model = CancerSubtypeNet(
            input_size=len(self.features),
            classes=len(self.classes)
        )

        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Comprobaciones básicas de compatibilidad al arrancar
        if self.mean.shape != (len(self.features),):
            raise ValueError("El scaler no coincide con las features de metadata.json.")

        if self.scale.shape != (len(self.features),):
            raise ValueError("El scaler no coincide con las features de metadata.json.")

        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all():
            raise ValueError("El scaler contiene valores no finitos.")

        if (self.scale <= 0).any():
            raise ValueError("El scaler contiene escalas inválidas.")

       
    def predict(self, X: np.ndarray):
        """Recibe una matriz de expresiones genéticas y devuelve probabilidades."""

        # Aplicar exactamente el mismo escalado del entrenamiento
        X = ((X - self.mean) / self.scale).astype(np.float32)

        with torch.no_grad():
            logits = self.model(torch.from_numpy(X))
            probabilities = torch.softmax(logits, dim=1).numpy()

        return probabilities
