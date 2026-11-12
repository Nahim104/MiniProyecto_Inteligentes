from contextlib import asynccontextmanager
from pathlib import Path
import csv
import io

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from servidor.model import CancerClassifier


# artifacts/ está en la raíz del proyecto
BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = BASE_DIR / "artifacts"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Se ejecuta UNA SOLA VEZ al iniciar la aplicación
    app.state.classifier = CancerClassifier(ARTIFACTS_DIR)

    print("Modelo cargado correctamente.")

    yield

    # Aquí se podrían liberar recursos al apagar el servidor.


app = FastAPI(
    title="Clasificación de subtipos de cáncer de mama",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # TODO: en prod, poner el origen del FE
    allow_methods=["POST"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(file: UploadFile = File(...)):

    # 1. Comprobar extensión
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="El archivo debe tener extensión .csv."
        )

    # 2. Obtener el modelo ya cargado
    classifier = app.state.classifier

    # 3. Leer el archivo (síncrono, porque la función ya no es async)
    content = file.file.read()

    if not content.strip():
        raise HTTPException(
            status_code=400,
            detail="El archivo está vacío."
        )

    try:
        # Leer encabezados originales
        headers = next(csv.reader(
            io.StringIO(content.decode("utf-8-sig"))
        ))

        if not headers:
            raise ValueError("El CSV no contiene encabezados.")

        present = set(headers)

        # Comprobar que estén todos los genes necesarios
        missing = [
            feature for feature in classifier.features
            if feature not in present
        ]

        if missing:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "missing_features",
                    "message": "El CSV no contiene todas las características requeridas.",
                    "missing_count": len(missing),
                    "examples": missing[:5]
                }
            )

        # Leer solamente las columnas que necesitamos
        optional = [
            column for column in ("samples", "type")
            if column in present
        ]

        table = pd.read_csv(
            io.BytesIO(content),
            usecols=classifier.features + optional,
            low_memory=False
        )

    except HTTPException:
        raise
    except (UnicodeDecodeError, csv.Error, pd.errors.ParserError,
            pd.errors.EmptyDataError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"No se pudo leer el CSV: {exc}"
        )

    # 4. Validar que existan muestras
    if table.empty:
        raise HTTPException(
            status_code=400,
            detail="El CSV no contiene muestras."
        )

    # 5. Validar expresiones genéticas
    try:
        X = table[classifier.features].to_numpy(dtype=np.float64)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=400,
            detail="Las expresiones genéticas deben contener solamente números."
        )

    if not np.isfinite(X).all():
        raise HTTPException(
            status_code=400,
            detail="El CSV contiene valores faltantes o no finitos."
        )

    # 6. Clasificar
    probabilities = classifier.predict(X)

    # 7. Construir respuesta
    predictions = []

    for i, probs in enumerate(probabilities):

        predicted_index = int(np.argmax(probs))

        sample_id = (
            str(table["samples"].iloc[i])
            if "samples" in table.columns
            else str(i + 1)
        )

        real = (
            str(table["type"].iloc[i])
            if "type" in table.columns
            else None
        )

        predictions.append({
            "sample": sample_id,
            "predicted": classifier.classes[predicted_index],
            "confidence": round(float(probs[predicted_index]), 6),
            "probabilities": {
                label: round(float(prob), 6)
                for label, prob in zip(classifier.classes, probs)
            },
            "real": real
        })

    return {
        "predictions": predictions
    }