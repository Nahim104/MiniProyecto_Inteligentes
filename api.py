"""Capa 2 (MLOps y API)

Recibe el CSV de una persona, lo valida, lo pasa por el modelo entrenado en Colab y
devuelve en JSON el subtipo predicho y la probabilidad de cada subtipo. El front-end
solo tiene que mostrar el resultado.

Ejecución:
    pip install fastapi "uvicorn[standard]" python-multipart numpy pandas torch
    python api.py                      # arranca el servidor en http://127.0.0.1:8000
    (o: uvicorn api:app --port 8000)
    Documentación interactiva con botón para subir un CSV: http://127.0.0.1:8000/docs

Variables de entorno opcionales:
    ARTIFACTS_DIR   fuerza otra carpeta de artefactos en lugar de la búsqueda anterior
    MAX_UPLOAD_MB   tamaño máximo del CSV (por defecto 200)
    HOST, PORT      dirección del servidor al ejecutar `python api.py`
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

# PyTorch solo hace falta para abrir el modelo; el resto funciona sin él.
try:
    import torch
    from torch import nn
    TORCH_ERROR = None
except Exception as exc:
    torch = nn = None
    TORCH_ERROR = exc


NAMES = {"adamw": "AdamW", "sgd": "SGD"}
ARTIFACT_FILES = ("modelo.pth", "scaler.npz", "metadata.json")
HISTORY_KEYS = ("epoch", "train_loss", "val_loss", "val_accuracy")
STATE_KEYS = ("hidden.weight", "hidden.bias", "norm.weight", "norm.bias",
              "norm.running_mean", "norm.running_var", "output.weight", "output.bias")

OK, WARNING, ERROR = "ok", "warning", "error"
LABEL = {OK: "[OK]", WARNING: "[AVISO]", ERROR: "[ERROR]"}


class BackendError(Exception):
    """Error con un mensaje listo para mostrarse a la persona usuaria."""


class InvalidCSV(BackendError):
    """El CSV recibido no cumple los requisitos (en la API equivale a un HTTP 400)."""

    def __init__(self, code, message, **extra):
        super().__init__(message)
        self.code, self.message, self.extra = code, message, extra



if torch is not None:
    class CancerSubtypeNet(nn.Module):
        """Misma arquitectura que el entrenamiento en Colab."""

        def __init__(self, input_size, classes, hidden=64):
            super().__init__()
            self.hidden = nn.Linear(input_size, hidden)
            self.norm = nn.BatchNorm1d(hidden)
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(p=0.5)
            self.output = nn.Linear(hidden, classes)

        def forward(self, X):
            X = self.activation(self.norm(self.hidden(X)))
            return self.output(self.dropout(X))


def _load_state(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # versiones antiguas de PyTorch
        return torch.load(path, map_location="cpu")


# ---------------------------------------------------------------- informe
@dataclass
class Check:
    level: str
    message: str


@dataclass
class Report:
    folder: Path
    optimizer: str | None = None
    history_ok: bool = False
    checks: list[Check] = field(default_factory=list)

    def add(self, level, message):
        self.checks.append(Check(level, message))

    @property
    def errors(self):
        return [c for c in self.checks if c.level == ERROR]

    @property
    def warnings(self):
        return [c for c in self.checks if c.level == WARNING]

    @property
    def success(self):
        return not self.errors

    @property
    def status(self):
        return ERROR if self.errors else WARNING if self.warnings else OK

    @property
    def name(self):
        return NAMES.get(self.optimizer, self.folder.name)

    @property
    def headline(self):
        if self.errors:
            return (f"{self.name}: el entrenamiento NO se puede dar por completado "
                    f"({len(self.errors)} error(es)).")
        if self.warnings:
            return (f"{self.name}: entrenamiento completado, con "
                    f"{len(self.warnings)} advertencia(s).")
        return f"{self.name}: entrenamiento completado correctamente."

    def format(self):
        lines = [f"{self.headline}  ({self.folder})"]
        lines += [f"  {LABEL[c.level]} {c.message}" for c in self.checks]
        return "\n".join(lines)


# ---------------------------------------------------------------- verificación
def verify_artifacts(folder, expected_epochs=None):
    """Revisa una carpeta de artefactos y devuelve un Report con mensajes de éxito o error."""
    folder = Path(folder)
    rep = Report(folder)
    if not folder.is_dir():
        rep.add(ERROR, f"La carpeta {folder} no existe.")
        return rep

    # 1. Archivos presentes y no vacíos
    for name in ARTIFACT_FILES:
        path = folder / name
        if not path.is_file():
            rep.add(ERROR, f"Falta {name}: el entrenamiento no terminó de guardar sus resultados.")
        elif path.stat().st_size == 0:
            rep.add(ERROR, f"{name} está vacío: el guardado se interrumpió.")
    meta_path = folder / "metadata.json"
    if not meta_path.is_file() or meta_path.stat().st_size == 0:
        return rep  # sin metadata no hay con qué contrastar lo demás
    if not rep.errors:
        rep.add(OK, "Están los tres archivos: modelo.pth, scaler.npz y metadata.json.")

    # 2. metadata.json
    try:
        meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        rep.add(ERROR, f"metadata.json no se puede leer ({exc}).")
        return rep
    missing = [k for k in ("optimizer", "features", "classes", "history") if k not in meta]
    if missing:
        rep.add(ERROR, f"metadata.json está incompleto: faltan {', '.join(missing)}.")
        return rep

    optimizer = str(meta["optimizer"]).lower()
    if optimizer in NAMES:
        rep.optimizer = optimizer
    else:
        rep.add(ERROR, f"Optimizador desconocido en metadata.json: {meta['optimizer']!r}.")
    features, classes = list(meta["features"]), list(meta["classes"])
    if not features or not classes:
        rep.add(ERROR, "metadata.json no registra genes (features) o subtipos (classes).")
        return rep

    # 3. Historial de épocas
    _check_history(rep, meta["history"], expected_epochs)

    # 4. Scaler y modelo
    if (folder / "scaler.npz").is_file():
        _check_scaler(rep, folder / "scaler.npz", len(features))
    if (folder / "modelo.pth").is_file():
        _check_model(rep, folder / "modelo.pth", len(features), len(classes))
    return rep


def _check_history(rep, history, expected_epochs):
    if not isinstance(history, list) or not history:
        rep.add(ERROR, "El historial está vacío: no se registró ninguna época.")
        return
    if any(not isinstance(r, dict) or any(k not in r for k in HISTORY_KEYS) for r in history):
        rep.add(ERROR, "El historial tiene épocas sin epoch, train_loss, val_loss o val_accuracy.")
        return
    try:
        values = np.array([[r["train_loss"], r["val_loss"], r["val_accuracy"]]
                           for r in history], dtype=float)
    except (TypeError, ValueError):
        rep.add(ERROR, "El historial contiene valores que no son numéricos.")
        return

    n = len(history)
    ok = True
    if [r["epoch"] for r in history] != list(range(1, n + 1)):
        rep.add(ERROR, "Las épocas del historial no son consecutivas desde 1: faltan épocas.")
        ok = False
    bad = np.flatnonzero(~np.isfinite(values).all(axis=1))
    if bad.size:
        rep.add(ERROR, f"La pérdida se volvió NaN o infinita en la época {bad[0] + 1}: "
                       "el entrenamiento divergió. Prueba una tasa de aprendizaje menor.")
        ok = False
    elif ((values[:, 2] < 0) | (values[:, 2] > 1)).any():
        rep.add(ERROR, "Hay accuracies fuera del rango 0 a 1 en el historial.")
        ok = False
    if expected_epochs is not None and n != expected_epochs:
        rep.add(ERROR, f"Entrenamiento incompleto: se registraron {n} de {expected_epochs} épocas.")
        ok = False
    rep.history_ok = ok
    if not ok:
        return

    train, val, acc = values[:, 0], values[:, 1], values[:, 2]
    best = int(acc.argmax())
    rep.add(OK, f"Historial completo: {n} épocas, accuracy de validación final {acc[-1]:.3f} "
                f"(mejor {acc[best]:.3f} en la época {best + 1}).")
    if train[-1] >= train[0]:
        rep.add(WARNING, "La pérdida de entrenamiento no bajó entre la primera y la última época: "
                         "el modelo casi no aprendió. Revisa la tasa de aprendizaje.")
    if val[-1] > 1.5 * val.min():
        rep.add(WARNING, f"La pérdida de validación terminó 50 % por encima de su mejor valor "
                         f"(época {int(val.argmin()) + 1}): posible sobreajuste.")
    if acc[-1] < 0.5:
        rep.add(WARNING, f"El accuracy de validación final es bajo ({acc[-1]:.3f}).")


def _check_scaler(rep, path, n_features):
    try:
        data = np.load(path)
        mean, scale = data["mean"], data["scale"]
    except Exception as exc:
        rep.add(ERROR, f"scaler.npz no se puede leer o no tiene mean y scale ({exc.__class__.__name__}).")
        return
    if mean.shape != (n_features,) or scale.shape != (n_features,):
        rep.add(ERROR, f"scaler.npz no coincide con metadata.json: tiene {mean.size} genes y se "
                       f"esperaban {n_features}.")
    elif not (np.isfinite(mean).all() and np.isfinite(scale).all()):
        rep.add(ERROR, "scaler.npz contiene NaN o infinitos.")
    elif (scale <= 0).any():
        rep.add(ERROR, "scaler.npz tiene desviaciones estándar iguales o menores que cero.")
    else:
        rep.add(OK, f"scaler.npz es coherente con los {n_features} genes del modelo.")


def _check_model(rep, path, n_features, n_classes):
    if torch is None:
        rep.add(WARNING, "No se pudo abrir modelo.pth porque PyTorch no está instalado; "
                         "solo se comprobó que el archivo existe.")
        return
    try:
        state = _load_state(path)
        missing = [k for k in STATE_KEYS if k not in state]
        if missing:
            rep.add(ERROR, f"modelo.pth no tiene los pesos esperados (faltan {missing[:3]}).")
            return
        hidden, n_in = state["hidden.weight"].shape
        n_out = state["output.weight"].shape[0]
    except Exception as exc:
        rep.add(ERROR, f"modelo.pth no se pudo leer ({exc.__class__.__name__}): "
                       "el archivo está dañado o no es un state_dict de la red.")
        return

    if n_in != n_features:
        rep.add(ERROR, f"modelo.pth espera {n_in} genes y metadata.json registra {n_features}.")
    if n_out != n_classes:
        rep.add(ERROR, f"modelo.pth tiene {n_out} salidas y metadata.json registra {n_classes} subtipos.")
    if n_in != n_features or n_out != n_classes:
        return
    bad = [k for k in STATE_KEYS if not torch.isfinite(state[k]).all()]
    if bad:
        rep.add(ERROR, f"Los pesos de modelo.pth contienen NaN o infinitos ({', '.join(bad[:3])}): "
                       "el entrenamiento divergió.")
        return
    seen = state.get("norm.num_batches_tracked")
    if seen is not None and int(seen) == 0:
        rep.add(ERROR, "BatchNorm nunca vio un lote: el modelo se guardó sin entrenar.")
        return
    try:
        model = CancerSubtypeNet(n_in, n_out, hidden)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(2, n_in))
        if tuple(out.shape) != (2, n_out) or not torch.isfinite(out).all():
            raise ValueError("la salida no tiene la forma o los valores esperados")
        rep.add(OK, f"modelo.pth carga correctamente y responde con {n_out} salidas finitas.")
    except Exception as exc:
        rep.add(ERROR, f"modelo.pth no es compatible con la arquitectura de la red: {exc}")


class Artifacts:
    """Lo que Colab guardó para un optimizador, listo para graficar y predecir."""

    def __init__(self, folder):
        self.folder = Path(folder)
        try:
            meta = json.loads((self.folder / "metadata.json").read_text(encoding="utf-8"))
            self.optimizer = str(meta["optimizer"]).lower()
            self.history = meta["history"]
            self.classes = list(meta["classes"])
            self.features = list(meta["features"])
        except (OSError, ValueError, KeyError) as exc:
            raise BackendError(f"No se pudo leer metadata.json en {self.folder}: {exc}") from exc
        self.settings = meta.get("optimizer_settings", {})
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        if torch is None:
            raise BackendError(f"PyTorch no está disponible: {TORCH_ERROR}")
        for name in ("modelo.pth", "scaler.npz"):
            if not (self.folder / name).is_file():
                raise BackendError(f"Falta {name} en {self.folder}.")
        try:
            scaler = np.load(self.folder / "scaler.npz")
            self._mean, self._scale = scaler["mean"], scaler["scale"]
            state = _load_state(self.folder / "modelo.pth")
            hidden, n_inputs = state["hidden.weight"].shape
            model = CancerSubtypeNet(n_inputs, state["output.weight"].shape[0], hidden)
            model.load_state_dict(state)
        except Exception as exc:
            raise BackendError(f"No se pudo cargar el modelo de {NAMES.get(self.optimizer)}: {exc}") from exc
        model.eval()  # Dropout apagado y BatchNorm con sus estadísticas guardadas
        self._model = model

    def predict(self, X):
        """X: matriz (muestras, genes). Devuelve probabilidades por subtipo."""
        self._load_model()
        Z = ((X - self._mean) / self._scale).astype(np.float32)
        with torch.no_grad():
            return torch.softmax(self._model(torch.from_numpy(Z)), dim=1).numpy()


def scan(root, expected_epochs=None):
    """Verifica los artefactos de la carpeta elegida y de sus subcarpetas directas.

    Devuelve (artefactos_usables, informes). Un optimizador es usable para graficar
    si su metadata.json y su historial son válidos, aunque otros archivos fallen.
    """
    root = Path(root)
    if not root.is_dir():
        raise BackendError(f"La carpeta {root} no existe.")
    folders = [root] + sorted(p for p in root.iterdir() if p.is_dir())
    folders = [f for f in folders if any((f / n).exists() for n in ARTIFACT_FILES)]
    found, reports = {}, []
    for folder in folders:
        report = verify_artifacts(folder, expected_epochs)
        reports.append(report)
        if report.optimizer and report.history_ok:
            try:
                found[report.optimizer] = Artifacts(folder)
            except BackendError as exc:
                report.add(ERROR, str(exc))
    return found, reports


# ---------------------------------------------------------------- clasificación
def validation_mask(labels, classes):
    """Reproduce la partición 80/20 del entrenamiento (semilla 42, por subtipo).

    Solo es válida si el CSV es el mismo que se usó en Colab.
    Devuelve un arreglo booleano (True = muestra de validación) o None.
    """
    y = np.array([classes.index(label) for label in labels])
    rng = np.random.default_rng(42)
    is_val = np.zeros(len(y), dtype=bool)
    for label in range(len(classes)):
        indices = rng.permutation(np.flatnonzero(y == label))
        if len(indices) < 2:
            return None
        count = min(len(indices) - 1, max(1, round(len(indices) * 0.2)))
        is_val[indices[:count]] = True
    return is_val


# Una firma genómica completa tiene unas 54 mil columnas (sondas del chip). Un CSV con
# muchas menos no puede ser una entrada válida, así que se rechaza antes de leerlo entero.
MIN_COLUMNS = 54_000


def validate_csv(source, features, min_columns=MIN_COLUMNS):
    """Valida y limpia el CSV de un usuario antes de pasarlo al modelo.

    source: bytes con el contenido del archivo, o una ruta.
    Cada regla que falla levanta InvalidCSV con un código y un mensaje claros:
      1. archivo_vacio / csv_ilegible: el archivo tiene contenido y es un CSV.
      2. columnas_insuficientes: tiene al menos min_columns columnas.
      3. genes_faltantes: contiene, por nombre, todos los genes que el modelo espera.
      4. sin_filas: se descartan las filas totalmente vacías; debe quedar alguna.
      5. columnas_no_numericas / valores_nulos / valores_infinitos: las mediciones son
         números finitos y no falta ninguno.
    Devuelve (tabla_limpia, info) con info = rows_received, columns, empty_rows_removed.
    """
    features = list(features)

    def opener():
        return io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source

    if isinstance(source, (bytes, bytearray)) and not source.strip():
        raise InvalidCSV("archivo_vacio", "El archivo está vacío.")
    try:
        # Solo la primera línea: con 54 mil columnas, pd.read_csv(nrows=0) tarda varios segundos.
        if isinstance(source, (bytes, bytearray)):
            with io.TextIOWrapper(io.BytesIO(source), encoding="utf-8-sig", newline="") as handle:
                columns = next(csv.reader(handle), [])
        else:
            with open(source, encoding="utf-8-sig", newline="") as handle:
                columns = next(csv.reader(handle), [])
    except Exception as exc:
        raise InvalidCSV("csv_ilegible", f"No se pudo leer el archivo como CSV: {exc}") from exc
    if not columns:
        raise InvalidCSV("archivo_vacio", "El archivo está vacío.")

    n_columns = len(columns)
    if n_columns < min_columns:
        raise InvalidCSV(
            "columnas_insuficientes",
            f"El archivo debe tener por lo menos {min_columns} columnas y el recibido "
            f"tiene {n_columns}.", columns_received=n_columns, columns_required=min_columns)

    present = set(columns)
    missing = [g for g in features if g not in present]
    if missing:
        raise InvalidCSV(
            "genes_faltantes",
            f"Al CSV le faltan {len(missing)} de los {len(features)} genes que el modelo "
            f"espera (por ejemplo: {missing[:5]}).",
            missing_count=len(missing), missing_examples=missing[:5])

    extra = [c for c in ("samples", "type") if c in present]
    try:
        table = pd.read_csv(opener(), usecols=features + extra, low_memory=False)
    except Exception as exc:
        raise InvalidCSV("csv_ilegible", f"No se pudo leer el archivo como CSV: {exc}") from exc

    n_received = len(table)
    values = table[features]  # una sola selección: con 54 mil columnas cada una cuesta segundos
    empty = values.isna().all(axis=1)
    removed = int(empty.sum())
    if removed:
        table = table.loc[~empty].reset_index(drop=True)
        values = values.loc[~empty]
    if table.empty:
        raise InvalidCSV("sin_filas", "El archivo no contiene ninguna fila con mediciones.",
                         rows_received=n_received)

    dtypes = values.dtypes
    non_numeric = [c for c, d in dtypes.items() if not pd.api.types.is_numeric_dtype(d)]
    if non_numeric:
        raise InvalidCSV(
            "columnas_no_numericas",
            f"{len(non_numeric)} columnas de expresión tienen valores que no son números "
            f"(por ejemplo: {non_numeric[:5]}).",
            columns_count=len(non_numeric), columns_examples=non_numeric[:5])

    X = values.to_numpy(dtype=np.float64)
    ids = (table["samples"].astype(str).to_numpy() if "samples" in table.columns
           else np.arange(1, len(table) + 1).astype(str))
    nulls = np.isnan(X)
    if nulls.any():
        rows = np.flatnonzero(nulls.any(axis=1))
        raise InvalidCSV(
            "valores_nulos",
            f"El archivo tiene {int(nulls.sum())} valores nulos en {len(rows)} fila(s) "
            f"(muestras: {ids[rows][:5].tolist()}). No se aceptan valores nulos.",
            null_values=int(nulls.sum()), rows_affected=int(len(rows)),
            example_samples=ids[rows][:5].tolist())
    infinite = np.isinf(X)
    if infinite.any():
        rows = np.flatnonzero(infinite.any(axis=1))
        raise InvalidCSV(
            "valores_infinitos",
            f"El archivo tiene {int(infinite.sum())} valores infinitos en {len(rows)} fila(s) "
            f"(muestras: {ids[rows][:5].tolist()}).",
            infinite_values=int(infinite.sum()), rows_affected=int(len(rows)),
            example_samples=ids[rows][:5].tolist())

    return table, dict(rows_received=n_received, columns=n_columns, empty_rows_removed=removed)


@dataclass
class Classification:
    optimizer: str
    sample_ids: list
    predicted: list
    confidence: list
    real: list | None
    subset: list              # "validación", "entrenamiento" o "" si se desconoce
    correct: list             # True, False o None si no hay etiqueta conocida
    n_scored: int
    accuracy_all: float | None
    n_val: int
    accuracy_val: float | None
    probabilities: list       # por muestra, una probabilidad por subtipo (mismo orden que classes)
    classes: list

    @property
    def summary(self):
        text = f"{NAMES[self.optimizer]}: {len(self.predicted)} muestras clasificadas."
        if self.real is None:
            return text + " El CSV no tiene columna type, así que no se calcula accuracy."
        if self.n_scored:
            text += f" Accuracy sobre todo el CSV: {self.accuracy_all:.3f}."
        if self.n_val:
            text += (f" Accuracy solo en las {self.n_val} muestras de validación: "
                     f"{self.accuracy_val:.3f}. La partición asume que este CSV es el mismo del "
                     "entrenamiento; las muestras de entrenamiento inflan el accuracy global.")
        return text


def classify_table(art, table):
    """Clasifica las filas de un DataFrame con el modelo de `art`."""
    missing = [c for c in art.features if c not in table.columns]
    if missing:
        raise BackendError(f"Al CSV le faltan {len(missing)} columnas de expresión que el modelo "
                           f"espera (por ejemplo: {missing[:3]}).")
    try:
        X = table[art.features].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise BackendError(f"Las columnas de expresión deben ser numéricas: {exc}") from exc
    if not np.isfinite(X).all():
        raise BackendError("El CSV contiene mediciones no finitas (NaN o infinitos).")

    probs = art.predict(X)
    predicted = [art.classes[i] for i in probs.argmax(axis=1)]
    confidence = probs.max(axis=1).tolist()
    n = len(table)
    ids = (table["samples"].astype(str).tolist() if "samples" in table.columns
           else [str(i + 1) for i in range(n)])
    real = table["type"].astype(str).tolist() if "type" in table.columns else None

    subset, correct = [""] * n, [None] * n
    n_scored = hits = n_val = val_hits = 0
    if real is not None:
        positions = [i for i, r in enumerate(real) if r in art.classes]
        mask = validation_mask([real[i] for i in positions], art.classes) if positions else None
        for k, i in enumerate(positions):
            hit = predicted[i] == real[i]
            correct[i] = hit
            n_scored += 1
            hits += hit
            if mask is not None:
                subset[i] = "validación" if mask[k] else "entrenamiento"
                if mask[k]:
                    n_val += 1
                    val_hits += hit
    return Classification(
        optimizer=art.optimizer, sample_ids=ids, predicted=predicted, confidence=confidence,
        real=real, subset=subset, correct=correct, n_scored=n_scored,
        accuracy_all=hits / n_scored if n_scored else None,
        n_val=n_val, accuracy_val=val_hits / n_val if n_val else None,
        probabilities=probs.astype(np.float64).round(6).tolist(), classes=list(art.classes))


# ================================================================ API (endpoints)
BASE_DIR = Path(__file__).resolve().parent


def find_default_artifacts_dir():
    """Carpeta con los artefactos: la de api.py o, si no hay nada ahí, ./artifacts.

    Devuelve None si no encuentra metadata.json en ninguna de las dos (ni directamente
    ni dentro de adamw/ o sgd/).
    """
    for candidate in (BASE_DIR, BASE_DIR / "artifacts"):
        places = [candidate] + [candidate / name for name in NAMES]
        if any((place / "metadata.json").is_file() for place in places):
            return candidate
    return None


ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR") or find_default_artifacts_dir() or BASE_DIR)
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "200")) * 1024 * 1024

# Modelos cargados al arrancar: {"adamw": Artifacts}, e informes de verificación.
state = {"artifacts": {}, "reports": {}, "all_reports": []}


def load_models():
    """Lee y verifica los artefactos del entrenamiento (se llama al arrancar)."""
    try:
        found, reports = scan(ARTIFACTS_DIR)
    except BackendError:
        found, reports = {}, []
    state["artifacts"] = found
    state["reports"] = {r.optimizer: r for r in reports if r.optimizer}
    state["all_reports"] = reports


@asynccontextmanager
async def lifespan(app):
    load_models()
    yield


app = FastAPI(title="Diagnóstico de subtipos de cáncer de mama", version="1.1.0",
              lifespan=lifespan)


def fail(status, code, message, **extra):
    raise HTTPException(status_code=status, detail={"error": code, "message": message, **extra})


def usable_model(optimizer):
    """Devuelve el modelo solo si está cargado y su entrenamiento pasó la verificación."""
    art, report = state["artifacts"].get(optimizer), state["reports"].get(optimizer)
    if art is None or report is None:
        fail(503, "modelo_no_cargado",
             f"El modelo de {NAMES[optimizer]} no está cargado en el servidor "
             f"(se buscó en {ARTIFACTS_DIR}).")
    if not report.success:
        fail(503, "modelo_no_verificado",
             f"El modelo de {NAMES[optimizer]} no pasó la verificación del entrenamiento.",
             problems=[c.message for c in report.errors])
    return art


def check_upload(optimizer, file):
    """Pasos comunes de /validate y /predict: optimizador, archivo y validación del CSV.

    Devuelve (optimizer, modelo, tabla_limpia, info). Si algo falla, responde con el
    error HTTP correspondiente (400, 413 o 503).
    """
    optimizer = optimizer.lower()
    if optimizer not in NAMES:
        fail(400, "optimizador_invalido", f"El optimizador debe ser uno de: {', '.join(NAMES)}.")
    art = usable_model(optimizer)

    if not (file.filename or "").lower().endswith(".csv"):
        fail(400, "archivo_no_csv", "El archivo debe tener extensión .csv.")
    raw = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        fail(413, "archivo_muy_grande",
             f"El archivo supera el máximo de {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")

    try:
        table, info = validate_csv(raw, art.features)
    except InvalidCSV as exc:
        fail(400, exc.code, exc.message, **exc.extra)
    return optimizer, art, table, info


def received_summary(info):
    return {"rows": info["rows_received"], "columns": info["columns"],
            "empty_rows_removed": info["empty_rows_removed"]}


@app.get("/health")
def health():
    ready = [o for o, r in state["reports"].items()
             if r.success and o in state["artifacts"]]
    if not ready:
        status = "unavailable"
    elif any(not r.success for r in state["all_reports"]):
        status = "degraded"
    else:
        status = "ok"
    body = {
        "status": status,
        "ready_models": ready,
        "min_columns": MIN_COLUMNS,
        "training": [
            {"optimizer": r.optimizer, "status": r.status, "headline": r.headline,
             "messages": [{"level": c.level, "message": c.message} for c in r.checks]}
            for r in state["all_reports"]],
    }
    return JSONResponse(body, status_code=503 if status == "unavailable" else 200)


@app.get("/models/{optimizer}/history")
def history(optimizer: str):
    optimizer = optimizer.lower()
    art = state["artifacts"].get(optimizer)
    if optimizer not in NAMES or art is None:
        fail(404, "modelo_no_encontrado", f"No hay historial cargado para {optimizer!r}.")
    return {"optimizer": optimizer, "classes": art.classes, "n_features": len(art.features),
            "settings": art.settings, "history": art.history}


@app.post("/validate")
def validate(file: UploadFile = File(...),
             optimizer: str = Query("adamw", description="adamw o sgd")):
    """Valida el CSV contra el modelo elegido sin clasificarlo."""
    optimizer, art, table, info = check_upload(optimizer, file)
    samples = (table["samples"].astype(str).tolist() if "samples" in table.columns
               else [str(i + 1) for i in range(len(table))])
    return {
        "valid": True,
        "optimizer": optimizer,
        "received": received_summary(info),
        "samples": samples,
        "has_labels": "type" in table.columns,
        "message": f"El CSV es válido: {len(table)} muestras listas para clasificar.",
    }


@app.post("/predict")
def predict(file: UploadFile = File(...),
            optimizer: str = Query("adamw", description="adamw o sgd")):
    """Valida el CSV y devuelve, por muestra, el subtipo predicho y su probabilidad."""
    optimizer, art, table, info = check_upload(optimizer, file)
    try:
        result = classify_table(art, table)
    except BackendError as exc:
        fail(503, "modelo_no_disponible", str(exc))

    predictions = [
        {"sample": result.sample_ids[i],
         "predicted": result.predicted[i],
         "confidence": round(result.confidence[i], 6),
         "probabilities": dict(zip(result.classes, result.probabilities[i])),
         "real": result.real[i] if result.real is not None else None}
        for i in range(len(result.predicted))]

    evaluation = None
    if result.real is not None and result.n_scored:
        evaluation = {"samples_with_label": result.n_scored, "accuracy": result.accuracy_all}

    return {
        "optimizer": optimizer,
        "classes": result.classes,
        "received": received_summary(info),
        "predictions": predictions,
        "evaluation": evaluation,
        "message": f"Se clasificaron {len(predictions)} muestras con {NAMES[optimizer]}.",
    }


def verify_cli(argv=None):
    parser = argparse.ArgumentParser(prog="python api.py verificar",
                                     description="Verifica los artefactos del entrenamiento.")
    parser.add_argument("path", nargs="?", default=None,
                        help="carpeta con los artefactos (por defecto, la de api.py)")
    parser.add_argument("--epochs", type=int, help="épocas que debieron completarse")
    parser.add_argument("--csv", help="CSV a clasificar con cada modelo verificado")
    args = parser.parse_args(argv)

    try:
        found, reports = scan(args.path or ARTIFACTS_DIR, args.epochs)
    except BackendError as exc:
        print(f"[ERROR] {exc}")
        return 1
    if not reports:
        print("[ERROR] No encontré modelo.pth, scaler.npz ni metadata.json en esa carpeta.")
        return 1
    for report in reports:
        print(report.format(), end="\n\n")

    if args.csv:
        try:
            table = pd.read_csv(args.csv)
        except Exception as exc:
            print(f"[ERROR] No se pudo leer el CSV: {exc}")
            return 1
        for report in reports:
            art = found.get(report.optimizer)
            if art is None or not report.success:
                continue
            try:
                print(f"[OK] {classify_table(art, table).summary}")
            except BackendError as exc:
                print(f"[ERROR] {report.name}: {exc}")
    return 0 if all(r.success for r in reports) else 1


def main(argv=None):
    """python api.py = servidor.  python api.py verificar verificación sin servidor."""
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "verificar":
        return verify_cli(argv[1:])
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "8000")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
