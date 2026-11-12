# Backend

API en FastAPI que recibe un CSV de expresión genética y devuelve el subtipo predicho y las probabilidades de cada clase.

## Requisitos

- uv [(opciones de instalación)](https://docs.astral.sh/uv/getting-started/installation/).
- Python 3.13. El archivo `.python-version` lo fija y uv lo descarga automáticamente si no lo encuentra.

## Ejecutar

```bash
cd servidor
uv sync
uv run uvicorn servidor.main:app --app-dir src
```

Para desarrollo, añadir `--reload` al último comando (reinicia el servidor al cambiar el código y vuelve a cargar el modelo).

## Uso

- Documentación interactiva (Swagger): http://127.0.0.1:8000/docs
- Estado del servidor: http://127.0.0.1:8000/health
- Clasificar muestras: `POST /predict` con un archivo `.csv`
