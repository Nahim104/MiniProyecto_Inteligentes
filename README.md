# Clasificador de subtipos de cáncer de mama

- **Backend** (`servidor/`): API en FastAPI que recibe un CSV de expresión genética y devuelve el subtipo predicho y las probabilidades de cada clase.
- **Frontend** (`frontend/`): página estática para subir el CSV y ver los resultados.

## Backend

### Requisitos

- uv [(opciones de instalación)](https://docs.astral.sh/uv/getting-started/installation/).
- Python 3.13. El archivo `.python-version` lo fija y uv lo descarga automáticamente si no lo encuentra.

### Ejecutar

```bash
cd servidor
uv sync
uv run uvicorn servidor.main:app --app-dir src
```

Para desarrollo, añadir `--reload` al último comando (reinicia el servidor al cambiar el código y vuelve a cargar el modelo).

### Uso

- Documentación interactiva (Swagger): http://127.0.0.1:8000/docs
- Estado del servidor: http://127.0.0.1:8000/health
- Clasificar muestras: `POST /predict` con un archivo `.csv`

## Frontend

Es una página estática (HTML, CSS y JavaScript), sin instalación ni servidor propio. Primero correr el backend y después abrir `frontend/index.html` en el navegador.

Si la página no logra conectar con el servidor, revisar que el backend esté corriendo y que la dirección en `frontend/app.js` (`API_URL`) coincida con la suya.
