# Clasificador de subtipos de cáncer de mama

- **Backend** (`servidor/`): API en FastAPI que recibe un CSV de expresión genética y devuelve el subtipo predicho y las probabilidades de cada clase.
- **Frontend** (`cliente/`): aplicación React (Vite) para subir uno o varios CSV, ver el progreso del lote y explorar los resultados por paciente.

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

### Requisitos

- Node.js 18 o superior.

### Ejecutar

Primero correr el backend y después:

```bash
cd cliente
npm install
npm run dev
```

Abrir http://localhost:5173 en el navegador.

- `npm run build` genera la versión estática en `cliente/dist/` (se puede servir con `npm run preview`).
- `npm test` corre las pruebas del troceado de lotes.

### Cómo funciona

Cada CSV se parte en bloques de filas (unos 20 bloques por lote, máximo 25 filas por bloque) y cada bloque se envía a `POST /predict`. Así la barra de progreso avanza por bloques y los resultados se van acumulando sin esperar al final. Si un archivo falla, se muestra el error y se sigue con los demás.

Si la página indica "Servidor no disponible", revisar que el backend esté corriendo y que la dirección en `cliente/src/api.js` (`API_BASE`) coincida con la suya.
