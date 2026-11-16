export const API_BASE = "http://127.0.0.1:8000";

export async function servidorDisponible() {
    try {
        return (await fetch(`${API_BASE}/health`)).ok;
    } catch {
        return false;
    }
}

export async function clasificar(file, signal) {
    const body = new FormData();
    body.append("file", file);

    const response = await fetch(`${API_BASE}/predict`, { method: "POST", body, signal });
    const data = await response.json().catch(() => null);

    if (!response.ok) throw new Error(textoError(data));
    if (!Array.isArray(data?.predictions) || !data.predictions.length) {
        throw new Error("El servidor devolvió una respuesta inválida.");
    }
    return data.predictions;
}

// El backend responde {detail: "texto"} o {detail: {message, ...}}.
function textoError(data) {
    const detail = data?.detail;
    if (typeof detail === "string") return detail;
    if (detail?.message) {
        const faltan = detail.missing_count ? ` Faltan ${detail.missing_count}.` : "";
        return detail.message + faltan;
    }
    return "Ocurrió un error al procesar el archivo.";
}
