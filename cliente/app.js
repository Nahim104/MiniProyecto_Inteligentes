const API_URL = "http://127.0.0.1:8000/predict";

const form = document.getElementById("upload-form");
const fileInput = document.getElementById("csv-file");
const submitButton = document.getElementById("submit-button");
const message = document.getElementById("message");
const results = document.getElementById("results");
const resultsList = document.getElementById("results-list");
const downloadButton = document.getElementById("download-button");

let predictions = null;

fileInput.addEventListener("change", () => {
    submitButton.disabled = !fileInput.files[0];
    toggleResults(false);
    predictions = null;
    showMessage("");
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const file = fileInput.files[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".csv")) {
        showMessage("El archivo debe ser un CSV.", true);
        return;
    }

    setLoading(true);
    showMessage("Procesando archivo...");

    try {
        const body = new FormData();
        body.append("file", file);

        const response = await fetch(API_URL, { method: "POST", body });
        const data = await response.json().catch(() => null);

        if (!response.ok) throw new Error(errorText(data));
        if (!Array.isArray(data?.predictions) || !data.predictions.length) {
            throw new Error("El servidor devolvió una respuesta inválida.");
        }

        predictions = data.predictions;
        showResults(predictions);
        showMessage(`Se clasificaron ${predictions.length} muestra(s).`);
    } catch (error) {
        toggleResults(false);
        const network = error instanceof TypeError;
        showMessage(
            network ? "No se pudo conectar con el servidor." : error.message,
            true,
        );
    } finally {
        setLoading(false);
    }
});

// El backend responde {detail: "texto"} o {detail: {message, ...}}.
function errorText(data) {
    const detail = data?.detail;
    if (typeof detail === "string") return detail;
    if (detail?.message) {
        const count = detail.missing_count
            ? ` Faltan ${detail.missing_count}.`
            : "";
        return detail.message + count;
    }
    return "Ocurrió un error al procesar el archivo.";
}

function showResults(items) {
    resultsList.replaceChildren();

    items.forEach((item) => {
        const card = document.createElement("article");
        card.className = "card";

        const title = document.createElement("h2");
        title.textContent = `Muestra ${item.sample}`;
        card.append(title);

        if (item.real !== null) {
            const label = document.createElement("p");
            label.className = "etiqueta";
            label.textContent = `Etiqueta del CSV: ${item.real}`;
            card.append(label);
        }

        const list = document.createElement("ul");
        list.className = "card-info";
        Object.entries(item.probabilities)
            .sort((a, b) => b[1] - a[1])
            .forEach(([label, probability]) => {
                const row = document.createElement("li");
                row.classList.toggle("top", label === item.predicted);
                const name = document.createElement("span");
                name.textContent = label;
                const value = document.createElement("span");
                value.textContent = percent(probability);
                row.append(name, value);
                list.append(row);
            });
        card.append(list);
        resultsList.append(card);
    });

    toggleResults(true);
}

downloadButton.addEventListener("click", () => {
    if (!predictions) return;

    const classes = Object.keys(predictions[0].probabilities);
    const header = [
        "muestra",
        "etiqueta_csv",
        "predicho",
        "confianza_%",
        ...classes.map((c) => `${c}_%`),
    ];
    const rows = predictions.map((p) => [
        p.sample,
        p.real ?? "",
        p.predicted,
        (p.confidence * 100).toFixed(2),
        ...classes.map((c) => (p.probabilities[c] * 100).toFixed(2)),
    ]);

    const csv = [header, ...rows]
        .map((r) => r.map(quote).join(","))
        .join("\r\n");
    const blob = new Blob(["\uFEFF" + csv], {
        type: "text/csv;charset=utf-8;",
    });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "resultados.csv";
    link.click();
    URL.revokeObjectURL(link.href);
});

const percent = (value) => `${(value * 100).toFixed(2)} %`;
const quote = (value) => `"${String(value).replaceAll('"', '""')}"`;

function setLoading(loading) {
    submitButton.disabled = loading || !fileInput.files[0];
}

function showMessage(text, isError = false) {
    message.textContent = text;
    message.hidden = !text;
    message.classList.toggle("error", isError);
}

function toggleResults(show) {
    results.hidden = !show;
}
