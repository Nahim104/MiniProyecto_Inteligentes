import { useState } from "react";

const tamanoLegible = (bytes) =>
    bytes > 1e6 ? `${(bytes / 1e6).toFixed(1)} MB` : `${Math.ceil(bytes / 1e3)} KB`;

export function ZonaCarga({ archivos, setArchivos, deshabilitado }) {
    const [encima, setEncima] = useState(false);
    const [aviso, setAviso] = useState("");

    function agregar(lista) {
        const nuevos = [...lista];
        const csv = nuevos.filter((f) => f.name.toLowerCase().endsWith(".csv"));
        setAviso(
            csv.length < nuevos.length ? "Se ignoraron archivos que no son .csv." : "",
        );
        // Evita duplicados por nombre y tamaño.
        setArchivos((prev) => [
            ...prev,
            ...csv.filter((f) => !prev.some((p) => p.name === f.name && p.size === f.size)),
        ]);
    }

    return (
        <div>
            <label
                className={`dropzone${encima ? " encima" : ""}${deshabilitado ? " deshabilitada" : ""}`}
                onDragOver={(e) => {
                    e.preventDefault();
                    if (!deshabilitado) setEncima(true);
                }}
                onDragLeave={() => setEncima(false)}
                onDrop={(e) => {
                    e.preventDefault();
                    setEncima(false);
                    if (!deshabilitado) agregar(e.dataTransfer.files);
                }}
            >
                <input
                    type="file"
                    accept=".csv,text/csv"
                    multiple
                    className="oculto"
                    disabled={deshabilitado}
                    onChange={(e) => {
                        agregar(e.target.files);
                        e.target.value = "";
                    }}
                />
                <svg className="dropzone-icono" viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M12 16V4m0 0-4 4m4-4 4 4M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
                </svg>
                <span className="dropzone-titulo">
                    Arrastra tus archivos CSV aquí o <u>haz clic para elegirlos</u>
                </span>
                <span className="dropzone-ayuda">
                    Puedes subir varios archivos a la vez · columnas opcionales:
                    <code>samples</code> y <code>type</code>
                </span>
            </label>

            {aviso && <p className="aviso">{aviso}</p>}

            {archivos.length > 0 && (
                <ul className="archivos">
                    {archivos.map((f) => (
                        <li key={f.name + f.size}>
                            <span className="archivo-icono" aria-hidden="true">CSV</span>
                            <span className="archivo-nombre">{f.name}</span>
                            <span className="archivo-tamano">{tamanoLegible(f.size)}</span>
                            <button
                                className="quitar"
                                aria-label={`Quitar ${f.name}`}
                                disabled={deshabilitado}
                                onClick={() => setArchivos((prev) => prev.filter((p) => p !== f))}
                            >
                                ×
                            </button>
                        </li>
                    ))}
                </ul>
            )}
        </div>
    );
}

const duracion = (ms) => {
    const s = Math.max(0, Math.round(ms / 1000));
    return s < 60 ? `${s} s` : `${Math.floor(s / 60)} min ${s % 60} s`;
};

export function Progreso({ progreso, procesando }) {
    const { hechos, bloque, total, archivo, inicio, ahora, terminado, cancelado } = progreso;
    const hasta = Math.min(hechos + Math.max(bloque, 1), total);
    const porcentaje = total ? Math.round((100 * hechos) / total) : 100;
    const transcurrido = ahora - inicio;
    const restante = hechos ? (transcurrido / hechos) * (total - hechos) : null;

    const titulo = terminado
        ? cancelado
            ? "Proceso detenido"
            : "Lote completado"
        : hasta > hechos + 1
          ? `Clasificando pacientes ${hechos + 1}–${hasta} de ${total}`
          : `Clasificando paciente ${hasta} de ${total}`;

    return (
        <div className={`progreso${terminado ? " terminado" : ""}`}>
            <div className="progreso-cabecera">
                <strong>{titulo}</strong>
                <span className="progreso-pct">{porcentaje} %</span>
            </div>
            <div
                className={`pista${procesando ? " activa" : ""}`}
                role="progressbar"
                aria-valuemin={0}
                aria-valuemax={total}
                aria-valuenow={hechos}
                aria-label="Progreso del lote"
            >
                <div className="relleno" style={{ width: `${porcentaje}%` }} />
            </div>
            <div className="progreso-detalle">
                <span>
                    {hechos} de {total} pacientes
                    {archivo && !terminado ? ` · ${archivo}` : ""}
                </span>
                <span>
                    {terminado
                        ? `Tiempo total: ${duracion(transcurrido)}`
                        : restante !== null
                          ? `Faltan ~${duracion(restante)}`
                          : "Calculando tiempo…"}
                </span>
            </div>
        </div>
    );
}
