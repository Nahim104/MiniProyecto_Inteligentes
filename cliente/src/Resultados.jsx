import { useMemo, useState } from "react";
import { ORDEN, UMBRAL_REVISION, pct, subtipo } from "./subtipos.js";

export function Resultados({ predicciones, total }) {
    const clases = useMemo(() => {
        const vistas = Object.keys(predicciones[0].probabilities);
        return [...ORDEN.filter((c) => vistas.includes(c)), ...vistas.filter((c) => !ORDEN.includes(c))];
    }, [predicciones]);

    return (
        <section className="resultados" aria-live="polite">
            <Indicadores predicciones={predicciones} total={total} />
            <div className="panel">
                <h2>Distribución por subtipo predicho</h2>
                <Distribucion predicciones={predicciones} clases={clases} />
            </div>
            <Tabla predicciones={predicciones} clases={clases} />
        </section>
    );
}

function Indicadores({ predicciones, total }) {
    const n = predicciones.length;
    const conteo = contar(predicciones);
    const [top, topN] = Object.entries(conteo).sort((a, b) => b[1] - a[1])[0];
    const media = predicciones.reduce((s, p) => s + p.confidence, 0) / n;
    const revisar = predicciones.filter((p) => p.confidence < UMBRAL_REVISION).length;
    const etiquetadas = predicciones.filter((p) => p.real !== null);
    const aciertos = etiquetadas.filter((p) => p.real === p.predicted).length;

    return (
        <div className="indicadores">
            <Tarjeta titulo="Pacientes clasificados" valor={n} nota={`de ${total} en el lote`} />
            <Tarjeta
                titulo="Subtipo más frecuente"
                valor={subtipo(top).nombre}
                nota={`${topN} pacientes (${pct(topN / n, 0)})`}
                color={subtipo(top).color}
            />
            <Tarjeta titulo="Confianza media" valor={pct(media)} nota="probabilidad del subtipo elegido" />
            {etiquetadas.length ? (
                <Tarjeta
                    titulo="Coinciden con la etiqueta"
                    valor={pct(aciertos / etiquetadas.length)}
                    nota={`${aciertos} de ${etiquetadas.length} con columna type`}
                />
            ) : (
                <Tarjeta
                    titulo="Para revisar"
                    valor={revisar}
                    nota={`confianza menor a ${pct(UMBRAL_REVISION, 0)}`}
                />
            )}
        </div>
    );
}

function Tarjeta({ titulo, valor, nota, color }) {
    return (
        <div className="tarjeta">
            <span className="tarjeta-titulo">{titulo}</span>
            <span className="tarjeta-valor">
                {color && <span className="chip" style={{ background: color }} aria-hidden="true" />}
                {valor}
            </span>
            <span className="tarjeta-nota">{nota}</span>
        </div>
    );
}

const contar = (predicciones) =>
    predicciones.reduce((c, p) => ({ ...c, [p.predicted]: (c[p.predicted] || 0) + 1 }), {});

function Distribucion({ predicciones, clases }) {
    const conteo = contar(predicciones);
    const max = Math.max(...Object.values(conteo));
    return (
        <ul className="distribucion">
            {clases.map((c) => {
                const n = conteo[c] || 0;
                const { nombre, color } = subtipo(c);
                return (
                    <li key={c} title={`${nombre}: ${n} pacientes (${pct(n / predicciones.length)})`}>
                        <span className="dist-nombre">{nombre}</span>
                        <span className="dist-pista">
                            <span
                                className="dist-barra"
                                style={{ width: `${(100 * n) / max}%`, background: color }}
                            />
                        </span>
                        <span className="dist-valor">
                            <strong>{n}</strong> · {pct(n / predicciones.length, 0)}
                        </span>
                    </li>
                );
            })}
        </ul>
    );
}

function Tabla({ predicciones, clases }) {
    const [busqueda, setBusqueda] = useState("");
    const [filtro, setFiltro] = useState("todos");
    const [soloRevisar, setSoloRevisar] = useState(false);
    const hayEtiquetas = predicciones.some((p) => p.real !== null);
    const varios = new Set(predicciones.map((p) => p.archivo)).size > 1;

    const filas = predicciones.filter(
        (p) =>
            (filtro === "todos" || p.predicted === filtro) &&
            (!soloRevisar || p.confidence < UMBRAL_REVISION) &&
            p.sample.toLowerCase().includes(busqueda.trim().toLowerCase()),
    );

    return (
        <div className="panel">
            <div className="tabla-cabecera">
                <h2>Pacientes</h2>
                <button className="btn" onClick={() => descargarCsv(predicciones, clases)}>
                    Descargar CSV
                </button>
            </div>

            <div className="filtros">
                <input
                    type="search"
                    placeholder="Buscar paciente…"
                    aria-label="Buscar paciente"
                    value={busqueda}
                    onChange={(e) => setBusqueda(e.target.value)}
                />
                <select
                    aria-label="Filtrar por subtipo"
                    value={filtro}
                    onChange={(e) => setFiltro(e.target.value)}
                >
                    <option value="todos">Todos los subtipos</option>
                    {clases.map((c) => (
                        <option key={c} value={c}>
                            {subtipo(c).nombre}
                        </option>
                    ))}
                </select>
                <label className="casilla">
                    <input
                        type="checkbox"
                        checked={soloRevisar}
                        onChange={(e) => setSoloRevisar(e.target.checked)}
                    />
                    Solo para revisar
                </label>
            </div>

            <div className="leyenda" aria-label="Leyenda de probabilidades">
                {clases.map((c) => (
                    <span key={c}>
                        <span className="chip" style={{ background: subtipo(c).color }} aria-hidden="true" />
                        {subtipo(c).nombre}
                    </span>
                ))}
            </div>

            <div className="tabla-scroll">
                <table>
                    <thead>
                        <tr>
                            <th>Paciente</th>
                            <th>Subtipo predicho</th>
                            <th>Confianza</th>
                            <th>Probabilidades</th>
                            {hayEtiquetas && <th>Etiqueta CSV</th>}
                        </tr>
                    </thead>
                    <tbody>
                        {filas.map((p) => (
                            <Fila
                                key={p.archivo + p.sample}
                                p={p}
                                clases={clases}
                                hayEtiquetas={hayEtiquetas}
                                varios={varios}
                            />
                        ))}
                    </tbody>
                </table>
                {!filas.length && <p className="vacio">Ningún paciente coincide con los filtros.</p>}
            </div>
            <p className="tabla-pie">
                Mostrando {filas.length} de {predicciones.length} pacientes
            </p>
        </div>
    );
}

function Fila({ p, clases, hayEtiquetas, varios }) {
    const { nombre, color } = subtipo(p.predicted);
    const revisar = p.confidence < UMBRAL_REVISION;
    const detalle = clases.map((c) => `${subtipo(c).nombre}: ${pct(p.probabilities[c])}`).join("\n");

    return (
        <tr>
            <td>
                <span className="paciente">{p.sample}</span>
                {varios && <span className="archivo-origen">{p.archivo}</span>}
            </td>
            <td>
                <span className="subtipo">
                    <span className="chip" style={{ background: color }} aria-hidden="true" />
                    {nombre}
                </span>
                {revisar && <span className="insignia">⚠ Revisar</span>}
            </td>
            <td>
                <span className="confianza">
                    <span className="medidor" aria-hidden="true">
                        <span style={{ width: pct(p.confidence), background: color }} />
                    </span>
                    {pct(p.confidence)}
                </span>
            </td>
            <td>
                <span className="apilada" title={detalle} role="img" aria-label={detalle}>
                    {clases.map((c) =>
                        p.probabilities[c] >= 0.005 ? (
                            <span
                                key={c}
                                style={{ flexGrow: p.probabilities[c], background: subtipo(c).color }}
                            />
                        ) : null,
                    )}
                </span>
            </td>
            {hayEtiquetas && (
                <td>
                    {p.real === null ? (
                        "—"
                    ) : p.real === p.predicted ? (
                        <span className="coincide">✓ {subtipo(p.real).nombre}</span>
                    ) : (
                        <span className="difiere">✗ {subtipo(p.real).nombre}</span>
                    )}
                </td>
            )}
        </tr>
    );
}

const comillas = (v) => `"${String(v).replaceAll('"', '""')}"`;

function descargarCsv(predicciones, clases) {
    const header = ["archivo", "muestra", "etiqueta_csv", "predicho", "confianza_%", ...clases.map((c) => `${c}_%`)];
    const rows = predicciones.map((p) => [
        p.archivo,
        p.sample,
        p.real ?? "",
        p.predicted,
        (p.confidence * 100).toFixed(2),
        ...clases.map((c) => (p.probabilities[c] * 100).toFixed(2)),
    ]);
    const csv = [header, ...rows].map((r) => r.map(comillas).join(",")).join("\r\n");
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob(["﻿" + csv], { type: "text/csv;charset=utf-8;" }));
    link.download = "resultados.csv";
    link.click();
    URL.revokeObjectURL(link.href);
}
