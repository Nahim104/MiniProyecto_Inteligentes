import { useEffect, useRef, useState } from "react";
import { clasificar, servidorDisponible } from "./api.js";
import { agrupar, leerFilas, tamanoBloque } from "./lote.js";
import { Progreso, ZonaCarga } from "./Carga.jsx";
import { Resultados } from "./Resultados.jsx";

export default function App() {
    const [archivos, setArchivos] = useState([]);
    const [procesando, setProcesando] = useState(false);
    const [progreso, setProgreso] = useState(null);
    const [predicciones, setPredicciones] = useState([]);
    const [errores, setErrores] = useState([]);
    const [servidor, setServidor] = useState("revisando");
    const abortar = useRef(null);

    useEffect(() => {
        servidorDisponible().then((ok) => setServidor(ok ? "ok" : "caido"));
    }, []);

    async function procesar() {
        const control = new AbortController();
        abortar.current = control;
        setProcesando(true);
        setPredicciones([]);
        setErrores([]);

        const trabajos = await Promise.all(
            archivos.map(async (f) => ({ nombre: f.name, ...leerFilas(await f.text()) })),
        );
        const total = trabajos.reduce((n, t) => n + t.rows.length, 0);
        const tamano = tamanoBloque(total);
        const inicio = performance.now();
        let hechos = 0;
        const avanzar = (archivo, bloque = 0) =>
            setProgreso({ hechos, bloque, total, archivo, inicio, ahora: performance.now() });

        avanzar(trabajos[0]?.nombre);

        for (const t of trabajos) {
            if (control.signal.aborted) break;
            if (!t.rows.length) {
                setErrores((e) => [...e, { archivo: t.nombre, texto: "No contiene pacientes." }]);
                continue;
            }
            const bloques = agrupar(t, tamano);
            for (const [i, b] of bloques.entries()) {
                avanzar(t.nombre, b.filas);
                try {
                    const items = await clasificar(
                        new File([b.texto], t.nombre, { type: "text/csv" }),
                        control.signal,
                    );
                    const conArchivo = items.map((p) => ({ ...p, archivo: t.nombre }));
                    setPredicciones((prev) => [...prev, ...conArchivo]);
                    setServidor("ok");
                    hechos += b.filas;
                } catch (error) {
                    if (control.signal.aborted) break;
                    const red = error instanceof TypeError;
                    if (red) setServidor("caido");
                    setErrores((e) => [
                        ...e,
                        {
                            archivo: t.nombre,
                            texto: red ? "No se pudo conectar con el servidor." : error.message,
                        },
                    ]);
                    // se omite el resto de este archivo, pero se sigue con los demás
                    hechos += bloques.slice(i).reduce((m, x) => m + x.filas, 0);
                    break;
                }
            }
        }

        avanzar(null);
        setProcesando(false);
        setProgreso((p) => ({ ...p, terminado: true, cancelado: control.signal.aborted }));
    }

    return (
        <>
            <header className="topbar">
                <div className="topbar-inner">
                    <span className="brand">
                        <span className="brand-mark" aria-hidden="true">◎</span>
                        Clasificador de subtipos de cáncer de mama
                    </span>
                    <EstadoServidor estado={servidor} />
                </div>
            </header>

            <main className="container">
                <section className="hero">
                    <h1>Clasifica lotes de pacientes</h1>
                    <p>
                        Sube uno o varios CSV de expresión genética. Cada fila es un
                        paciente; verás el avance en vivo y los resultados se irán
                        acumulando a medida que el modelo los clasifica.
                    </p>
                </section>

                <section className="panel">
                    <ZonaCarga
                        archivos={archivos}
                        setArchivos={setArchivos}
                        deshabilitado={procesando}
                    />
                    <div className="acciones">
                        {procesando ? (
                            <button className="btn" onClick={() => abortar.current?.abort()}>
                                Detener
                            </button>
                        ) : (
                            <button
                                className="btn primario"
                                disabled={!archivos.length}
                                onClick={procesar}
                            >
                                Clasificar pacientes
                            </button>
                        )}
                    </div>
                    {progreso && (
                        <Progreso progreso={progreso} procesando={procesando} />
                    )}
                    {errores.length > 0 && (
                        <ul className="alertas" role="alert">
                            {errores.map((e, i) => (
                                <li key={i}>
                                    <strong>{e.archivo}:</strong> {e.texto}
                                </li>
                            ))}
                        </ul>
                    )}
                </section>

                {predicciones.length > 0 && (
                    <Resultados
                        predicciones={predicciones}
                        total={progreso?.total ?? predicciones.length}
                    />
                )}
            </main>

            <footer className="footer">
                Clasificador de cáncer de mama · 1905 Sistemas Inteligentes 2026Q4
            </footer>
        </>
    );
}

function EstadoServidor({ estado }) {
    const textos = {
        revisando: "Comprobando servidor…",
        ok: "Servidor conectado",
        caido: "Servidor no disponible",
    };
    return (
        <span className={`estado estado-${estado}`} role="status">
            <span className="punto" aria-hidden="true" />
            {textos[estado]}
        </span>
    );
}
