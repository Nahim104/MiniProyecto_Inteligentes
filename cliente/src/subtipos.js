// Color fijo por subtipo (sigue a la entidad, no al ranking). Orden = clases del modelo.
const SUBTIPOS = {
    basal: { nombre: "Basal", color: "var(--series-1)" },
    HER: { nombre: "HER2", color: "var(--series-2)" },
    luminal_A: { nombre: "Luminal A", color: "var(--series-3)" },
    luminal_B: { nombre: "Luminal B", color: "var(--series-4)" },
};

export const ORDEN = Object.keys(SUBTIPOS);

export const subtipo = (clave) =>
    SUBTIPOS[clave] ?? { nombre: clave, color: "var(--series-other)" };

export const UMBRAL_REVISION = 0.6;

export const pct = (v, dec = 1) => `${(v * 100).toFixed(dec)} %`;
