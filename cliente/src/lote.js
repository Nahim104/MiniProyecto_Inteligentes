// Troceado de CSV en bloques para enviar al backend y mostrar progreso.

// shortcut: parte por saltos de línea; falla con saltos dentro de comillas, usar un parser CSV si aparecen
export function leerFilas(texto) {
    const [header = "", ...rows] = texto
        .split(/\r?\n/)
        .filter((line) => line.trim());
    return { header, rows };
}

// ~20 pasos de progreso sin importar el tamaño del lote, con tope de 25 filas por petición.
export function tamanoBloque(totalFilas) {
    return Math.min(25, Math.max(1, Math.ceil(totalFilas / 20)));
}

export function agrupar({ header, rows }, tamano) {
    const bloques = [];
    for (let i = 0; i < rows.length; i += tamano) {
        const parte = rows.slice(i, i + tamano);
        bloques.push({ filas: parte.length, texto: [header, ...parte].join("\n") });
    }
    return bloques;
}
