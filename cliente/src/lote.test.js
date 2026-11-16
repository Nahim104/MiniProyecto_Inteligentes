import { test } from "node:test";
import assert from "node:assert/strict";
import { agrupar, leerFilas, tamanoBloque } from "./lote.js";

test("leerFilas separa encabezado y descarta líneas vacías", () => {
    const r = leerFilas("samples,g1\r\nP1,1\n\nP2,2\n");
    assert.equal(r.header, "samples,g1");
    assert.deepEqual(r.rows, ["P1,1", "P2,2"]);
});

test("tamanoBloque da progreso visible en lotes chicos y topa en 25", () => {
    assert.equal(tamanoBloque(3), 1);
    assert.equal(tamanoBloque(60), 3);
    assert.equal(tamanoBloque(10000), 25);
});

test("agrupar repite el encabezado y conserva todas las filas", () => {
    const bloques = agrupar({ header: "h", rows: ["a", "b", "c"] }, 2);
    assert.deepEqual(bloques, [
        { filas: 2, texto: "h\na\nb" },
        { filas: 1, texto: "h\nc" },
    ]);
});
