/* @ts-self-types="./ruvector_onnx_embeddings_wasm.d.ts" */
import * as wasm from "./ruvector_onnx_embeddings_wasm_bg.wasm";
import { __wbg_set_wasm } from "./ruvector_onnx_embeddings_wasm_bg.js";

__wbg_set_wasm(wasm);
wasm.__wbindgen_start();
export {
    PoolingStrategy, WasmEmbedder, WasmEmbedderConfig, cosineSimilarity, init, normalizeL2, simd_available, version
} from "./ruvector_onnx_embeddings_wasm_bg.js";
