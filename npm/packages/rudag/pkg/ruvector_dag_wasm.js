/* @ts-self-types="./ruvector_dag_wasm.d.ts" */
import * as wasm from "./ruvector_dag_wasm_bg.wasm";
import { __wbg_set_wasm } from "./ruvector_dag_wasm_bg.js";

__wbg_set_wasm(wasm);

export {
    WasmDag
} from "./ruvector_dag_wasm_bg.js";
