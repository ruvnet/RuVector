/**
 * Strategy for pooling token embeddings into a single sentence embedding
 * @enum {0 | 1 | 2 | 3 | 4}
 */
export const PoolingStrategy = Object.freeze({
    /**
     * Average all token embeddings (most common)
     */
    Mean: 0, "0": "Mean",
    /**
     * Use only the [CLS] token embedding
     */
    Cls: 1, "1": "Cls",
    /**
     * Take the maximum value across all tokens for each dimension
     */
    Max: 2, "2": "Max",
    /**
     * Mean pooling normalized by sqrt of sequence length
     */
    MeanSqrtLen: 3, "3": "MeanSqrtLen",
    /**
     * Use the last token embedding (for decoder models)
     */
    LastToken: 4, "4": "LastToken",
});

/**
 * WASM-compatible embedder using Tract for inference
 */
export class WasmEmbedder {
    static __wrap(ptr) {
        const obj = Object.create(WasmEmbedder.prototype);
        obj.__wbg_ptr = ptr;
        WasmEmbedderFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmEmbedderFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmembedder_free(ptr, 0);
    }
    /**
     * Get the embedding dimension
     * @returns {number}
     */
    dimension() {
        const ret = wasm.wasmembedder_dimension(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Generate embeddings for multiple texts
     * @param {string[]} texts
     * @returns {Float32Array}
     */
    embedBatch(texts) {
        const ptr0 = passArrayJsValueToWasm0(texts, wasm.__wbindgen_malloc_command_export);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmembedder_embedBatch(this.__wbg_ptr, ptr0, len0);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free_command_export(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Generate embedding for a single text
     * @param {string} text
     * @returns {Float32Array}
     */
    embedOne(text) {
        const ptr0 = passStringToWasm0(text, wasm.__wbindgen_malloc_command_export, wasm.__wbindgen_realloc_command_export);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmembedder_embedOne(this.__wbg_ptr, ptr0, len0);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free_command_export(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Get maximum sequence length
     * @returns {number}
     */
    maxLength() {
        const ret = wasm.wasmembedder_maxLength(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Create a new embedder from model and tokenizer bytes
     *
     * # Arguments
     * * `model_bytes` - ONNX model file bytes
     * * `tokenizer_json` - Tokenizer JSON configuration
     * @param {Uint8Array} model_bytes
     * @param {string} tokenizer_json
     */
    constructor(model_bytes, tokenizer_json) {
        const ptr0 = passArray8ToWasm0(model_bytes, wasm.__wbindgen_malloc_command_export);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(tokenizer_json, wasm.__wbindgen_malloc_command_export, wasm.__wbindgen_realloc_command_export);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmembedder_new(ptr0, len0, ptr1, len1);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        WasmEmbedderFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Compute similarity between two texts
     * @param {string} text1
     * @param {string} text2
     * @returns {number}
     */
    similarity(text1, text2) {
        const ptr0 = passStringToWasm0(text1, wasm.__wbindgen_malloc_command_export, wasm.__wbindgen_realloc_command_export);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(text2, wasm.__wbindgen_malloc_command_export, wasm.__wbindgen_realloc_command_export);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmembedder_similarity(this.__wbg_ptr, ptr0, len0, ptr1, len1);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0];
    }
    /**
     * Create embedder with custom configuration
     * @param {Uint8Array} model_bytes
     * @param {string} tokenizer_json
     * @param {WasmEmbedderConfig} config
     * @returns {WasmEmbedder}
     */
    static withConfig(model_bytes, tokenizer_json, config) {
        const ptr0 = passArray8ToWasm0(model_bytes, wasm.__wbindgen_malloc_command_export);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(tokenizer_json, wasm.__wbindgen_malloc_command_export, wasm.__wbindgen_realloc_command_export);
        const len1 = WASM_VECTOR_LEN;
        _assertClass(config, WasmEmbedderConfig);
        var ptr2 = config.__destroy_into_raw();
        const ret = wasm.wasmembedder_withConfig(ptr0, len0, ptr1, len1, ptr2);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmEmbedder.__wrap(ret[0]);
    }
}
if (Symbol.dispose) WasmEmbedder.prototype[Symbol.dispose] = WasmEmbedder.prototype.free;

/**
 * Configuration for the WASM embedder
 */
export class WasmEmbedderConfig {
    static __wrap(ptr) {
        const obj = Object.create(WasmEmbedderConfig.prototype);
        obj.__wbg_ptr = ptr;
        WasmEmbedderConfigFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmEmbedderConfigFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmembedderconfig_free(ptr, 0);
    }
    /**
     * Create a new configuration
     */
    constructor() {
        const ret = wasm.wasmembedderconfig_new();
        this.__wbg_ptr = ret;
        WasmEmbedderConfigFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Set maximum sequence length
     * @param {number} max_length
     * @returns {WasmEmbedderConfig}
     */
    setMaxLength(max_length) {
        const ptr = this.__destroy_into_raw();
        const ret = wasm.wasmembedderconfig_setMaxLength(ptr, max_length);
        return WasmEmbedderConfig.__wrap(ret);
    }
    /**
     * Set whether to normalize embeddings
     * @param {boolean} normalize
     * @returns {WasmEmbedderConfig}
     */
    setNormalize(normalize) {
        const ptr = this.__destroy_into_raw();
        const ret = wasm.wasmembedderconfig_setNormalize(ptr, normalize);
        return WasmEmbedderConfig.__wrap(ret);
    }
    /**
     * Set pooling strategy (0=Mean, 1=Cls, 2=Max, 3=MeanSqrtLen, 4=LastToken)
     * @param {number} pooling
     * @returns {WasmEmbedderConfig}
     */
    setPooling(pooling) {
        const ptr = this.__destroy_into_raw();
        const ret = wasm.wasmembedderconfig_setPooling(ptr, pooling);
        return WasmEmbedderConfig.__wrap(ret);
    }
}
if (Symbol.dispose) WasmEmbedderConfig.prototype[Symbol.dispose] = WasmEmbedderConfig.prototype.free;

/**
 * Compute cosine similarity between two embedding vectors (JS-friendly)
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @returns {number}
 */
export function cosineSimilarity(a, b) {
    const ptr0 = passArrayF32ToWasm0(a, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF32ToWasm0(b, wasm.__wbindgen_malloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.cosineSimilarity(ptr0, len0, ptr1, len1);
    return ret;
}

/**
 * Initialize panic hook for better error messages in WASM
 */
export function init() {
    wasm.init();
}

/**
 * L2 normalize an embedding vector (JS-friendly)
 * @param {Float32Array} embedding
 * @returns {Float32Array}
 */
export function normalizeL2(embedding) {
    const ptr0 = passArrayF32ToWasm0(embedding, wasm.__wbindgen_malloc_command_export);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.normalizeL2(ptr0, len0);
    var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free_command_export(ret[0], ret[1] * 4, 4);
    return v2;
}

/**
 * Check if SIMD is available (for performance info)
 * Returns true if compiled with WASM SIMD128 support
 * @returns {boolean}
 */
export function simd_available() {
    const ret = wasm.simd_available();
    return ret !== 0;
}

/**
 * Get the library version
 * @returns {string}
 */
export function version() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.version();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free_command_export(deferred1_0, deferred1_1, 1);
    }
}
export function __wbg___wbindgen_is_function_1ff95bcc5517c252(arg0) {
    const ret = typeof(arg0) === 'function';
    return ret;
}
export function __wbg___wbindgen_is_object_a27215656b807791(arg0) {
    const val = arg0;
    const ret = typeof(val) === 'object' && val !== null;
    return ret;
}
export function __wbg___wbindgen_is_string_ea5e6cc2e4141dfe(arg0) {
    const ret = typeof(arg0) === 'string';
    return ret;
}
export function __wbg___wbindgen_is_undefined_c05833b95a3cf397(arg0) {
    const ret = arg0 === undefined;
    return ret;
}
export function __wbg___wbindgen_string_get_b0ca35b86a603356(arg0, arg1) {
    const obj = arg1;
    const ret = typeof(obj) === 'string' ? obj : undefined;
    var ptr1 = isLikeNone(ret) ? 0 : passStringToWasm0(ret, wasm.__wbindgen_malloc_command_export, wasm.__wbindgen_realloc_command_export);
    var len1 = WASM_VECTOR_LEN;
    getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
    getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
}
export function __wbg___wbindgen_throw_344f42d3211c4765(arg0, arg1) {
    throw new Error(getStringFromWasm0(arg0, arg1));
}
export function __wbg_call_a6e5c5dce5018821() { return handleError(function (arg0, arg1, arg2) {
    const ret = arg0.call(arg1, arg2);
    return ret;
}, arguments); }
export function __wbg_crypto_38df2bab126b63dc(arg0) {
    const ret = arg0.crypto;
    return ret;
}
export function __wbg_error_a6fa202b58aa1cd3(arg0, arg1) {
    let deferred0_0;
    let deferred0_1;
    try {
        deferred0_0 = arg0;
        deferred0_1 = arg1;
        console.error(getStringFromWasm0(arg0, arg1));
    } finally {
        wasm.__wbindgen_free_command_export(deferred0_0, deferred0_1, 1);
    }
}
export function __wbg_getRandomValues_c44a50d8cfdaebeb() { return handleError(function (arg0, arg1) {
    arg0.getRandomValues(arg1);
}, arguments); }
export function __wbg_getRandomValues_cc7f052a444bb2ce() { return handleError(function (arg0, arg1) {
    globalThis.crypto.getRandomValues(getArrayU8FromWasm0(arg0, arg1));
}, arguments); }
export function __wbg_length_1f0964f4a5e2c6d8(arg0) {
    const ret = arg0.length;
    return ret;
}
export function __wbg_msCrypto_bd5a034af96bcba6(arg0) {
    const ret = arg0.msCrypto;
    return ret;
}
export function __wbg_new_227d7c05414eb861() {
    const ret = new Error();
    return ret;
}
export function __wbg_new_with_length_e6785c33c8e4cce8(arg0) {
    const ret = new Uint8Array(arg0 >>> 0);
    return ret;
}
export function __wbg_node_84ea875411254db1(arg0) {
    const ret = arg0.node;
    return ret;
}
export function __wbg_process_44c7a14e11e9f69e(arg0) {
    const ret = arg0.process;
    return ret;
}
export function __wbg_prototypesetcall_4770620bbe4688a0(arg0, arg1, arg2) {
    Uint8Array.prototype.set.call(getArrayU8FromWasm0(arg0, arg1), arg2);
}
export function __wbg_randomFillSync_6c25eac9869eb53c() { return handleError(function (arg0, arg1) {
    arg0.randomFillSync(arg1);
}, arguments); }
export function __wbg_require_b4edbdcf3e2a1ef0() { return handleError(function () {
    const ret = module.require;
    return ret;
}, arguments); }
export function __wbg_stack_3b0d974bbf31e44f(arg0, arg1) {
    const ret = arg1.stack;
    const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc_command_export, wasm.__wbindgen_realloc_command_export);
    const len1 = WASM_VECTOR_LEN;
    getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
    getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
}
export function __wbg_static_accessor_GLOBAL_4ef717fb391d88b7() {
    const ret = typeof global === 'undefined' ? null : global;
    return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
}
export function __wbg_static_accessor_GLOBAL_THIS_8d1badc68b5a74f4() {
    const ret = typeof globalThis === 'undefined' ? null : globalThis;
    return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
}
export function __wbg_static_accessor_SELF_146583524fe1469b() {
    const ret = typeof self === 'undefined' ? null : self;
    return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
}
export function __wbg_static_accessor_WINDOW_f2829a2234d7819e() {
    const ret = typeof window === 'undefined' ? null : window;
    return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
}
export function __wbg_subarray_3ed232c8a6baee09(arg0, arg1, arg2) {
    const ret = arg0.subarray(arg1 >>> 0, arg2 >>> 0);
    return ret;
}
export function __wbg_versions_276b2795b1c6a219(arg0) {
    const ret = arg0.versions;
    return ret;
}
export function __wbindgen_cast_0000000000000001(arg0, arg1) {
    // Cast intrinsic for `Ref(Slice(U8)) -> NamedExternref("Uint8Array")`.
    const ret = getArrayU8FromWasm0(arg0, arg1);
    return ret;
}
export function __wbindgen_cast_0000000000000002(arg0, arg1) {
    // Cast intrinsic for `Ref(String) -> Externref`.
    const ret = getStringFromWasm0(arg0, arg1);
    return ret;
}
export function __wbindgen_init_externref_table() {
    const table = wasm.__wbindgen_externrefs;
    const offset = table.grow(4);
    table.set(0, undefined);
    table.set(offset + 0, undefined);
    table.set(offset + 1, null);
    table.set(offset + 2, true);
    table.set(offset + 3, false);
}
const WasmEmbedderFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmembedder_free(ptr, 1));
const WasmEmbedderConfigFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmembedderconfig_free(ptr, 1));

function addToExternrefTable0(obj) {
    const idx = wasm.__externref_table_alloc_command_export();
    wasm.__wbindgen_externrefs.set(idx, obj);
    return idx;
}

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
}

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    return decodeText(ptr >>> 0, len);
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        const idx = addToExternrefTable0(e);
        wasm.__wbindgen_exn_store_command_export(idx);
    }
}

function isLikeNone(x) {
    return x === undefined || x === null;
}

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getFloat32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayJsValueToWasm0(array, malloc) {
    const ptr = malloc(array.length * 4, 4) >>> 0;
    for (let i = 0; i < array.length; i++) {
        const add = addToExternrefTable0(array[i]);
        getDataViewMemory0().setUint32(ptr + 4 * i, add, true);
    }
    WASM_VECTOR_LEN = array.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc_command_export(idx);
    return value;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;


let wasm;
export function __wbg_set_wasm(val) {
    wasm = val;
}
