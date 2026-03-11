/* @ts-self-types="./ruvector_cnn_wasm.d.ts" */

/**
 * Configuration for CNN embedder
 */
export class EmbedderConfig {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        EmbedderConfigFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_embedderconfig_free(ptr, 0);
    }
    constructor() {
        const ret = wasm.embedderconfig_new();
        this.__wbg_ptr = ret >>> 0;
        EmbedderConfigFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Output embedding dimension
     * @returns {number}
     */
    get embedding_dim() {
        const ret = wasm.__wbg_get_embedderconfig_embedding_dim(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Input image size (square)
     * @returns {number}
     */
    get input_size() {
        const ret = wasm.__wbg_get_embedderconfig_input_size(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Whether to L2 normalize embeddings
     * @returns {boolean}
     */
    get normalize() {
        const ret = wasm.__wbg_get_embedderconfig_normalize(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Output embedding dimension
     * @param {number} arg0
     */
    set embedding_dim(arg0) {
        wasm.__wbg_set_embedderconfig_embedding_dim(this.__wbg_ptr, arg0);
    }
    /**
     * Input image size (square)
     * @param {number} arg0
     */
    set input_size(arg0) {
        wasm.__wbg_set_embedderconfig_input_size(this.__wbg_ptr, arg0);
    }
    /**
     * Whether to L2 normalize embeddings
     * @param {boolean} arg0
     */
    set normalize(arg0) {
        wasm.__wbg_set_embedderconfig_normalize(this.__wbg_ptr, arg0);
    }
}
if (Symbol.dispose) EmbedderConfig.prototype[Symbol.dispose] = EmbedderConfig.prototype.free;

/**
 * Layer operations for building custom networks
 */
export class LayerOps {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        LayerOpsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_layerops_free(ptr, 0);
    }
    /**
     * Apply batch normalization (returns new array)
     * @param {Float32Array} input
     * @param {Float32Array} gamma
     * @param {Float32Array} beta
     * @param {Float32Array} mean
     * @param {Float32Array} _var
     * @param {number} epsilon
     * @returns {Float32Array}
     */
    static batch_norm(input, gamma, beta, mean, _var, epsilon) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_export2);
            const len0 = WASM_VECTOR_LEN;
            const ptr1 = passArrayF32ToWasm0(gamma, wasm.__wbindgen_export2);
            const len1 = WASM_VECTOR_LEN;
            const ptr2 = passArrayF32ToWasm0(beta, wasm.__wbindgen_export2);
            const len2 = WASM_VECTOR_LEN;
            const ptr3 = passArrayF32ToWasm0(mean, wasm.__wbindgen_export2);
            const len3 = WASM_VECTOR_LEN;
            const ptr4 = passArrayF32ToWasm0(_var, wasm.__wbindgen_export2);
            const len4 = WASM_VECTOR_LEN;
            wasm.layerops_batch_norm(retptr, ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, ptr4, len4, epsilon);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v6 = getArrayF32FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export(r0, r1 * 4, 4);
            return v6;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Apply global average pooling
     * Returns one value per channel
     * @param {Float32Array} input
     * @param {number} height
     * @param {number} width
     * @param {number} channels
     * @returns {Float32Array}
     */
    static global_avg_pool(input, height, width, channels) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passArrayF32ToWasm0(input, wasm.__wbindgen_export2);
            const len0 = WASM_VECTOR_LEN;
            wasm.layerops_global_avg_pool(retptr, ptr0, len0, height, width, channels);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v2 = getArrayF32FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export(r0, r1 * 4, 4);
            return v2;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
}
if (Symbol.dispose) LayerOps.prototype[Symbol.dispose] = LayerOps.prototype.free;

/**
 * SIMD-optimized operations
 */
export class SimdOps {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        SimdOpsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_simdops_free(ptr, 0);
    }
    /**
     * Dot product of two vectors
     * @param {Float32Array} a
     * @param {Float32Array} b
     * @returns {number}
     */
    static dot_product(a, b) {
        const ptr0 = passArrayF32ToWasm0(a, wasm.__wbindgen_export2);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(b, wasm.__wbindgen_export2);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.simdops_dot_product(ptr0, len0, ptr1, len1);
        return ret;
    }
    /**
     * L2 normalize a vector (returns new array)
     * @param {Float32Array} data
     * @returns {Float32Array}
     */
    static l2_normalize(data) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passArrayF32ToWasm0(data, wasm.__wbindgen_export2);
            const len0 = WASM_VECTOR_LEN;
            wasm.simdops_l2_normalize(retptr, ptr0, len0);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v2 = getArrayF32FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export(r0, r1 * 4, 4);
            return v2;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * ReLU activation (returns new array)
     * @param {Float32Array} data
     * @returns {Float32Array}
     */
    static relu(data) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passArrayF32ToWasm0(data, wasm.__wbindgen_export2);
            const len0 = WASM_VECTOR_LEN;
            wasm.simdops_relu(retptr, ptr0, len0);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v2 = getArrayF32FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export(r0, r1 * 4, 4);
            return v2;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * ReLU6 activation (returns new array)
     * @param {Float32Array} data
     * @returns {Float32Array}
     */
    static relu6(data) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passArrayF32ToWasm0(data, wasm.__wbindgen_export2);
            const len0 = WASM_VECTOR_LEN;
            wasm.simdops_relu6(retptr, ptr0, len0);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v2 = getArrayF32FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export(r0, r1 * 4, 4);
            return v2;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
}
if (Symbol.dispose) SimdOps.prototype[Symbol.dispose] = SimdOps.prototype.free;

/**
 * WASM CNN Embedder for image feature extraction
 */
export class WasmCnnEmbedder {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmCnnEmbedderFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmcnnembedder_free(ptr, 0);
    }
    /**
     * Compute cosine similarity between two embeddings
     * @param {Float32Array} a
     * @param {Float32Array} b
     * @returns {number}
     */
    cosine_similarity(a, b) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passArrayF32ToWasm0(a, wasm.__wbindgen_export2);
            const len0 = WASM_VECTOR_LEN;
            const ptr1 = passArrayF32ToWasm0(b, wasm.__wbindgen_export2);
            const len1 = WASM_VECTOR_LEN;
            wasm.wasmcnnembedder_cosine_similarity(retptr, this.__wbg_ptr, ptr0, len0, ptr1, len1);
            var r0 = getDataViewMemory0().getFloat32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return r0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Get the embedding dimension
     * @returns {number}
     */
    get embedding_dim() {
        const ret = wasm.wasmcnnembedder_embedding_dim(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Extract embedding from image data (RGB format, row-major)
     * @param {Uint8Array} image_data
     * @param {number} width
     * @param {number} height
     * @returns {Float32Array}
     */
    extract(image_data, width, height) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passArray8ToWasm0(image_data, wasm.__wbindgen_export2);
            const len0 = WASM_VECTOR_LEN;
            wasm.wasmcnnembedder_extract(retptr, this.__wbg_ptr, ptr0, len0, width, height);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            var r3 = getDataViewMemory0().getInt32(retptr + 4 * 3, true);
            if (r3) {
                throw takeObject(r2);
            }
            var v2 = getArrayF32FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export(r0, r1 * 4, 4);
            return v2;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Create a new CNN embedder
     * @param {EmbedderConfig | null} [config]
     */
    constructor(config) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            let ptr0 = 0;
            if (!isLikeNone(config)) {
                _assertClass(config, EmbedderConfig);
                ptr0 = config.__destroy_into_raw();
            }
            wasm.wasmcnnembedder_new(retptr, ptr0);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            this.__wbg_ptr = r0 >>> 0;
            WasmCnnEmbedderFinalization.register(this, this.__wbg_ptr, this);
            return this;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
}
if (Symbol.dispose) WasmCnnEmbedder.prototype[Symbol.dispose] = WasmCnnEmbedder.prototype.free;

/**
 * InfoNCE loss for contrastive learning (SimCLR style)
 */
export class WasmInfoNCELoss {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmInfoNCELossFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasminfonceloss_free(ptr, 0);
    }
    /**
     * Compute loss for a batch of embedding pairs
     * embeddings: [2N, D] flattened where (i, i+N) are positive pairs
     * @param {Float32Array} embeddings
     * @param {number} batch_size
     * @param {number} dim
     * @returns {number}
     */
    forward(embeddings, batch_size, dim) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passArrayF32ToWasm0(embeddings, wasm.__wbindgen_export2);
            const len0 = WASM_VECTOR_LEN;
            wasm.wasminfonceloss_forward(retptr, this.__wbg_ptr, ptr0, len0, batch_size, dim);
            var r0 = getDataViewMemory0().getFloat32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return r0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Create new InfoNCE loss with temperature parameter
     * @param {number} temperature
     */
    constructor(temperature) {
        const ret = wasm.wasminfonceloss_new(temperature);
        this.__wbg_ptr = ret >>> 0;
        WasmInfoNCELossFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Get the temperature parameter
     * @returns {number}
     */
    get temperature() {
        const ret = wasm.wasminfonceloss_temperature(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) WasmInfoNCELoss.prototype[Symbol.dispose] = WasmInfoNCELoss.prototype.free;

/**
 * Triplet loss for metric learning
 */
export class WasmTripletLoss {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmTripletLossFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmtripletloss_free(ptr, 0);
    }
    /**
     * Compute loss for a batch of triplets
     * @param {Float32Array} anchors
     * @param {Float32Array} positives
     * @param {Float32Array} negatives
     * @param {number} dim
     * @returns {number}
     */
    forward(anchors, positives, negatives, dim) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passArrayF32ToWasm0(anchors, wasm.__wbindgen_export2);
            const len0 = WASM_VECTOR_LEN;
            const ptr1 = passArrayF32ToWasm0(positives, wasm.__wbindgen_export2);
            const len1 = WASM_VECTOR_LEN;
            const ptr2 = passArrayF32ToWasm0(negatives, wasm.__wbindgen_export2);
            const len2 = WASM_VECTOR_LEN;
            wasm.wasmtripletloss_forward(retptr, this.__wbg_ptr, ptr0, len0, ptr1, len1, ptr2, len2, dim);
            var r0 = getDataViewMemory0().getFloat32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return r0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Compute loss for a single triplet
     * @param {Float32Array} anchor
     * @param {Float32Array} positive
     * @param {Float32Array} negative
     * @returns {number}
     */
    forward_single(anchor, positive, negative) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passArrayF32ToWasm0(anchor, wasm.__wbindgen_export2);
            const len0 = WASM_VECTOR_LEN;
            const ptr1 = passArrayF32ToWasm0(positive, wasm.__wbindgen_export2);
            const len1 = WASM_VECTOR_LEN;
            const ptr2 = passArrayF32ToWasm0(negative, wasm.__wbindgen_export2);
            const len2 = WASM_VECTOR_LEN;
            wasm.wasmtripletloss_forward_single(retptr, this.__wbg_ptr, ptr0, len0, ptr1, len1, ptr2, len2);
            var r0 = getDataViewMemory0().getFloat32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return r0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Get the margin parameter
     * @returns {number}
     */
    get margin() {
        const ret = wasm.wasmtripletloss_margin(this.__wbg_ptr);
        return ret;
    }
    /**
     * Create new triplet loss with margin
     * @param {number} margin
     */
    constructor(margin) {
        const ret = wasm.wasmtripletloss_new(margin);
        this.__wbg_ptr = ret >>> 0;
        WasmTripletLossFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
}
if (Symbol.dispose) WasmTripletLoss.prototype[Symbol.dispose] = WasmTripletLoss.prototype.free;

/**
 * Initialize panic hook for better error messages
 */
export function init() {
    wasm.init();
}

function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_throw_39bc967c0e5a9b58: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg_error_a6fa202b58aa1cd3: function(arg0, arg1) {
            let deferred0_0;
            let deferred0_1;
            try {
                deferred0_0 = arg0;
                deferred0_1 = arg1;
                console.error(getStringFromWasm0(arg0, arg1));
            } finally {
                wasm.__wbindgen_export(deferred0_0, deferred0_1, 1);
            }
        },
        __wbg_new_227d7c05414eb861: function() {
            const ret = new Error();
            return addHeapObject(ret);
        },
        __wbg_stack_3b0d974bbf31e44f: function(arg0, arg1) {
            const ret = getObject(arg1).stack;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_export2, wasm.__wbindgen_export3);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbindgen_cast_0000000000000001: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return addHeapObject(ret);
        },
        __wbindgen_object_drop_ref: function(arg0) {
            takeObject(arg0);
        },
    };
    return {
        __proto__: null,
        "./ruvector_cnn_wasm_bg.js": import0,
    };
}

const EmbedderConfigFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_embedderconfig_free(ptr >>> 0, 1));
const LayerOpsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_layerops_free(ptr >>> 0, 1));
const SimdOpsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_simdops_free(ptr >>> 0, 1));
const WasmCnnEmbedderFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmcnnembedder_free(ptr >>> 0, 1));
const WasmInfoNCELossFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasminfonceloss_free(ptr >>> 0, 1));
const WasmTripletLossFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmtripletloss_free(ptr >>> 0, 1));

function addHeapObject(obj) {
    if (heap_next === heap.length) heap.push(heap.length + 1);
    const idx = heap_next;
    heap_next = heap[idx];

    heap[idx] = obj;
    return idx;
}

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
}

function dropObject(idx) {
    if (idx < 1028) return;
    heap[idx] = heap_next;
    heap_next = idx;
}

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
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
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function getObject(idx) { return heap[idx]; }

let heap = new Array(1024).fill(undefined);
heap.push(undefined, null, true, false);

let heap_next = heap.length;

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

function takeObject(idx) {
    const ret = getObject(idx);
    dropObject(idx);
    return ret;
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

let wasmModule, wasm;
function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    wasmModule = module;
    cachedDataViewMemory0 = null;
    cachedFloat32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('ruvector_cnn_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
