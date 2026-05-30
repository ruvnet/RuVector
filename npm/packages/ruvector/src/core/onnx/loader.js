/**
 * Model Loader for RuVector ONNX Embeddings WASM
 *
 * Provides easy loading of pre-trained models from HuggingFace Hub
 */

/**
 * Pre-configured models with their HuggingFace URLs
 */
export const MODELS = {
    // Sentence Transformers - Small & Fast
    'all-MiniLM-L6-v2': {
        name: 'all-MiniLM-L6-v2',
        dimension: 384,
        maxLength: 256,
        size: '23MB',
        description: 'Fast, general-purpose embeddings',
        model: 'https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx',
        tokenizer: 'https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json',
    },
    'all-MiniLM-L12-v2': {
        name: 'all-MiniLM-L12-v2',
        dimension: 384,
        maxLength: 256,
        size: '33MB',
        description: 'Better quality, balanced speed',
        model: 'https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2/resolve/main/onnx/model.onnx',
        tokenizer: 'https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2/resolve/main/tokenizer.json',
    },

    // BGE Models - State of the art
    'bge-small-en-v1.5': {
        name: 'bge-small-en-v1.5',
        dimension: 384,
        maxLength: 512,
        size: '33MB',
        description: 'State-of-the-art small model',
        model: 'https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx',
        tokenizer: 'https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/tokenizer.json',
    },
    'bge-base-en-v1.5': {
        name: 'bge-base-en-v1.5',
        dimension: 768,
        maxLength: 512,
        size: '110MB',
        description: 'Best overall quality',
        model: 'https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/onnx/model.onnx',
        tokenizer: 'https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/tokenizer.json',
    },

    // E5 Models - Microsoft
    'e5-small-v2': {
        name: 'e5-small-v2',
        dimension: 384,
        maxLength: 512,
        size: '33MB',
        description: 'Excellent for search & retrieval',
        model: 'https://huggingface.co/intfloat/e5-small-v2/resolve/main/onnx/model.onnx',
        tokenizer: 'https://huggingface.co/intfloat/e5-small-v2/resolve/main/tokenizer.json',
    },

    // GTE Models - Alibaba
    'gte-small': {
        name: 'gte-small',
        dimension: 384,
        maxLength: 512,
        size: '33MB',
        description: 'Good multilingual support',
        model: 'https://huggingface.co/thenlper/gte-small/resolve/main/onnx/model.onnx',
        tokenizer: 'https://huggingface.co/thenlper/gte-small/resolve/main/tokenizer.json',
    },
};

/**
 * Default model for quick start
 */
export const DEFAULT_MODEL = 'all-MiniLM-L6-v2';

/**
 * In-memory memo of loaded models, keyed by model name. Deduplicates the
 * (re-)download + decode when multiple embedder instances load the same model
 * in one process. In Node there is no Cache API, so without this every
 * ModelLoader.loadModel() re-fetches the model from HuggingFace (issue #523).
 */
const _inMemoryModelCache = new Map();

/**
 * Model loader with caching support
 */
export class ModelLoader {
    constructor(options = {}) {
        this.cache = options.cache ?? true;
        this.cacheStorage = options.cacheStorage ?? 'ruvector-models';
        this.onProgress = options.onProgress ?? null;
    }

    /**
     * Load a pre-configured model by name
     * @param {string} modelName - Model name from MODELS
     * @returns {Promise<{modelBytes: Uint8Array, tokenizerJson: string, config: object}>}
     */
    async loadModel(modelName = DEFAULT_MODEL) {
        const modelConfig = MODELS[modelName];
        if (!modelConfig) {
            throw new Error(`Unknown model: ${modelName}. Available: ${Object.keys(MODELS).join(', ')}`);
        }

        // In-memory memo: a second load of the same model in this process reuses
        // the already-downloaded bytes instead of re-fetching (issue #523).
        if (this.cache && _inMemoryModelCache.has(modelName)) {
            return _inMemoryModelCache.get(modelName);
        }

        // On-disk cache (Node only): models persist across processes so they are
        // downloaded once, not every run. The browser has the Cache API instead
        // (handled in fetchWithCache). See issue #523.
        if (this.cache) {
            const disk = await this._loadFromDisk(modelName);
            if (disk) {
                const cached = { ...disk, config: modelConfig };
                _inMemoryModelCache.set(modelName, cached);
                return cached;
            }
        }

        console.log(`Loading model: ${modelConfig.name} (${modelConfig.size})`);

        const [modelBytes, tokenizerJson] = await Promise.all([
            this.fetchWithCache(modelConfig.model, `${modelName}-model.onnx`, 'arraybuffer'),
            this.fetchWithCache(modelConfig.tokenizer, `${modelName}-tokenizer.json`, 'text'),
        ]);

        const result = {
            modelBytes: new Uint8Array(modelBytes),
            tokenizerJson,
            config: modelConfig,
        };

        if (this.cache) {
            _inMemoryModelCache.set(modelName, result);
            await this._saveToDisk(modelName, result.modelBytes, tokenizerJson);
        }

        return result;
    }

    /**
     * Resolve the Node on-disk cache dir for a model (null in non-Node envs).
     * Uses dynamic import so this module stays loadable in browsers/bundlers.
     */
    async _diskCacheDir(modelName) {
        if (typeof process === 'undefined' || !process.versions?.node) return null;
        const home = process.env.RUVECTOR_CACHE_DIR
            || process.env.HOME || process.env.USERPROFILE || '/tmp';
        const path = await import('node:path');
        return path.join(home, '.ruvector', 'models', modelName);
    }

    /** Load model bytes + tokenizer from the Node disk cache, or null if absent. */
    async _loadFromDisk(modelName) {
        const dir = await this._diskCacheDir(modelName);
        if (!dir) return null;
        try {
            const fs = await import('node:fs');
            const path = await import('node:path');
            const modelPath = path.join(dir, 'model.onnx');
            const tokPath = path.join(dir, 'tokenizer.json');
            if (!fs.existsSync(modelPath) || !fs.existsSync(tokPath)) return null;
            const modelBytes = new Uint8Array(fs.readFileSync(modelPath));
            const tokenizerJson = fs.readFileSync(tokPath, 'utf8');
            if (modelBytes.length === 0 || tokenizerJson.length === 0) return null;
            console.log(`  Disk cache hit: ${modelName}`);
            return { modelBytes, tokenizerJson };
        } catch {
            return null;
        }
    }

    /** Persist model bytes + tokenizer to the Node disk cache (best-effort). */
    async _saveToDisk(modelName, modelBytes, tokenizerJson) {
        const dir = await this._diskCacheDir(modelName);
        if (!dir) return;
        try {
            const fs = await import('node:fs');
            const path = await import('node:path');
            fs.mkdirSync(dir, { recursive: true });
            // Write to temp files then rename, so a crash mid-write can't leave a
            // truncated cache entry that later reads would trust.
            const mTmp = path.join(dir, 'model.onnx.tmp');
            const tTmp = path.join(dir, 'tokenizer.json.tmp');
            fs.writeFileSync(mTmp, Buffer.from(modelBytes));
            fs.writeFileSync(tTmp, tokenizerJson);
            fs.renameSync(mTmp, path.join(dir, 'model.onnx'));
            fs.renameSync(tTmp, path.join(dir, 'tokenizer.json'));
        } catch {
            // Cache write is best-effort; embedding still works without it.
        }
    }

    /**
     * Load model from custom URLs
     * @param {string} modelUrl - URL to ONNX model
     * @param {string} tokenizerUrl - URL to tokenizer.json
     * @returns {Promise<{modelBytes: Uint8Array, tokenizerJson: string}>}
     */
    async loadFromUrls(modelUrl, tokenizerUrl) {
        const [modelBytes, tokenizerJson] = await Promise.all([
            this.fetchWithCache(modelUrl, null, 'arraybuffer'),
            this.fetchWithCache(tokenizerUrl, null, 'text'),
        ]);

        return {
            modelBytes: new Uint8Array(modelBytes),
            tokenizerJson,
        };
    }

    /**
     * Load model from local files (Node.js)
     * @param {string} modelPath - Path to ONNX model
     * @param {string} tokenizerPath - Path to tokenizer.json
     * @returns {Promise<{modelBytes: Uint8Array, tokenizerJson: string}>}
     */
    async loadFromFiles(modelPath, tokenizerPath) {
        // Node.js environment
        if (typeof process !== 'undefined' && process.versions?.node) {
            const fs = await import('fs/promises');
            const [modelBytes, tokenizerJson] = await Promise.all([
                fs.readFile(modelPath),
                fs.readFile(tokenizerPath, 'utf8'),
            ]);
            return {
                modelBytes: new Uint8Array(modelBytes),
                tokenizerJson,
            };
        }
        throw new Error('loadFromFiles is only available in Node.js');
    }

    /**
     * Fetch with optional caching (uses Cache API in browsers)
     */
    async fetchWithCache(url, cacheKey, responseType) {
        // Try cache first (browser only)
        if (this.cache && typeof caches !== 'undefined' && cacheKey) {
            try {
                const cache = await caches.open(this.cacheStorage);
                const cached = await cache.match(cacheKey);
                if (cached) {
                    console.log(`  Cache hit: ${cacheKey}`);
                    return responseType === 'arraybuffer'
                        ? await cached.arrayBuffer()
                        : await cached.text();
                }
            } catch (e) {
                // Cache API not available, continue with fetch
            }
        }

        // Fetch from network
        console.log(`  Downloading: ${url}`);
        const response = await this.fetchWithProgress(url);

        if (!response.ok) {
            throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
        }

        // Clone for caching
        const responseClone = response.clone();

        // Cache the response (browser only)
        if (this.cache && typeof caches !== 'undefined' && cacheKey) {
            try {
                const cache = await caches.open(this.cacheStorage);
                await cache.put(cacheKey, responseClone);
            } catch (e) {
                // Cache write failed, continue
            }
        }

        return responseType === 'arraybuffer'
            ? await response.arrayBuffer()
            : await response.text();
    }

    /**
     * Fetch with progress reporting
     */
    async fetchWithProgress(url) {
        const response = await fetch(url);

        if (!this.onProgress || !response.body) {
            return response;
        }

        const contentLength = response.headers.get('content-length');
        if (!contentLength) {
            return response;
        }

        const total = parseInt(contentLength, 10);
        let loaded = 0;

        const reader = response.body.getReader();
        const chunks = [];

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            chunks.push(value);
            loaded += value.length;

            this.onProgress({
                loaded,
                total,
                percent: Math.round((loaded / total) * 100),
            });
        }

        const body = new Uint8Array(loaded);
        let position = 0;
        for (const chunk of chunks) {
            body.set(chunk, position);
            position += chunk.length;
        }

        return new Response(body, {
            headers: response.headers,
            status: response.status,
            statusText: response.statusText,
        });
    }

    /**
     * Clear cached models
     */
    async clearCache() {
        if (typeof caches !== 'undefined') {
            await caches.delete(this.cacheStorage);
            console.log('Model cache cleared');
        }
    }

    /**
     * List available models
     */
    static listModels() {
        return Object.entries(MODELS).map(([key, config]) => ({
            id: key,
            ...config,
        }));
    }
}

/**
 * Quick helper to create an embedder with a pre-configured model
 *
 * @example
 * ```javascript
 * import { createEmbedder } from './loader.js';
 *
 * const embedder = await createEmbedder('all-MiniLM-L6-v2');
 * const embedding = embedder.embedOne("Hello world");
 * ```
 */
export async function createEmbedder(modelName = DEFAULT_MODEL, wasmModule = null) {
    // Import WASM module if not provided
    if (!wasmModule) {
        wasmModule = await import('./pkg/ruvector_onnx_embeddings_wasm.js');
        await wasmModule.default();
    }

    const loader = new ModelLoader();
    const { modelBytes, tokenizerJson, config } = await loader.loadModel(modelName);

    const embedderConfig = new wasmModule.WasmEmbedderConfig()
        .setMaxLength(config.maxLength)
        .setNormalize(true)
        .setPooling(0); // Mean pooling

    const embedder = wasmModule.WasmEmbedder.withConfig(
        modelBytes,
        tokenizerJson,
        embedderConfig
    );

    return embedder;
}

/**
 * Quick helper for one-off embedding (loads model, embeds, returns)
 *
 * @example
 * ```javascript
 * import { embed } from './loader.js';
 *
 * const embedding = await embed("Hello world");
 * const embeddings = await embed(["Hello", "World"]);
 * ```
 */
export async function embed(text, modelName = DEFAULT_MODEL) {
    const embedder = await createEmbedder(modelName);

    if (Array.isArray(text)) {
        return embedder.embedBatch(text);
    }
    return embedder.embedOne(text);
}

/**
 * Quick helper for similarity comparison
 *
 * @example
 * ```javascript
 * import { similarity } from './loader.js';
 *
 * const score = await similarity("I love dogs", "I adore puppies");
 * console.log(score); // ~0.85
 * ```
 */
export async function similarity(text1, text2, modelName = DEFAULT_MODEL) {
    const embedder = await createEmbedder(modelName);
    return embedder.similarity(text1, text2);
}

export default {
    MODELS,
    DEFAULT_MODEL,
    ModelLoader,
    createEmbedder,
    embed,
    similarity,
};
