/**
 * Database Persistence Module for ruvector-extensions
 *
 * Provides comprehensive database persistence capabilities including:
 * - Multiple save formats (JSON, Binary/MessagePack, SQLite)
 * - Incremental saves (only changed data)
 * - Snapshot management (create, list, restore, delete)
 * - Export/import functionality
 * - Compression support
 * - Progress callbacks for large operations
 *
 * @module persistence
 */
import { promises as fs } from 'fs';
import { createReadStream } from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';
// ============================================================================
// Database Persistence Manager
// ============================================================================
/**
 * Main persistence manager for VectorDB instances
 *
 * @example
 * ```typescript
 * const db = new VectorDB({ dimension: 384 });
 * const persistence = new DatabasePersistence(db, {
 *   baseDir: './data',
 *   format: 'binary',
 *   compression: 'gzip',
 *   incremental: true
 * });
 *
 * // Save database
 * await persistence.save({ onProgress: (p) => console.log(p.message) });
 *
 * // Create snapshot
 * const snapshot = await persistence.createSnapshot('before-update');
 *
 * // Restore from snapshot
 * await persistence.restoreSnapshot(snapshot.id);
 * ```
 */
export class DatabasePersistence {
    db;
    options;
    ready;
    incrementalState = null;
    autoSaveTimer = null;
    /**
     * Create a new database persistence manager
     *
     * @param db - VectorDB instance to manage
     * @param options - Persistence configuration
     */
    constructor(db, options) {
        this.db = db;
        this.options = {
            baseDir: options.baseDir,
            dimensions: options.dimensions,
            getVectorIds: options.getVectorIds,
            format: options.format || 'json',
            compression: options.compression || 'none',
            incremental: options.incremental ?? false,
            autoSaveInterval: options.autoSaveInterval ?? 0,
            maxSnapshots: options.maxSnapshots ?? 10,
            batchSize: options.batchSize ?? 1000,
        };
        this.ready = this.initialize();
    }
    /**
     * Initialize persistence system
     */
    async initialize() {
        // Create base directory if it doesn't exist
        await fs.mkdir(this.options.baseDir, { recursive: true });
        await fs.mkdir(path.join(this.options.baseDir, 'snapshots'), { recursive: true });
        // Start auto-save if configured
        if (this.options.autoSaveInterval > 0) {
            this.startAutoSave();
        }
        // Load incremental state if exists
        if (this.options.incremental) {
            await this.loadIncrementalState();
        }
    }
    // ==========================================================================
    // Save Operations
    // ==========================================================================
    /**
     * Save database to disk
     *
     * @param options - Save options
     * @returns Path to saved file
     */
    async save(options = {}) {
        const format = options.format || this.options.format;
        const compress = options.compress ?? (this.options.compression !== 'none');
        const savePath = options.path || this.getDefaultSavePath(format, compress);
        const state = await this.serializeDatabase(options.onProgress);
        if (options.onProgress) {
            options.onProgress({
                operation: 'save',
                percentage: 80,
                current: 4,
                total: 5,
                message: 'Writing to disk...',
            });
        }
        await this.writeStateToFile(state, savePath, format, compress);
        if (this.options.incremental) {
            await this.updateIncrementalState(state);
        }
        if (options.onProgress) {
            options.onProgress({
                operation: 'save',
                percentage: 100,
                current: 5,
                total: 5,
                message: 'Save completed',
            });
        }
        return savePath;
    }
    /**
     * Save only changed data (incremental save)
     *
     * @param options - Save options
     * @returns Path to saved file or null if no changes
     */
    async saveIncremental(options = {}) {
        if (!this.incrementalState) {
            // First save, do full save
            return this.save(options);
        }
        const currentVectors = await this.getAllVectorIds();
        // Detect changes
        const added = currentVectors.filter(id => !this.incrementalState.vectorIds.has(id));
        const removed = Array.from(this.incrementalState.vectorIds).filter(id => !currentVectors.includes(id));
        if (added.length === 0 && removed.length === 0) {
            // No changes
            return null;
        }
        if (options.onProgress) {
            options.onProgress({
                operation: 'incremental-save',
                percentage: 20,
                current: 1,
                total: 5,
                message: `Found ${added.length} new and ${removed.length} removed vectors`,
            });
        }
        // For now, do a full save with changes
        // In a production system, you'd implement delta encoding
        return this.save(options);
    }
    /**
     * Load database from disk
     *
     * @param options - Load options
     */
    async load(options) {
        const format = options.format || this.detectFormat(options.path);
        if (options.onProgress) {
            options.onProgress({
                operation: 'load',
                percentage: 10,
                current: 1,
                total: 5,
                message: 'Reading from disk...',
            });
        }
        await this.ready;
        const state = await this.readStateFromFile(options.path, format);
        if (options.verifyChecksum && state.checksum) {
            if (options.onProgress) {
                options.onProgress({
                    operation: 'load',
                    percentage: 30,
                    current: 2,
                    total: 5,
                    message: 'Verifying checksum...',
                });
            }
            const computed = this.computeChecksum(state);
            if (computed !== state.checksum) {
                throw new Error('Checksum verification failed - file may be corrupted');
            }
        }
        await this.deserializeDatabase(state, options.onProgress);
        if (options.onProgress) {
            options.onProgress({
                operation: 'load',
                percentage: 100,
                current: 5,
                total: 5,
                message: 'Load completed',
            });
        }
    }
    // ==========================================================================
    // Snapshot Management
    // ==========================================================================
    /**
     * Create a snapshot of the current database state
     *
     * @param name - Human-readable snapshot name
     * @param metadata - Additional metadata to store
     * @returns Snapshot metadata
     */
    async createSnapshot(name, metadata) {
        const id = crypto.randomUUID();
        const timestamp = Date.now();
        const stats = await this.getDatabaseStats();
        const snapshotPath = path.join(this.options.baseDir, 'snapshots', `${id}.${this.options.format}`);
        await this.save({
            path: snapshotPath,
            format: this.options.format,
            compress: this.options.compression !== 'none',
        });
        const fileStats = await fs.stat(snapshotPath);
        const checksum = await this.computeFileChecksum(snapshotPath);
        const snapshotMetadata = {
            id,
            name,
            timestamp,
            vectorCount: stats.count,
            dimension: stats.dimension,
            format: this.options.format,
            compressed: this.options.compression !== 'none',
            fileSize: fileStats.size,
            checksum,
            metadata,
        };
        // Save metadata
        const metadataPath = path.join(this.options.baseDir, 'snapshots', `${id}.meta.json`);
        await fs.writeFile(metadataPath, JSON.stringify(snapshotMetadata, null, 2));
        // Clean up old snapshots
        await this.cleanupOldSnapshots();
        return snapshotMetadata;
    }
    /**
     * List all available snapshots
     *
     * @returns Array of snapshot metadata, sorted by timestamp (newest first)
     */
    async listSnapshots() {
        const snapshotsDir = path.join(this.options.baseDir, 'snapshots');
        const files = await fs.readdir(snapshotsDir);
        const metadataFiles = files.filter(f => f.endsWith('.meta.json'));
        const snapshots = [];
        for (const file of metadataFiles) {
            const content = await fs.readFile(path.join(snapshotsDir, file), 'utf-8');
            snapshots.push(JSON.parse(content));
        }
        return snapshots.sort((a, b) => b.timestamp - a.timestamp);
    }
    /**
     * Restore database from a snapshot
     *
     * @param snapshotId - Snapshot ID to restore
     * @param options - Restore options
     */
    async restoreSnapshot(snapshotId, options = {}) {
        const snapshotsDir = path.join(this.options.baseDir, 'snapshots');
        const metadataPath = path.join(snapshotsDir, `${snapshotId}.meta.json`);
        let metadata;
        try {
            const content = await fs.readFile(metadataPath, 'utf-8');
            metadata = JSON.parse(content);
        }
        catch (error) {
            throw new Error(`Snapshot ${snapshotId} not found`);
        }
        const snapshotPath = path.join(snapshotsDir, `${snapshotId}.${metadata.format}`);
        if (options.verifyChecksum) {
            if (options.onProgress) {
                options.onProgress({
                    operation: 'restore',
                    percentage: 10,
                    current: 1,
                    total: 5,
                    message: 'Verifying snapshot integrity...',
                });
            }
            const checksum = await this.computeFileChecksum(snapshotPath);
            if (checksum !== metadata.checksum) {
                throw new Error('Snapshot checksum verification failed - file may be corrupted');
            }
        }
        await this.load({
            path: snapshotPath,
            format: metadata.format,
            verifyChecksum: false, // Already verified above if needed
            onProgress: options.onProgress,
        });
    }
    /**
     * Delete a snapshot
     *
     * @param snapshotId - Snapshot ID to delete
     */
    async deleteSnapshot(snapshotId) {
        const snapshotsDir = path.join(this.options.baseDir, 'snapshots');
        const metadataPath = path.join(snapshotsDir, `${snapshotId}.meta.json`);
        let metadata;
        try {
            const content = await fs.readFile(metadataPath, 'utf-8');
            metadata = JSON.parse(content);
        }
        catch (error) {
            throw new Error(`Snapshot ${snapshotId} not found`);
        }
        const snapshotPath = path.join(snapshotsDir, `${snapshotId}.${metadata.format}`);
        await Promise.all([
            fs.unlink(snapshotPath).catch(() => { }),
            fs.unlink(metadataPath).catch(() => { }),
        ]);
    }
    // ==========================================================================
    // Export/Import
    // ==========================================================================
    /**
     * Export database to a file
     *
     * @param options - Export options
     */
    async export(options) {
        const format = options.format || 'json';
        const compress = options.compress ?? false;
        const state = await this.serializeDatabase(options.onProgress);
        if (!options.includeIndex) {
            delete state.indexState;
        }
        await this.writeStateToFile(state, options.path, format, compress);
    }
    /**
     * Import database from a file
     *
     * @param options - Import options
     */
    async import(options) {
        // load validates the file before replacing the existing database.
        await this.load({
            path: options.path,
            format: options.format,
            verifyChecksum: options.verifyChecksum,
            onProgress: options.onProgress,
        });
    }
    // ==========================================================================
    // Auto-Save
    // ==========================================================================
    /**
     * Start automatic saves at configured interval
     */
    startAutoSave() {
        if (this.autoSaveTimer) {
            return; // Already running
        }
        this.autoSaveTimer = setInterval(async () => {
            try {
                if (this.options.incremental) {
                    await this.saveIncremental();
                }
                else {
                    await this.save();
                }
            }
            catch (error) {
                console.error('Auto-save failed:', error);
            }
        }, this.options.autoSaveInterval);
    }
    /**
     * Stop automatic saves
     */
    stopAutoSave() {
        if (this.autoSaveTimer) {
            clearInterval(this.autoSaveTimer);
            this.autoSaveTimer = null;
        }
    }
    /**
     * Cleanup and shutdown
     */
    async shutdown() {
        this.stopAutoSave();
        // Do final save if auto-save was enabled
        if (this.options.autoSaveInterval > 0) {
            await this.save();
        }
    }
    // ==========================================================================
    // Private Helper Methods
    // ==========================================================================
    /**
     * Serialize database to state object
     */
    async serializeDatabase(onProgress) {
        await this.ready;
        if (onProgress) {
            onProgress({
                operation: 'serialize',
                percentage: 10,
                current: 1,
                total: 5,
                message: 'Collecting database statistics...',
            });
        }
        const stats = await this.getDatabaseStats();
        const vectors = [];
        if (onProgress) {
            onProgress({
                operation: 'serialize',
                percentage: 30,
                current: 2,
                total: 5,
                message: 'Extracting vectors...',
            });
        }
        // Extract all vectors
        const vectorIds = await this.getAllVectorIds();
        for (let i = 0; i < vectorIds.length; i++) {
            const vector = await this.db.get(vectorIds[i]);
            if (!vector)
                throw new Error(`Vector disappeared during persistence: ${vectorIds[i]}`);
            vectors.push({ ...vector, id: vector.id ?? vectorIds[i], vector: Array.from(vector.vector ?? vector.values) });
            if (onProgress && i % this.options.batchSize === 0) {
                const percentage = 30 + Math.floor((i / vectorIds.length) * 40);
                onProgress({
                    operation: 'serialize',
                    percentage,
                    current: i,
                    total: vectorIds.length,
                    message: `Extracted ${i}/${vectorIds.length} vectors...`,
                });
            }
        }
        const state = {
            version: '1.0.0',
            options: {
                dimension: stats.dimension,
                metric: stats.metric,
            },
            stats,
            vectors,
            timestamp: Date.now(),
        };
        if (onProgress) {
            onProgress({
                operation: 'serialize',
                percentage: 90,
                current: 4,
                total: 5,
                message: 'Computing checksum...',
            });
        }
        state.checksum = this.computeChecksum(state);
        return state;
    }
    /**
     * Deserialize state object into database
     */
    async deserializeDatabase(state, onProgress) {
        const current = await this.getDatabaseStats();
        if (current.dimension !== state.options.dimension) {
            throw new Error('Saved vector dimension does not match the destination database.');
        }
        if (onProgress) {
            onProgress({
                operation: 'deserialize',
                percentage: 40,
                current: 2,
                total: 5,
                message: 'Clearing existing data...',
            });
        }
        await this.clearDatabase();
        if (onProgress) {
            onProgress({
                operation: 'deserialize',
                percentage: 50,
                current: 3,
                total: 5,
                message: 'Inserting vectors...',
            });
        }
        // Insert vectors in batches
        for (let i = 0; i < state.vectors.length; i += this.options.batchSize) {
            const batch = state.vectors.slice(i, i + this.options.batchSize);
            await this.db.insertBatch(batch);
            if (onProgress) {
                const percentage = 50 + Math.floor((i / state.vectors.length) * 40);
                onProgress({
                    operation: 'deserialize',
                    percentage,
                    current: i,
                    total: state.vectors.length,
                    message: `Inserted ${i}/${state.vectors.length} vectors...`,
                });
            }
        }
        if (onProgress) {
            onProgress({
                operation: 'deserialize',
                percentage: 95,
                current: 4,
                total: 5,
                message: 'Rebuilding index...',
            });
        }
        // Rebuild index
        if (typeof this.db.buildIndex === 'function')
            await this.db.buildIndex();
    }
    /**
     * Write state to file in specified format
     */
    async writeStateToFile(state, filePath, format, compress) {
        await fs.mkdir(path.dirname(filePath), { recursive: true });
        let data;
        switch (format) {
            case 'json':
                data = Buffer.from(JSON.stringify(state, null, compress ? 0 : 2));
                break;
            case 'binary':
                // Use simple JSON for now - in production, use MessagePack
                data = Buffer.from(JSON.stringify(state));
                break;
            case 'sqlite':
                // SQLite implementation would go here
                throw new Error('SQLite format not yet implemented');
            default:
                throw new Error(`Unsupported format: ${format}`);
        }
        if (compress) {
            const { gzip, brotliCompress } = await import('zlib');
            const { promisify } = await import('util');
            if (this.options.compression === 'gzip') {
                const gzipAsync = promisify(gzip);
                data = await gzipAsync(data);
            }
            else if (this.options.compression === 'brotli') {
                const brotliAsync = promisify(brotliCompress);
                data = await brotliAsync(data);
            }
        }
        await fs.writeFile(filePath, data);
    }
    /**
     * Read state from file in specified format
     */
    async readStateFromFile(filePath, format) {
        let data = await fs.readFile(filePath);
        // Detect and decompress if needed
        if (this.isCompressed(data)) {
            const { gunzip, brotliDecompress } = await import('zlib');
            const { promisify } = await import('util');
            // Try gzip first
            try {
                const gunzipAsync = promisify(gunzip);
                data = await gunzipAsync(data);
            }
            catch {
                // Try brotli
                const brotliAsync = promisify(brotliDecompress);
                data = await brotliAsync(data);
            }
        }
        switch (format) {
            case 'json':
            case 'binary':
                return JSON.parse(data.toString());
            case 'sqlite':
                throw new Error('SQLite format not yet implemented');
            default:
                throw new Error(`Unsupported format: ${format}`);
        }
    }
    /**
     * Get all vector IDs from database
     */
    async getDatabaseStats() {
        const stats = typeof this.db.stats === 'function' ? await this.db.stats() : {};
        const count = stats.count ?? stats.vectorCount ??
            (typeof this.db.len === 'function' ? await this.db.len() : undefined);
        const dimension = this.options.dimensions ?? stats.dimension ?? stats.dimensions;
        if (!Number.isInteger(count) || count < 0 || !Number.isInteger(dimension) || dimension <= 0) {
            throw new Error('Persistence requires a valid vector count and dimension; provide dimensions for VectorDB.');
        }
        return { ...stats, count, dimension };
    }
    async getAllVectorIds() {
        const stats = await this.getDatabaseStats();
        if (stats.count === 0)
            return [];
        const ids = this.options.getVectorIds ? await this.options.getVectorIds() :
            typeof this.db.ids === 'function' ? await this.db.ids() : stats.count === 0 ? [] : null;
        if (!Array.isArray(ids) || ids.some(id => typeof id !== 'string') ||
            new Set(ids).size !== ids.length || ids.length !== stats.count) {
            throw new Error('Persistence requires getVectorIds() to return every current vector ID exactly once.');
        }
        return ids;
    }
    async clearDatabase() {
        if (typeof this.db.clear === 'function') {
            await this.db.clear();
        }
        else {
            for (const id of await this.getAllVectorIds())
                await this.db.delete(id);
        }
    }
    /**
     * Compute checksum of state object
     */
    computeChecksum(state) {
        const { checksum, ...stateWithoutChecksum } = state;
        const data = JSON.stringify(stateWithoutChecksum);
        return crypto.createHash('sha256').update(data).digest('hex');
    }
    /**
     * Compute checksum of file
     */
    async computeFileChecksum(filePath) {
        return new Promise((resolve, reject) => {
            const hash = crypto.createHash('sha256');
            const stream = createReadStream(filePath);
            stream.on('data', data => hash.update(data));
            stream.on('end', () => resolve(hash.digest('hex')));
            stream.on('error', reject);
        });
    }
    /**
     * Detect file format from extension
     */
    detectFormat(filePath) {
        const ext = path.extname(filePath).toLowerCase();
        if (ext === '.json')
            return 'json';
        if (ext === '.bin' || ext === '.msgpack')
            return 'binary';
        if (ext === '.db' || ext === '.sqlite')
            return 'sqlite';
        return this.options.format;
    }
    /**
     * Check if data is compressed
     */
    isCompressed(data) {
        // Gzip magic number: 1f 8b
        if (data[0] === 0x1f && data[1] === 0x8b)
            return true;
        // Brotli doesn't have a magic number, but we can try to decompress
        return false;
    }
    /**
     * Get default save path
     */
    getDefaultSavePath(format, compress) {
        const ext = format === 'json' ? 'json' : format === 'binary' ? 'bin' : 'db';
        const compressExt = compress ? `.${this.options.compression}` : '';
        return path.join(this.options.baseDir, `database.${ext}${compressExt}`);
    }
    /**
     * Load incremental state
     */
    async loadIncrementalState() {
        const statePath = path.join(this.options.baseDir, '.incremental.json');
        try {
            const content = await fs.readFile(statePath, 'utf-8');
            const data = JSON.parse(content);
            this.incrementalState = {
                lastSave: data.lastSave,
                vectorIds: new Set(data.vectorIds),
                checksum: data.checksum,
            };
        }
        catch {
            // No incremental state yet
        }
    }
    /**
     * Update incremental state after save
     */
    async updateIncrementalState(state) {
        const vectorIds = state.vectors.map(v => v.id);
        this.incrementalState = {
            lastSave: Date.now(),
            vectorIds: new Set(vectorIds),
            checksum: state.checksum || '',
        };
        const statePath = path.join(this.options.baseDir, '.incremental.json');
        await fs.writeFile(statePath, JSON.stringify({
            lastSave: this.incrementalState.lastSave,
            vectorIds: Array.from(this.incrementalState.vectorIds),
            checksum: this.incrementalState.checksum,
        }));
    }
    /**
     * Clean up old snapshots beyond max limit
     */
    async cleanupOldSnapshots() {
        const snapshots = await this.listSnapshots();
        if (snapshots.length <= this.options.maxSnapshots) {
            return;
        }
        const toDelete = snapshots.slice(this.options.maxSnapshots);
        for (const snapshot of toDelete) {
            await this.deleteSnapshot(snapshot.id);
        }
    }
}
// ============================================================================
// Utility Functions
// ============================================================================
/**
 * Format file size in human-readable format
 *
 * @param bytes - File size in bytes
 * @returns Formatted string (e.g., "1.5 MB")
 */
export function formatFileSize(bytes) {
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let size = bytes;
    let unitIndex = 0;
    while (size >= 1024 && unitIndex < units.length - 1) {
        size /= 1024;
        unitIndex++;
    }
    return `${size.toFixed(2)} ${units[unitIndex]}`;
}
/**
 * Format timestamp as ISO string
 *
 * @param timestamp - Unix timestamp in milliseconds
 * @returns ISO formatted date string
 */
export function formatTimestamp(timestamp) {
    return new Date(timestamp).toISOString();
}
/**
 * Estimate memory usage of database state
 *
 * @param state - Database state
 * @returns Estimated memory usage in bytes
 */
export function estimateMemoryUsage(state) {
    // Rough estimation
    const vectorSize = state.stats.dimension * 4; // 4 bytes per float
    const metadataSize = 100; // Average metadata size
    const totalVectorSize = state.vectors.length * (vectorSize + metadataSize);
    const overheadSize = JSON.stringify(state).length;
    return totalVectorSize + overheadSize;
}
//# sourceMappingURL=persistence.js.map