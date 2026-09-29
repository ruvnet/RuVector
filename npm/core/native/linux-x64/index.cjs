/**
 * Native binding wrapper for linux-x64
 */

const nativeBinding = require('./ruvector.node');

// The native module exports VectorDb (lowercase 'b') but we want VectorDB
// Also need to add the withDimensions static method since it's not exported properly

class VectorDB {
  constructor(options) {
    // Create internal instance
    this._db = new nativeBinding.VectorDb(options);
  }

  static withDimensions(dimensions) {
    // Factory method - in-memory database with default options. No
    // storagePath: a shared './ruvector.db' made unrelated instances share
    // one store (issue #1063).
    return new VectorDB({
      dimensions: dimensions,
      distanceMetric: 'Cosine'
    });
  }

  async insert(entry) {
    return this._db.insert(entry);
  }

  async insertBatch(entries) {
    return this._db.insertBatch(entries);
  }

  async search(query) {
    return this._db.search(query);
  }

  async delete(id) {
    return this._db.delete(id);
  }

  async get(id) {
    return this._db.get(id);
  }

  async len() {
    return this._db.len();
  }

  async isEmpty() {
    return this._db.isEmpty();
  }
}

module.exports = {
  VectorDB,
  CollectionManager: nativeBinding.CollectionManager,
  version: nativeBinding.version,
  hello: nativeBinding.hello,
  getMetrics: nativeBinding.getMetrics,
  getHealth: nativeBinding.getHealth,
  DistanceMetric: nativeBinding.JsDistanceMetric
};
