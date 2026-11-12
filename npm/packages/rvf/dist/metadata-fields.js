"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.FieldRegistry = exports.FIELD_REGISTRY_KEY = void 0;
exports.literalToNative = literalToNative;
const errors_1 = require("./errors");
/** File-metadata key holding the registry JSON. */
exports.FIELD_REGISTRY_KEY = 'sdk.rvf.metadata.fields';
/** Highest native field id (u16). */
const MAX_FIELD_ID = 0xffff;
function typeOf(value) {
    if (value === null)
        return null;
    switch (typeof value) {
        case 'number':
            return 'f64';
        case 'string':
            return 'string';
        case 'boolean':
            return 'bool';
        case 'object':
            return 'json';
        default:
            throw new errors_1.RvfError(errors_1.RvfErrorCode.MetadataTypeConflict, `unsupported metadata value of type ${typeof value}`);
    }
}
function encodeValue(type, value) {
    if (value === null)
        return { valueType: 'null', value: '' };
    switch (type) {
        case 'f64': {
            const n = value;
            if (!Number.isFinite(n)) {
                throw new errors_1.RvfError(errors_1.RvfErrorCode.MetadataTypeConflict, `non-finite number ${n}`);
            }
            return { valueType: 'f64', value: String(n) };
        }
        case 'string':
            return { valueType: 'string', value: value };
        case 'bool':
            return { valueType: 'bool', value: value ? 'true' : 'false' };
        case 'json':
            return { valueType: 'string', value: JSON.stringify(value) };
    }
}
function decodeNative(entry, type) {
    switch (entry.valueType) {
        case 'null':
            return null;
        case 'bool':
            return entry.value === 'true';
        case 'f64':
        case 'u64':
        case 'i64':
            return Number(entry.value);
        case 'string':
            if (type === 'json') {
                try {
                    return JSON.parse(entry.value);
                }
                catch {
                    return entry.value;
                }
            }
            return entry.value;
        default:
            // bytes (hex) and future types are returned in their string form.
            return entry.value;
    }
}
class FieldRegistry {
    constructor() {
        this.byName = new Map();
        this.byId = new Map();
        this.nextId = 0;
    }
    /** Load from the stored JSON; `null`/empty means no fields yet. */
    static parse(json) {
        const reg = new FieldRegistry();
        if (!json)
            return reg;
        let parsed;
        try {
            parsed = JSON.parse(json);
        }
        catch (err) {
            throw new errors_1.RvfError(errors_1.RvfErrorCode.InvalidMetadataRegistry, `${exports.FIELD_REGISTRY_KEY} is not valid JSON: ${err.message}`);
        }
        const fields = parsed?.fields;
        if (!fields || typeof fields !== 'object') {
            throw new errors_1.RvfError(errors_1.RvfErrorCode.InvalidMetadataRegistry, `${exports.FIELD_REGISTRY_KEY} has no fields map`);
        }
        for (const [name, def] of Object.entries(fields)) {
            if (!def ||
                !Number.isInteger(def.id) ||
                def.id < 0 ||
                def.id > MAX_FIELD_ID ||
                !['f64', 'string', 'bool', 'json'].includes(def.type) ||
                reg.byId.has(def.id)) {
                throw new errors_1.RvfError(errors_1.RvfErrorCode.InvalidMetadataRegistry, `bad entry for field "${name}"`);
            }
            reg.byName.set(name, { id: def.id, type: def.type });
            reg.byId.set(def.id, name);
            reg.nextId = Math.max(reg.nextId, def.id + 1);
        }
        return reg;
    }
    get size() {
        return this.byName.size;
    }
    toJSON() {
        return JSON.stringify({ version: 1, fields: Object.fromEntries(this.byName) });
    }
    /**
     * Encode one metadata object into native entries, registering new fields.
     * Returns whether the registry changed (and so must be persisted before
     * the ingest is committed).
     */
    encodeRecord(metadata) {
        let changed = false;
        const fields = [];
        for (const name of Object.keys(metadata)) {
            const value = metadata[name];
            if (value === undefined)
                continue;
            const type = typeOf(value);
            let def = this.byName.get(name);
            if (!def) {
                if (this.nextId > MAX_FIELD_ID) {
                    throw new errors_1.RvfError(errors_1.RvfErrorCode.MetadataTypeConflict, 'too many metadata fields (max 65536)');
                }
                // A field first seen as null is registered as a string field.
                def = { id: this.nextId++, type: type ?? 'string' };
                this.byName.set(name, def);
                this.byId.set(def.id, name);
                changed = true;
            }
            else if (type !== null && type !== def.type) {
                throw new errors_1.RvfError(errors_1.RvfErrorCode.MetadataTypeConflict, `field "${name}" holds ${def.type} values; got ${type}`);
            }
            fields.push({ fieldId: def.id, ...encodeValue(def.type, value) });
        }
        return { fields, changed };
    }
    /** Decode native entries back into a name-keyed object. */
    decodeRecord(entries) {
        const out = Object.create(null);
        for (const entry of entries) {
            const name = this.byId.get(entry.fieldId);
            const type = name !== undefined ? this.byName.get(name)?.type : undefined;
            // Fields written through the raw fieldId API have no name: use the
            // runtime's own schema naming rather than dropping them.
            out[name ?? `field.${entry.fieldId}`] = decodeNative(entry, type);
        }
        return out;
    }
    /** Resolve a leaf's field to `{ fieldId, type }`. */
    resolveField(expr) {
        if (typeof expr.field === 'string') {
            const def = this.byName.get(expr.field);
            // An unknown field matches no record: point it at an id no record uses.
            if (!def)
                return { fieldId: this.nextId <= MAX_FIELD_ID ? this.nextId : MAX_FIELD_ID };
            return { fieldId: def.id, type: def.type };
        }
        if (typeof expr.fieldId === 'number') {
            const name = this.byId.get(expr.fieldId);
            return { fieldId: expr.fieldId, type: name ? this.byName.get(name)?.type : undefined };
        }
        throw new errors_1.RvfError(errors_1.RvfErrorCode.InvalidFilter, 'filter leaf needs a "field" name or a numeric "fieldId"');
    }
    /** Serialize a filter to the native JSON form, typing literals by field. */
    filterToNative(expr) {
        switch (expr.op) {
            case 'and':
                return { op: 'and', children: expr.exprs.map((e) => this.filterToNative(e)) };
            case 'or':
                return { op: 'or', children: expr.exprs.map((e) => this.filterToNative(e)) };
            case 'not':
                return { op: 'not', child: this.filterToNative(expr.expr) };
            case 'in': {
                const { fieldId, type } = this.resolveField(expr);
                const values = expr.values.map((v) => literalToNative(v, type));
                return { op: 'in', fieldId, valueType: values[0]?.valueType ?? 'string', values: values.map((v) => v.value) };
            }
            case 'range': {
                const { fieldId, type } = this.resolveField(expr);
                const lo = literalToNative(expr.low, type);
                const hi = literalToNative(expr.high, type);
                return { op: 'range', fieldId, valueType: lo.valueType, low: lo.value, high: hi.value };
            }
            default: {
                const { fieldId, type } = this.resolveField(expr);
                const { valueType, value } = literalToNative(expr.value, type);
                return { op: expr.op, fieldId, valueType, value };
            }
        }
    }
}
exports.FieldRegistry = FieldRegistry;
/**
 * Type a filter literal. With a known field type the literal takes that type
 * (native comparisons only match identical types). Without one (raw fieldId
 * written outside the registry) fall back to inference from the JS value.
 */
function literalToNative(value, type) {
    if (type === 'f64' && typeof value === 'number')
        return { valueType: 'f64', value: String(value) };
    if ((type === 'string' || type === 'json') && typeof value === 'string')
        return { valueType: 'string', value };
    if (type === 'bool' && typeof value === 'boolean')
        return { valueType: 'bool', value: value ? 'true' : 'false' };
    if (typeof value === 'boolean')
        return { valueType: 'bool', value: value ? 'true' : 'false' };
    if (typeof value === 'string')
        return { valueType: 'string', value };
    if (!Number.isInteger(value))
        return { valueType: 'f64', value: String(value) };
    return { valueType: value >= 0 ? 'u64' : 'i64', value: String(value) };
}
//# sourceMappingURL=metadata-fields.js.map