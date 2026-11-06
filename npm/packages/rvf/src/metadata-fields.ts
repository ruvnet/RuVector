/**
 * Field-name registry for per-vector metadata (issue #704).
 *
 * The native store keys metadata by a numeric `fieldId` (u16) with a fixed
 * value type per field. The SDK exposes `Record<string, …>` metadata, so it
 * keeps a name → `{ id, type }` registry. The registry is stored in the RVF
 * file itself, as file-level metadata under {@link FIELD_REGISTRY_KEY}, so it
 * travels with the file through reopen, `derive` and `branch`.
 *
 * Each field's type is fixed the first time the field is written:
 *  - numbers → `f64` (JS numbers are doubles; this keeps `5` and `5.5` in one
 *    field and makes integer-valued filter literals match),
 *  - strings → `string`, booleans → `bool`,
 *  - objects and arrays → `json` (stored as a JSON string, decoded on read).
 * `null` is allowed in any field. Writing a value of a different type to an
 * existing field throws `MetadataTypeConflict` instead of dropping it.
 */
import type { RvfFilterExpr, RvfFilterValue, RvfMetadataValue } from './types';
import { RvfError, RvfErrorCode } from './errors';

/** File-metadata key holding the registry JSON. */
export const FIELD_REGISTRY_KEY = 'sdk.rvf.metadata.fields';

/** Highest native field id (u16). */
const MAX_FIELD_ID = 0xffff;

export type FieldType = 'f64' | 'string' | 'bool' | 'json';

interface FieldDef {
  id: number;
  type: FieldType;
}

/** Native `{ fieldId, valueType, value }` entry (see rvf-node RvfMetadataEntry). */
export interface NativeMetadataEntry {
  fieldId: number;
  valueType: string;
  value: string;
}

function typeOf(value: RvfMetadataValue): FieldType | null {
  if (value === null) return null;
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
      throw new RvfError(
        RvfErrorCode.MetadataTypeConflict,
        `unsupported metadata value of type ${typeof value}`,
      );
  }
}

function encodeValue(type: FieldType, value: RvfMetadataValue): { valueType: string; value: string } {
  if (value === null) return { valueType: 'null', value: '' };
  switch (type) {
    case 'f64': {
      const n = value as number;
      if (!Number.isFinite(n)) {
        throw new RvfError(RvfErrorCode.MetadataTypeConflict, `non-finite number ${n}`);
      }
      return { valueType: 'f64', value: String(n) };
    }
    case 'string':
      return { valueType: 'string', value: value as string };
    case 'bool':
      return { valueType: 'bool', value: value ? 'true' : 'false' };
    case 'json':
      return { valueType: 'string', value: JSON.stringify(value) };
  }
}

function decodeNative(entry: NativeMetadataEntry, type: FieldType | undefined): RvfMetadataValue {
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
          return JSON.parse(entry.value) as RvfMetadataValue;
        } catch {
          return entry.value;
        }
      }
      return entry.value;
    default:
      // bytes (hex) and future types are returned in their string form.
      return entry.value;
  }
}

export class FieldRegistry {
  private byName = new Map<string, FieldDef>();
  private byId = new Map<number, string>();
  private nextId = 0;

  /** Load from the stored JSON; `null`/empty means no fields yet. */
  static parse(json: string | null | undefined): FieldRegistry {
    const reg = new FieldRegistry();
    if (!json) return reg;
    let parsed: unknown;
    try {
      parsed = JSON.parse(json);
    } catch (err) {
      throw new RvfError(
        RvfErrorCode.InvalidMetadataRegistry,
        `${FIELD_REGISTRY_KEY} is not valid JSON: ${(err as Error).message}`,
      );
    }
    const fields = (parsed as { fields?: unknown })?.fields;
    if (!fields || typeof fields !== 'object') {
      throw new RvfError(RvfErrorCode.InvalidMetadataRegistry, `${FIELD_REGISTRY_KEY} has no fields map`);
    }
    for (const [name, def] of Object.entries(fields as Record<string, FieldDef>)) {
      if (
        !def ||
        !Number.isInteger(def.id) ||
        def.id < 0 ||
        def.id > MAX_FIELD_ID ||
        !['f64', 'string', 'bool', 'json'].includes(def.type) ||
        reg.byId.has(def.id)
      ) {
        throw new RvfError(RvfErrorCode.InvalidMetadataRegistry, `bad entry for field "${name}"`);
      }
      reg.byName.set(name, { id: def.id, type: def.type });
      reg.byId.set(def.id, name);
      reg.nextId = Math.max(reg.nextId, def.id + 1);
    }
    return reg;
  }

  get size(): number {
    return this.byName.size;
  }

  toJSON(): string {
    return JSON.stringify({ version: 1, fields: Object.fromEntries(this.byName) });
  }

  /**
   * Encode one metadata object into native entries, registering new fields.
   * Returns whether the registry changed (and so must be persisted before
   * the ingest is committed).
   */
  encodeRecord(metadata: Record<string, RvfMetadataValue>): { fields: NativeMetadataEntry[]; changed: boolean } {
    let changed = false;
    const fields: NativeMetadataEntry[] = [];
    for (const name of Object.keys(metadata)) {
      const value = metadata[name];
      if (value === undefined) continue;
      const type = typeOf(value);
      let def = this.byName.get(name);
      if (!def) {
        if (this.nextId > MAX_FIELD_ID) {
          throw new RvfError(RvfErrorCode.MetadataTypeConflict, 'too many metadata fields (max 65536)');
        }
        // A field first seen as null is registered as a string field.
        def = { id: this.nextId++, type: type ?? 'string' };
        this.byName.set(name, def);
        this.byId.set(def.id, name);
        changed = true;
      } else if (type !== null && type !== def.type) {
        throw new RvfError(
          RvfErrorCode.MetadataTypeConflict,
          `field "${name}" holds ${def.type} values; got ${type}`,
        );
      }
      fields.push({ fieldId: def.id, ...encodeValue(def.type, value) });
    }
    return { fields, changed };
  }

  /** Decode native entries back into a name-keyed object. */
  decodeRecord(entries: NativeMetadataEntry[]): Record<string, RvfMetadataValue> {
    const out: Record<string, RvfMetadataValue> = Object.create(null);
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
  private resolveField(expr: { field?: string; fieldId?: number }): { fieldId: number; type?: FieldType } {
    if (typeof expr.field === 'string') {
      const def = this.byName.get(expr.field);
      // An unknown field matches no record: point it at an id no record uses.
      if (!def) return { fieldId: this.nextId <= MAX_FIELD_ID ? this.nextId : MAX_FIELD_ID };
      return { fieldId: def.id, type: def.type };
    }
    if (typeof expr.fieldId === 'number') {
      const name = this.byId.get(expr.fieldId);
      return { fieldId: expr.fieldId, type: name ? this.byName.get(name)?.type : undefined };
    }
    throw new RvfError(RvfErrorCode.InvalidFilter, 'filter leaf needs a "field" name or a numeric "fieldId"');
  }

  /** Serialize a filter to the native JSON form, typing literals by field. */
  filterToNative(expr: RvfFilterExpr): Record<string, unknown> {
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

/**
 * Type a filter literal. With a known field type the literal takes that type
 * (native comparisons only match identical types). Without one (raw fieldId
 * written outside the registry) fall back to inference from the JS value.
 */
export function literalToNative(value: RvfFilterValue, type?: FieldType): { valueType: string; value: string } {
  if (type === 'f64' && typeof value === 'number') return { valueType: 'f64', value: String(value) };
  if ((type === 'string' || type === 'json') && typeof value === 'string') return { valueType: 'string', value };
  if (type === 'bool' && typeof value === 'boolean') return { valueType: 'bool', value: value ? 'true' : 'false' };
  if (typeof value === 'boolean') return { valueType: 'bool', value: value ? 'true' : 'false' };
  if (typeof value === 'string') return { valueType: 'string', value };
  if (!Number.isInteger(value)) return { valueType: 'f64', value: String(value) };
  return { valueType: value >= 0 ? 'u64' : 'i64', value: String(value) };
}
