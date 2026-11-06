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
/** File-metadata key holding the registry JSON. */
export declare const FIELD_REGISTRY_KEY = "sdk.rvf.metadata.fields";
export type FieldType = 'f64' | 'string' | 'bool' | 'json';
/** Native `{ fieldId, valueType, value }` entry (see rvf-node RvfMetadataEntry). */
export interface NativeMetadataEntry {
    fieldId: number;
    valueType: string;
    value: string;
}
export declare class FieldRegistry {
    private byName;
    private byId;
    private nextId;
    /** Load from the stored JSON; `null`/empty means no fields yet. */
    static parse(json: string | null | undefined): FieldRegistry;
    get size(): number;
    toJSON(): string;
    /**
     * Encode one metadata object into native entries, registering new fields.
     * Returns whether the registry changed (and so must be persisted before
     * the ingest is committed).
     */
    encodeRecord(metadata: Record<string, RvfMetadataValue>): {
        fields: NativeMetadataEntry[];
        changed: boolean;
    };
    /** Decode native entries back into a name-keyed object. */
    decodeRecord(entries: NativeMetadataEntry[]): Record<string, RvfMetadataValue>;
    /** Resolve a leaf's field to `{ fieldId, type }`. */
    private resolveField;
    /** Serialize a filter to the native JSON form, typing literals by field. */
    filterToNative(expr: RvfFilterExpr): Record<string, unknown>;
}
/**
 * Type a filter literal. With a known field type the literal takes that type
 * (native comparisons only match identical types). Without one (raw fieldId
 * written outside the registry) fall back to inference from the JS value.
 */
export declare function literalToNative(value: RvfFilterValue, type?: FieldType): {
    valueType: string;
    value: string;
};
