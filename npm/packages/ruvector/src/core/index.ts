/**
 * Core module exports
 *
 * These wrappers provide safe, type-flexible interfaces to the underlying
 * native packages, handling array type conversions automatically.
 */

export * from './gnn-wrapper';
export * from './attention-fallbacks';
export * from './agentdb-fast';
export * from './sona-wrapper';
export * from './intelligence-engine';

// Re-export default objects for convenience
export { default as gnnWrapper } from './gnn-wrapper';
export { default as attentionFallbacks } from './attention-fallbacks';
export { default as agentdbFast } from './agentdb-fast';
export { default as Sona } from './sona-wrapper';
export { default as IntelligenceEngine } from './intelligence-engine';
