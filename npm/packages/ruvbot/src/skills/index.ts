/**
 * Skills module exports
 *
 * Provides extensible skill system with hot-reload support.
 */

export { SkillEntity } from '../core/entities/Skill.js';
export type { SkillExecutor, SkillOptions } from '../core/entities/Skill.js';
export type {
  SkillDefinition,
  SkillInput,
  SkillOutput,
  SkillContext,
  SkillResult,
  SkillExample,
} from '../core/types.js';

// Placeholder for skill registry - to be implemented
export const SKILLS_MODULE_VERSION = '0.1.0';

export interface SkillRegistryOptions {
  builtinSkills?: string[];
  customSkillsDir?: string;
  hotReload?: boolean;
}
