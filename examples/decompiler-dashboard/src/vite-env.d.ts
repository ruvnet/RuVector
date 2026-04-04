/// <reference types="vite/client" />

declare module 'js-beautify' {
  export function js_beautify(source: string, options?: Record<string, unknown>): string;
  export function css_beautify(source: string, options?: Record<string, unknown>): string;
  export function html_beautify(source: string, options?: Record<string, unknown>): string;
}
