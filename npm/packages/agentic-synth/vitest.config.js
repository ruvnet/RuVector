import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html', 'lcov'],
      exclude: [
        'node_modules/',
        'tests/',
        'dist/',
        '**/*.config.js',
        '**/*.d.ts'
      ],
      lines: 90,
      functions: 90,
      branches: 85,
      statements: 90
    },
    testTimeout: 10000,
    hookTimeout: 10000,
    teardownTimeout: 10000,
    mockReset: true,
    restoreMocks: true,
    clearMocks: true
  }
});
