/**
 * Shared helpers for the Node unit tests (`npm run test:api`, node:test + built-in module mocks;
 * no test framework is installed).
 */
import { mock } from 'node:test';

/**
 * Replace `src/api/config.ts` (which reads Vite's `import.meta.env`, absent in Node) with fixed
 * values. Call BEFORE dynamically importing a module that imports config.
 * `exports` is the current option name; @types/node 22 still only types the deprecated
 * `namedExports`, hence the cast.
 */
export function mockConfig(baseUrl = 'https://api.example.test'): void {
  mock.module('../../src/api/config.ts', {
    exports: { API_BASE_URL: baseUrl, API_PREFIX: '/api/v1' },
  } as never);
}

/** Replace a global (e.g. `fetch`, `XMLHttpRequest`, `localStorage`) and return a restore function. */
export function stubGlobal(name: string, value: unknown): () => void {
  const original = Object.getOwnPropertyDescriptor(globalThis, name);
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
  return () => {
    if (original) Object.defineProperty(globalThis, name, original);
    else delete (globalThis as Record<string, unknown>)[name];
  };
}

export const UUID_V4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
