import { test } from 'node:test';
import assert from 'node:assert/strict';
import { DEV_FALLBACK_BASE_URL, resolveApiBaseUrl } from '../../src/api/baseUrl.ts';

test('trims whitespace and trailing slashes', () => {
  assert.equal(resolveApiBaseUrl('https://api.example.com/', false), 'https://api.example.com');
  assert.equal(resolveApiBaseUrl('  https://api.example.com///  ', false), 'https://api.example.com');
  assert.equal(resolveApiBaseUrl('http://127.0.0.1:8000', true), 'http://127.0.0.1:8000');
});

test('unset value: dev falls back to the local backend', () => {
  assert.equal(resolveApiBaseUrl(undefined, true), DEV_FALLBACK_BASE_URL);
  assert.equal(resolveApiBaseUrl('   ', true), DEV_FALLBACK_BASE_URL);
  assert.equal(DEV_FALLBACK_BASE_URL, 'http://127.0.0.1:8000');
});

test('unset value: a production build fails loudly', () => {
  assert.throws(() => resolveApiBaseUrl(undefined, false), /VITE_API_BASE_URL is not set/);
  assert.throws(() => resolveApiBaseUrl('', false), /VITE_API_BASE_URL is not set/);
});

test('rejects values that are not absolute http(s) URLs', () => {
  assert.throws(() => resolveApiBaseUrl('api.example.com', false), /absolute http\(s\) URL/);
  assert.throws(() => resolveApiBaseUrl('ftp://api.example.com', true), /http or https/);
});
