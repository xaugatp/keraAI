import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mockConfig } from './helpers.ts';

mockConfig('https://api.example.test');
const { absoluteImageUrl } = await import('../../src/api/urls.ts');

test('prefixes the API origin onto relative image URLs', () => {
  const rel = '/api/v1/analyses/abc/image?variant=original&exp=1790000000&sig=XyZ-_';
  assert.equal(absoluteImageUrl(rel), `https://api.example.test${rel}`);
});

test('tolerates a missing leading slash', () => {
  assert.equal(absoluteImageUrl('api/v1/x'), 'https://api.example.test/api/v1/x');
});

test('null / undefined pass through as null', () => {
  assert.equal(absoluteImageUrl(null), null);
  assert.equal(absoluteImageUrl(undefined), null);
});

test('already-absolute URLs are left alone', () => {
  assert.equal(absoluteImageUrl('https://cdn.example.com/a.png'), 'https://cdn.example.com/a.png');
});
