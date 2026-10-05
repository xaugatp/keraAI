import { strict as assert } from 'node:assert';
import { test } from 'node:test';
import { withMinDuration } from '../../src/api/minDuration.ts';

test('waits for the floor even when the promise resolves immediately', async () => {
  const start = Date.now();
  const value = await withMinDuration(Promise.resolve('fast'), 40);
  assert.equal(value, 'fast');
  assert.ok(Date.now() - start >= 35, 'should have waited close to the floor');
});

test('does not add extra delay when the promise is already slower than the floor', async () => {
  const slow = new Promise<string>((resolve) => setTimeout(() => resolve('slow'), 60));
  const start = Date.now();
  const value = await withMinDuration(slow, 10);
  assert.equal(value, 'slow');
  const elapsed = Date.now() - start;
  assert.ok(elapsed >= 55 && elapsed < 150, `expected ~60ms, got ${elapsed}ms`);
});

test('propagates a rejection without waiting for the floor to matter', async () => {
  await assert.rejects(withMinDuration(Promise.reject(new Error('boom')), 20), /boom/);
});
