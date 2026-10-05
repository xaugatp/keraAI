import { test } from 'node:test';
import assert from 'node:assert/strict';
import { resolveCapturedAt } from '../../src/api/capturedAt.ts';

test('a camera shutter timestamp wins over a GPS fix time', () => {
  const result = resolveCapturedAt({
    cameraCapturedAt: '2026-10-05T10:00:00.000Z',
    gpsCapturedAt: new Date('2026-10-05T09:55:00.000Z'), // an older, cached GPS fix
  });
  assert.equal(result, '2026-10-05T10:00:00.000Z');
});

test('falls back to the GPS fix time when there is no camera capture (e.g. an upload)', () => {
  const result = resolveCapturedAt({ cameraCapturedAt: null, gpsCapturedAt: new Date('2026-10-05T09:55:00.000Z') });
  assert.equal(result, '2026-10-05T09:55:00.000Z');
});

test('resolves to null when neither is known', () => {
  assert.equal(resolveCapturedAt({}), null);
  assert.equal(resolveCapturedAt({ cameraCapturedAt: null, gpsCapturedAt: null }), null);
});

test('an invalid camera timestamp is ignored, falling back to the GPS fix', () => {
  const result = resolveCapturedAt({
    cameraCapturedAt: 'not-a-date',
    gpsCapturedAt: new Date('2026-10-05T09:55:00.000Z'),
  });
  assert.equal(result, '2026-10-05T09:55:00.000Z');
});

test('accepts a Date for the camera timestamp too', () => {
  const result = resolveCapturedAt({ cameraCapturedAt: new Date('2026-10-05T10:00:00.000Z'), gpsCapturedAt: null });
  assert.equal(result, '2026-10-05T10:00:00.000Z');
});
