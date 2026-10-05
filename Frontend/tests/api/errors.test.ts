import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  AbortedError,
  ApiError,
  KNOWN_ERROR_CODES,
  NetworkError,
  isOfflineError,
  parseApiError,
  parseRetryAfter,
  requestIdFor,
  userMessageFor,
} from '../../src/api/errors.ts';

const problem = (over: Record<string, unknown> = {}) =>
  JSON.stringify({
    type: '/errors/image-too-large',
    title: 'Image too large',
    status: 413,
    detail: 'Image exceeds the 10 MB limit.',
    instance: '/api/v1/tree/predict',
    code: 'IMAGE_TOO_LARGE',
    request_id: 'req-123',
    ...over,
  });

test('parseApiError reads a problem+json body', () => {
  const err = parseApiError({ status: 413, body: problem() });
  assert.ok(err instanceof ApiError);
  assert.equal(err.code, 'IMAGE_TOO_LARGE');
  assert.equal(err.status, 413);
  assert.equal(err.title, 'Image too large');
  assert.equal(err.detail, 'Image exceeds the 10 MB limit.');
  assert.equal(err.requestId, 'req-123');
  assert.equal(err.retryAfterSeconds, undefined);
});

test('parseApiError falls back to the X-Request-ID header, and to null', () => {
  const noId = problem({ request_id: null });
  assert.equal(parseApiError({ status: 413, body: noId, requestIdHeader: 'hdr-1' }).requestId, 'hdr-1');
  assert.equal(parseApiError({ status: 413, body: noId }).requestId, null);
});

test('parseApiError exposes validation field errors and drops malformed entries', () => {
  const body = problem({
    status: 422,
    code: 'VALIDATION_ERROR',
    errors: [{ field: 'body.latitude', message: 'must be <= 90' }, { nope: 1 }, 'x'],
  });
  const err = parseApiError({ status: 422, body });
  assert.deepEqual(err.fieldErrors, [{ field: 'body.latitude', message: 'must be <= 90' }]);
});

test('parseApiError reads Retry-After (seconds)', () => {
  const body = problem({ status: 429, code: 'RATE_LIMITED' });
  const err = parseApiError({ status: 429, body, retryAfterHeader: '42' });
  assert.equal(err.code, 'RATE_LIMITED');
  assert.equal(err.retryAfterSeconds, 42);
});

test('parseRetryAfter handles seconds, HTTP dates and junk', () => {
  assert.equal(parseRetryAfter('7'), 7);
  assert.equal(parseRetryAfter(' 0 '), 0);
  const now = Date.parse('2026-10-05T00:00:00Z');
  assert.equal(parseRetryAfter('Mon, 05 Oct 2026 00:00:30 GMT', now), 30);
  assert.equal(parseRetryAfter('Mon, 05 Oct 2026 00:00:00 GMT', now + 60_000), 0); // past date clamps to 0
  assert.equal(parseRetryAfter('soon'), undefined);
  assert.equal(parseRetryAfter(''), undefined);
  assert.equal(parseRetryAfter(null), undefined);
});

test('parseApiError copes with non-JSON proxy / empty responses', () => {
  const html = parseApiError({ status: 502, body: '<html>Bad gateway</html>' });
  assert.equal(html.code, 'SERVER_UNREACHABLE');
  assert.equal(html.status, 502);
  assert.equal(parseApiError({ status: 530, body: '' }).code, 'SERVER_UNREACHABLE');
  assert.equal(parseApiError({ status: 429, body: 'slow down' }).code, 'RATE_LIMITED');
  assert.equal(parseApiError({ status: 413, body: '' }).code, 'IMAGE_TOO_LARGE');
  assert.equal(parseApiError({ status: 404, body: 'nope' }).code, 'NOT_FOUND');
  assert.equal(parseApiError({ status: 418, body: '' }).code, 'HTTP_ERROR');
  // truncated JSON must not throw
  assert.equal(parseApiError({ status: 500, body: '{"code":"INFER' }).code, 'HTTP_ERROR');
  // JSON that is not an object
  assert.equal(parseApiError({ status: 500, body: '[1,2]' }).code, 'HTTP_ERROR');
});

test('userMessageFor has a friendly message for every code the backend can send', () => {
  const backendCodes = [
    'RATE_LIMITED',
    'IMAGE_TOO_LARGE',
    'UNSUPPORTED_MEDIA_TYPE',
    'INVALID_IMAGE',
    'IMAGE_TOO_SMALL',
    'IMAGE_TOO_LARGE_DIMENSIONS',
    'VALIDATION_ERROR',
    'MISSING_CLIENT_ID',
    'INVALID_CLIENT_ID',
    'INFERENCE_FAILED',
    'MODEL_UNAVAILABLE',
    'DATABASE_UNAVAILABLE',
    'ANALYSIS_NOT_FOUND',
    'SAMPLE_IMMUTABLE',
    'INVALID_SIGNATURE',
  ];
  for (const code of backendCodes) {
    assert.ok((KNOWN_ERROR_CODES as readonly string[]).includes(code), `${code} missing from KNOWN_ERROR_CODES`);
  }
  const generic = userMessageFor(new Error('x'));
  for (const code of KNOWN_ERROR_CODES) {
    const message = userMessageFor(new ApiError({ code, status: 400, title: 't', detail: 'd' }));
    assert.ok(message.length > 0 && message.length < 160, `${code}: bad message "${message}"`);
    assert.ok(!message.includes('undefined'), `${code}: "${message}"`);
    if (code !== 'HTTP_ERROR') assert.notEqual(message, generic, `${code} fell through to the generic message`);
  }
});

test('userMessageFor: RATE_LIMITED uses Retry-After when known', () => {
  const base = { code: 'RATE_LIMITED', status: 429, title: 't', detail: 'd' };
  assert.match(userMessageFor(new ApiError({ ...base, retryAfterSeconds: 12 })), /12 seconds/);
  assert.match(userMessageFor(new ApiError({ ...base, retryAfterSeconds: 1 })), /1 second\b/);
  assert.match(userMessageFor(new ApiError(base)), /a minute/);
});

test('userMessageFor: network, timeout, abort and unknown errors', () => {
  assert.match(userMessageFor(new NetworkError('offline')), /Can't reach the KeraAI server/);
  assert.match(userMessageFor(new NetworkError('timeout')), /too long/);
  assert.equal(userMessageFor(new AbortedError()), 'Cancelled.');
  assert.match(
    userMessageFor(new ApiError({ code: 'SOMETHING_NEW', status: 400, title: 't', detail: 'd' })),
    /went wrong/,
  );
  assert.match(userMessageFor('boom'), /went wrong/);
  assert.match(userMessageFor(undefined), /went wrong/);
  // prototype keys must not be mistaken for codes
  assert.match(
    userMessageFor(new ApiError({ code: 'toString', status: 400, title: 't', detail: 'd' })),
    /went wrong/,
  );
});

test('requestIdFor / isOfflineError', () => {
  const withId = new ApiError({ code: 'X', status: 500, title: 't', detail: 'd', requestId: 'r1' });
  assert.equal(requestIdFor(withId), 'r1');
  assert.equal(requestIdFor(new NetworkError()), null);
  assert.equal(requestIdFor('x'), null);
  assert.equal(isOfflineError(new NetworkError('timeout')), true);
  assert.equal(isOfflineError(parseApiError({ status: 502, body: '' })), true);
  assert.equal(isOfflineError(parseApiError({ status: 400, body: problem({ code: 'INVALID_IMAGE' }) })), false);
});
