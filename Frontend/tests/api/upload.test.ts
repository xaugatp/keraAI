import { test } from 'node:test';
import assert from 'node:assert/strict';
import { UUID_V4, mockConfig, stubGlobal } from './helpers.ts';

mockConfig('https://api.example.test');
const { buildPredictFormData, predict } = await import('../../src/api/upload.ts');
const { ApiError, NetworkError, AbortedError } = await import('../../src/api/errors.ts');
type Progress = import('../../src/api/upload.ts').UploadProgress;

// --- FormData building ---------------------------------------------------------------------

const photo = () => new File([new Uint8Array([1, 2, 3])], 'leaf.jpg', { type: 'image/jpeg' });
const keys = (form: FormData) => [...form.keys()].sort();

test('FormData: image + source only when nothing else is known', () => {
  const form = buildPredictFormData({ file: photo(), source: 'upload' });
  assert.deepEqual(keys(form), ['image', 'source']);
  assert.equal(form.get('source'), 'upload');
  assert.equal((form.get('image') as File).name, 'leaf.jpg');
});

test('FormData: latitude and longitude are sent together, with accuracy and capture time', () => {
  const form = buildPredictFormData({
    file: photo(),
    source: 'camera',
    latitude: 27.71724,
    longitude: -85.32402,
    accuracyM: 4.2,
    capturedAt: new Date('2026-10-01T01:12:15Z'),
  });
  assert.deepEqual(keys(form), ['captured_at', 'gps_accuracy_m', 'image', 'latitude', 'longitude', 'source']);
  assert.equal(form.get('latitude'), '27.71724');
  assert.equal(form.get('longitude'), '-85.32402'); // sign preserved
  assert.equal(form.get('gps_accuracy_m'), '4.2');
  assert.equal(form.get('captured_at'), '2026-10-01T01:12:15.000Z');
  assert.equal(form.get('source'), 'camera');
});

test('FormData: a lone latitude or longitude is dropped (never half a pair)', () => {
  for (const partial of [
    { latitude: 10 },
    { longitude: 10 },
    { latitude: 10, longitude: null },
    { latitude: null, longitude: 10 },
    { latitude: 10, longitude: undefined, accuracyM: 5 },
  ]) {
    const form = buildPredictFormData({ file: photo(), source: 'upload', ...partial });
    assert.equal(form.has('latitude'), false, JSON.stringify(partial));
    assert.equal(form.has('longitude'), false, JSON.stringify(partial));
    assert.equal(form.has('gps_accuracy_m'), false, 'accuracy without coordinates is meaningless');
  }
});

test('FormData: out-of-range or non-finite coordinates are dropped as a pair', () => {
  for (const [latitude, longitude] of [[91, 0], [-91, 0], [0, 181], [0, -181], [NaN, 0], [0, Infinity]]) {
    const form = buildPredictFormData({ file: photo(), source: 'upload', latitude, longitude });
    assert.equal(form.has('latitude'), false, `${latitude},${longitude}`);
    assert.equal(form.has('longitude'), false, `${latitude},${longitude}`);
  }
  // the boundary values are valid, and 0 is a real coordinate (not "missing")
  const edge = buildPredictFormData({ file: photo(), source: 'upload', latitude: 0, longitude: 0 });
  assert.equal(edge.get('latitude'), '0');
  assert.equal(edge.get('longitude'), '0');
  const max = buildPredictFormData({ file: photo(), source: 'upload', latitude: 90, longitude: -180 });
  assert.equal(max.get('latitude'), '90');
});

test('FormData: negative accuracy and an invalid capture time are omitted', () => {
  const form = buildPredictFormData({
    file: photo(),
    source: 'upload',
    latitude: 1,
    longitude: 2,
    accuracyM: -3,
    capturedAt: 'garbage',
  });
  assert.equal(form.has('gps_accuracy_m'), false);
  assert.equal(form.has('captured_at'), false);
});

test('FormData: a capture time can be sent without coordinates; a bare Blob gets a filename', () => {
  const form = buildPredictFormData({
    file: new Blob([new Uint8Array([9])], { type: 'image/jpeg' }),
    source: 'camera',
    capturedAt: '2026-10-01T01:12:15Z',
  });
  assert.deepEqual(keys(form), ['captured_at', 'image', 'source']);
  assert.equal((form.get('image') as File).name, 'photo.jpg');
});

// --- predict() over a fake XMLHttpRequest --------------------------------------------------

class FakeXHR {
  static last: FakeXHR;
  method = '';
  url = '';
  headers: Record<string, string> = {};
  upload: { onprogress: ((e: unknown) => void) | null; onload: (() => void) | null } = {
    onprogress: null,
    onload: null,
  };
  responseType = '';
  timeout = 0;
  status = 0;
  responseText = '';
  responseHeaders: Record<string, string> = {};
  sent: unknown;
  aborted = false;
  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
  ontimeout: (() => void) | null = null;
  onabort: (() => void) | null = null;
  constructor() {
    FakeXHR.last = this;
  }
  open(method: string, url: string) {
    this.method = method;
    this.url = url;
  }
  setRequestHeader(k: string, v: string) {
    this.headers[k.toLowerCase()] = v;
  }
  getResponseHeader(k: string) {
    return this.responseHeaders[k.toLowerCase()] ?? null;
  }
  send(body: unknown) {
    this.sent = body;
  }
  abort() {
    this.aborted = true;
    this.onabort?.();
  }
}

async function withFakeXhr(run: () => Promise<void>) {
  const restore = stubGlobal('XMLHttpRequest', FakeXHR);
  try {
    await run();
  } finally {
    restore();
  }
}

const ANALYSIS = { id: 'a1', model_key: 'tree_classification', status: 'completed' };

test('predict: posts to the tree route with X-Client-Id, no Content-Type, and resolves the 201 body', () =>
  withFakeXhr(async () => {
    const phases: Progress[] = [];
    const promise = predict(
      'tree_classification',
      { file: photo(), source: 'upload', latitude: 1, longitude: 2 },
      { onProgress: (p) => phases.push(p) },
    );
    const xhr = FakeXHR.last;
    assert.equal(xhr.method, 'POST');
    assert.equal(xhr.url, 'https://api.example.test/api/v1/tree/predict');
    assert.match(xhr.headers['x-client-id'], UUID_V4);
    assert.equal('content-type' in xhr.headers, false, 'Content-Type must be left to the browser (multipart boundary)');
    assert.ok(xhr.sent instanceof FormData);
    assert.equal((xhr.sent as FormData).get('latitude'), '1');

    xhr.upload.onprogress?.({ lengthComputable: true, loaded: 25, total: 100 });
    xhr.upload.onprogress?.({ lengthComputable: true, loaded: 100, total: 100 });
    xhr.upload.onprogress?.({ lengthComputable: false, loaded: 0, total: 0 }); // ignored
    xhr.upload.onload?.();
    xhr.upload.onload?.(); // phase reported once
    xhr.upload.onprogress?.({ lengthComputable: true, loaded: 100, total: 100 }); // ignored after analysing
    xhr.status = 201;
    xhr.responseText = JSON.stringify(ANALYSIS);
    xhr.onload?.();

    assert.deepEqual(await promise, ANALYSIS);
    assert.deepEqual(phases, [
      { phase: 'uploading', percent: 0 },
      { phase: 'uploading', percent: 25 },
      { phase: 'uploading', percent: 100 },
      { phase: 'analysing' },
    ]);
  }));

test('predict: leaf segmentation uses its own route', () =>
  withFakeXhr(async () => {
    const promise = predict('leaf_segmentation', { file: photo(), source: 'camera' });
    assert.equal(FakeXHR.last.url, 'https://api.example.test/api/v1/leaf-segmentation/predict');
    FakeXHR.last.status = 201;
    FakeXHR.last.responseText = JSON.stringify(ANALYSIS);
    FakeXHR.last.onload?.();
    await promise;
  }));

test('predict: a problem+json error becomes an ApiError with code and request id', () =>
  withFakeXhr(async () => {
    const promise = predict('tree_classification', { file: photo(), source: 'upload' });
    const xhr = FakeXHR.last;
    xhr.status = 429;
    xhr.responseText = JSON.stringify({
      type: '/errors/rate-limited',
      title: 'Too many requests',
      status: 429,
      detail: 'Rate limit exceeded: 10 per 1 minute',
      instance: '/api/v1/tree/predict',
      code: 'RATE_LIMITED',
      request_id: 'req-9',
    });
    xhr.responseHeaders['retry-after'] = '30';
    xhr.onload?.();
    await assert.rejects(promise, (err: unknown) => {
      assert.ok(err instanceof ApiError);
      assert.equal(err.code, 'RATE_LIMITED');
      assert.equal(err.status, 429);
      assert.equal(err.requestId, 'req-9');
      assert.equal(err.retryAfterSeconds, 30);
      return true;
    });
  }));

test('predict: a non-JSON error page maps to a status-derived code', () =>
  withFakeXhr(async () => {
    const promise = predict('tree_classification', { file: photo(), source: 'upload' });
    FakeXHR.last.status = 502;
    FakeXHR.last.responseText = '<html>Bad gateway</html>';
    FakeXHR.last.onload?.();
    await assert.rejects(promise, (err: unknown) => err instanceof ApiError && err.code === 'SERVER_UNREACHABLE');
  }));

test('predict: a 201 whose body is not JSON is an INVALID_RESPONSE error', () =>
  withFakeXhr(async () => {
    const promise = predict('tree_classification', { file: photo(), source: 'upload' });
    FakeXHR.last.status = 201;
    FakeXHR.last.responseText = '<html>captive portal</html>';
    FakeXHR.last.onload?.();
    await assert.rejects(promise, (err: unknown) => err instanceof ApiError && err.code === 'INVALID_RESPONSE');
  }));

test('predict: network failure and timeout become NetworkError', () =>
  withFakeXhr(async () => {
    const offline = predict('tree_classification', { file: photo(), source: 'upload' });
    FakeXHR.last.onerror?.();
    await assert.rejects(offline, (err: unknown) => err instanceof NetworkError && err.reason === 'offline');

    const slow = predict('tree_classification', { file: photo(), source: 'upload' }, { timeoutMs: 5000 });
    assert.equal(FakeXHR.last.timeout, 5000);
    FakeXHR.last.ontimeout?.();
    await assert.rejects(slow, (err: unknown) => err instanceof NetworkError && err.reason === 'timeout');
  }));

test('predict: aborting the signal cancels the XHR and rejects with AbortedError', () =>
  withFakeXhr(async () => {
    const controller = new AbortController();
    const promise = predict('tree_classification', { file: photo(), source: 'upload' }, { signal: controller.signal });
    controller.abort();
    assert.equal(FakeXHR.last.aborted, true);
    await assert.rejects(promise, (err: unknown) => err instanceof AbortedError);

    // already-aborted signal: never even opens a request
    const before = FakeXHR.last;
    await assert.rejects(
      predict('tree_classification', { file: photo(), source: 'upload' }, { signal: controller.signal }),
      (err: unknown) => err instanceof AbortedError,
    );
    assert.equal(FakeXHR.last, before);
  }));

test('predict: a model without a route is rejected', async () => {
  await assert.rejects(
    predict('leaf_disease' as never, { file: photo(), source: 'upload' }),
    /no predict route/,
  );
});
