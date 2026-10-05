import { test } from 'node:test';
import assert from 'node:assert/strict';
import { UUID_V4, mockConfig, stubGlobal } from './helpers.ts';

mockConfig('https://api.example.test');
const { apiFetch, buildUrl } = await import('../../src/api/client.ts');
const { listAnalyses, getAnalysis, deleteAnalysis } = await import('../../src/api/analyses.ts');
const { getReady } = await import('../../src/api/health.ts');
const { ApiError, NetworkError, AbortedError } = await import('../../src/api/errors.ts');

interface Call {
  url: string;
  init: RequestInit;
}

function stubFetch(respond: (call: Call) => Response | Promise<Response>) {
  const calls: Call[] = [];
  const restore = stubGlobal('fetch', async (url: string, init: RequestInit) => {
    const call = { url, init };
    calls.push(call);
    return respond(call);
  });
  return { calls, restore };
}

const json = (body: unknown, status = 200, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json', ...headers } });

test('buildUrl skips null/undefined query values', () => {
  assert.equal(buildUrl('/x'), 'https://api.example.test/x');
  assert.equal(
    buildUrl('/x', { a: 1, b: undefined, c: null, d: 'e f', g: false }),
    'https://api.example.test/x?a=1&d=e+f&g=false',
  );
});

test('apiFetch sends a valid X-Client-Id and parses JSON', async () => {
  const { calls, restore } = stubFetch(() => json({ hello: 'world' }));
  try {
    assert.deepEqual(await apiFetch('/api/v1/models'), { hello: 'world' });
    const headers = calls[0].init.headers as Record<string, string>;
    assert.match(headers['X-Client-Id'], UUID_V4);
    assert.equal(headers['Content-Type'], undefined);
  } finally {
    restore();
  }
});

test('apiFetch can omit X-Client-Id', async () => {
  const { calls, restore } = stubFetch(() => json([]));
  try {
    await apiFetch('/api/v1/models', { withClientId: false });
    assert.equal('X-Client-Id' in (calls[0].init.headers as Record<string, string>), false);
  } finally {
    restore();
  }
});

test('apiFetch never sets Content-Type for FormData, and sets JSON for plain objects', async () => {
  const { calls, restore } = stubFetch(() => json({}));
  try {
    const form = new FormData();
    form.append('a', 'b');
    await apiFetch('/x', { method: 'POST', body: form });
    assert.equal((calls[0].init.headers as Record<string, string>)['Content-Type'], undefined);
    assert.equal(calls[0].init.body, form);

    await apiFetch('/x', { method: 'POST', body: { k: 1 } });
    assert.equal((calls[1].init.headers as Record<string, string>)['Content-Type'], 'application/json');
    assert.equal(calls[1].init.body, '{"k":1}');
  } finally {
    restore();
  }
});

test('apiFetch maps an error response to ApiError (problem+json, request id, Retry-After)', async () => {
  const { restore } = stubFetch(() =>
    json(
      { title: 'Too many requests', status: 429, detail: 'slow', instance: '/x', code: 'RATE_LIMITED', request_id: 'rid-1' },
      429,
      { 'Content-Type': 'application/problem+json', 'Retry-After': '9' },
    ),
  );
  try {
    await assert.rejects(apiFetch('/x'), (err: unknown) => {
      assert.ok(err instanceof ApiError);
      assert.equal(err.code, 'RATE_LIMITED');
      assert.equal(err.requestId, 'rid-1');
      assert.equal(err.retryAfterSeconds, 9);
      return true;
    });
  } finally {
    restore();
  }
});

test('apiFetch: 204 resolves undefined; a non-JSON 200 is INVALID_RESPONSE', async () => {
  let respond: () => Response = () => new Response(null, { status: 204 });
  const { restore } = stubFetch(() => respond());
  try {
    assert.equal(await apiFetch('/x', { method: 'DELETE' }), undefined);
    respond = () => new Response('<html>portal</html>', { status: 200 });
    await assert.rejects(apiFetch('/x'), (err: unknown) => err instanceof ApiError && err.code === 'INVALID_RESPONSE');
  } finally {
    restore();
  }
});

test('apiFetch: a rejected fetch is a NetworkError; abort is an AbortedError', async () => {
  const restoreOffline = stubGlobal('fetch', async () => {
    throw new TypeError('Failed to fetch');
  });
  try {
    await assert.rejects(apiFetch('/x'), (err: unknown) => err instanceof NetworkError && err.reason === 'offline');
  } finally {
    restoreOffline();
  }

  const restoreHang = stubGlobal(
    'fetch',
    (_url: string, init: RequestInit) =>
      new Promise((_resolve, reject) => {
        init.signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')));
      }),
  );
  try {
    const controller = new AbortController();
    const pending = apiFetch('/x', { signal: controller.signal });
    controller.abort();
    await assert.rejects(pending, (err: unknown) => err instanceof AbortedError);

    await assert.rejects(
      apiFetch('/x', { timeoutMs: 20 }),
      (err: unknown) => err instanceof NetworkError && err.reason === 'timeout',
    );
    await assert.rejects(apiFetch('/x', { signal: controller.signal }), (err: unknown) => err instanceof AbortedError);
  } finally {
    restoreHang();
  }
});

test('analyses endpoints build the right requests', async () => {
  const { calls, restore } = stubFetch(({ init }) =>
    init.method === 'DELETE' ? new Response(null, { status: 204 }) : json({ items: [], page: 1, page_size: 20, total: 0, total_pages: 0 }),
  );
  try {
    await listAnalyses({ modelKey: 'tree_classification', scope: 'samples', page: 2, pageSize: 50 });
    assert.equal(
      calls[0].url,
      'https://api.example.test/api/v1/analyses?model_key=tree_classification&scope=samples&page=2&page_size=50',
    );
    await listAnalyses();
    assert.equal(calls[1].url, 'https://api.example.test/api/v1/analyses');

    await getAnalysis('abc');
    assert.equal(calls[2].url, 'https://api.example.test/api/v1/analyses/abc');
    assert.equal(calls[2].init.method, 'GET');

    await deleteAnalysis('abc');
    assert.equal(calls[3].url, 'https://api.example.test/api/v1/analyses/abc');
    assert.equal(calls[3].init.method, 'DELETE');
  } finally {
    restore();
  }
});

test('getReady: 200 is ok, 503 with a body is DATA (degraded), anything else throws', async () => {
  const ok = { status: 'ok', components: { database: { status: 'ok', detail: null }, models: { status: 'ok', detail: 'placeholder' } } };
  const bad = {
    status: 'degraded',
    components: { database: { status: 'unavailable', detail: 'down' }, models: { status: 'ok', detail: null } },
  };
  let respond: () => Response = () => json(ok);
  const { calls, restore } = stubFetch(() => respond());
  try {
    const good = await getReady();
    assert.equal(good.ok, true);
    assert.equal(good.components.models.detail, 'placeholder');
    assert.equal(calls[0].url, 'https://api.example.test/health/ready'); // outside /api/v1
    assert.equal('X-Client-Id' in (calls[0].init.headers as Record<string, string>), false);

    respond = () => json(bad, 503);
    const degraded = await getReady();
    assert.equal(degraded.ok, false);
    assert.equal(degraded.components.database.status, 'unavailable');

    respond = () => new Response('<html>tunnel down</html>', { status: 503 });
    await assert.rejects(getReady(), (err: unknown) => err instanceof ApiError && err.code === 'SERVER_UNREACHABLE');

    respond = () => new Response('', { status: 530 });
    await assert.rejects(getReady(), (err: unknown) => err instanceof ApiError && err.code === 'SERVER_UNREACHABLE');
  } finally {
    restore();
  }
});
