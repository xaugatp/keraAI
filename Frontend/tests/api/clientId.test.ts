import { test } from 'node:test';
import assert from 'node:assert/strict';
import { UUID_V4, stubGlobal } from './helpers.ts';

// Each import gets a unique query string so the module's in-memory id starts fresh.
let n = 0;
const freshModule = () => import(`../../src/api/clientId.ts?case=${++n}`) as Promise<typeof import('../../src/api/clientId.ts')>;

function memoryStorage(initial: Record<string, string> = {}) {
  const data = new Map(Object.entries(initial));
  return {
    data,
    getItem: (k: string) => data.get(k) ?? null,
    setItem: (k: string, v: string) => void data.set(k, v),
    removeItem: (k: string) => void data.delete(k),
  };
}

test('generates a v4 UUID, persists it, and returns the same one next time', async () => {
  const storage = memoryStorage();
  const restore = stubGlobal('localStorage', storage);
  try {
    const mod = await freshModule();
    const id = mod.getClientId();
    assert.match(id, UUID_V4);
    assert.equal(mod.getClientId(), id);
    assert.equal(storage.data.get(mod.CLIENT_ID_STORAGE_KEY), id);

    // A "new page load" (fresh module) picks the stored id up again.
    const again = await freshModule();
    assert.equal(again.getClientId(), id);
  } finally {
    restore();
  }
});

test('replaces a corrupted stored value with a valid v4 UUID', async () => {
  const mod0 = await freshModule();
  const storage = memoryStorage({ [mod0.CLIENT_ID_STORAGE_KEY]: 'not-a-uuid' });
  const restore = stubGlobal('localStorage', storage);
  try {
    const mod = await freshModule();
    const id = mod.getClientId();
    assert.match(id, UUID_V4);
    assert.equal(storage.data.get(mod.CLIENT_ID_STORAGE_KEY), id);
  } finally {
    restore();
  }
});

test('rejects a stored UUID that is not version 4', async () => {
  const mod0 = await freshModule();
  const v1 = '6ba7b810-9dad-11d1-80b4-00c04fd430c8';
  const restore = stubGlobal('localStorage', memoryStorage({ [mod0.CLIENT_ID_STORAGE_KEY]: v1 }));
  try {
    const id = (await freshModule()).getClientId();
    assert.match(id, UUID_V4);
    assert.notEqual(id, v1);
  } finally {
    restore();
  }
});

test('storage that throws (private mode / blocked): still a stable v4 id, held in memory', async () => {
  const throwing = {
    getItem: () => {
      throw new Error('SecurityError');
    },
    setItem: () => {
      throw new Error('QuotaExceededError');
    },
  };
  const restore = stubGlobal('localStorage', throwing);
  try {
    const mod = await freshModule();
    const id = mod.getClientId();
    assert.match(id, UUID_V4);
    assert.equal(mod.getClientId(), id);
  } finally {
    restore();
  }
});

test('no localStorage at all (accessing it throws): still works', async () => {
  const original = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
  Object.defineProperty(globalThis, 'localStorage', {
    configurable: true,
    get() {
      throw new Error('SecurityError: access denied');
    },
  });
  try {
    const id = (await freshModule()).getClientId();
    assert.match(id, UUID_V4);
  } finally {
    if (original) Object.defineProperty(globalThis, 'localStorage', original);
    else delete (globalThis as Record<string, unknown>).localStorage;
  }
});

test('insecure context (no crypto.randomUUID): falls back to getRandomValues and is still v4', async () => {
  const restore = stubGlobal('crypto', {
    getRandomValues: (bytes: Uint8Array) => {
      for (let i = 0; i < bytes.length; i++) bytes[i] = (i * 37 + 11) & 0xff;
      return bytes;
    },
  });
  try {
    const mod = await freshModule();
    for (let i = 0; i < 5; i++) assert.match(mod.generateUuidV4(), UUID_V4);
  } finally {
    restore();
  }
});

test('even with all-ones / all-zeros random bytes the version and variant bits are right', async () => {
  const mod = await freshModule();
  for (const fill of [0x00, 0xff]) {
    const restore = stubGlobal('crypto', { getRandomValues: (b: Uint8Array) => b.fill(fill) });
    try {
      assert.match(mod.generateUuidV4(), UUID_V4);
    } finally {
      restore();
    }
  }
});

test('no crypto at all: last-resort generator is still a valid v4 UUID', async () => {
  const restore = stubGlobal('crypto', undefined);
  try {
    const mod = await freshModule();
    const ids = new Set<string>();
    for (let i = 0; i < 20; i++) {
      const id = mod.generateUuidV4();
      assert.match(id, UUID_V4);
      ids.add(id);
    }
    assert.ok(ids.size > 1);
  } finally {
    restore();
  }
});

test('isUuidV4', async () => {
  const mod = await freshModule();
  assert.equal(mod.isUuidV4('9b2f4a1e-3c5d-4e6f-8a7b-0c1d2e3f4a5b'), true);
  assert.equal(mod.isUuidV4('9B2F4A1E-3C5D-4E6F-8A7B-0C1D2E3F4A5B'), true);
  assert.equal(mod.isUuidV4('00000000-0000-0000-0000-000000000000'), false); // nil UUID
  assert.equal(mod.isUuidV4('9b2f4a1e-3c5d-1e6f-8a7b-0c1d2e3f4a5b'), false); // version 1
  assert.equal(mod.isUuidV4(null), false);
  assert.equal(mod.isUuidV4(42), false);
});
