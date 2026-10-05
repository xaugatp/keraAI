import { test } from 'node:test';
import assert from 'node:assert/strict';
import { MAX_UPLOAD_BYTES, MAX_UPLOAD_MB } from '../../src/api/limits.ts';
import { validateImageFile } from '../../src/api/validateImage.ts';

test('accepts every backend-supported type under the size limit', () => {
  for (const type of ['image/jpeg', 'image/png', 'image/webp']) {
    assert.equal(validateImageFile({ name: 'photo.jpg', type, size: 1024 }), null);
  }
});

test('rejects HEIC/HEIF with a dedicated, friendlier message (by type or by extension)', () => {
  const byType = validateImageFile({ name: 'IMG_0001', type: 'image/heic', size: 1024 });
  const byExtension = validateImageFile({ name: 'IMG_0001.HEIC', type: '', size: 1024 });
  assert.match(byType ?? '', /HEIC/);
  assert.match(byExtension ?? '', /HEIC/);
});

test('rejects an unsupported MIME type with the generic message', () => {
  const message = validateImageFile({ name: 'scan.pdf', type: 'application/pdf', size: 1024 });
  assert.match(message ?? '', /isn't supported/);
});

test('rejects a file over MAX_UPLOAD_BYTES, naming the MB limit', () => {
  const message = validateImageFile({ name: 'huge.jpg', type: 'image/jpeg', size: MAX_UPLOAD_BYTES + 1 });
  assert.match(message ?? '', new RegExp(`${MAX_UPLOAD_MB} MB`));
});

test('a file exactly at the size limit is accepted (boundary)', () => {
  assert.equal(validateImageFile({ name: 'ok.jpg', type: 'image/jpeg', size: MAX_UPLOAD_BYTES }), null);
});
