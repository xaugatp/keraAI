import { test } from 'node:test';
import assert from 'node:assert/strict';
import { shouldAutoSubmitHandoff } from '../../src/api/handoff.ts';
import type { AnalysisDetail } from '../../src/api/types.ts';

const FAKE_LEAF_RESULT = {} as AnalysisDetail;

test('submits when a hand-off file is present, there is no leaf result yet, and it was never sent', () => {
  const file = new File([], 'tree.jpg');
  assert.equal(shouldAutoSubmitHandoff({ file, leaf: null, alreadySubmittedFile: null }), true);
});

test('does not submit when there is no file (a direct visit with nothing handed off)', () => {
  assert.equal(shouldAutoSubmitHandoff({ file: null, leaf: null, alreadySubmittedFile: null }), false);
});

test('does not submit once a leaf result already exists for this session', () => {
  const file = new File([], 'tree.jpg');
  assert.equal(
    shouldAutoSubmitHandoff({ file, leaf: FAKE_LEAF_RESULT, alreadySubmittedFile: null }),
    false,
  );
});

test('does not re-submit the same file that was already sent (guards re-renders and failed submits)', () => {
  const file = new File([], 'tree.jpg');
  assert.equal(shouldAutoSubmitHandoff({ file, leaf: null, alreadySubmittedFile: file }), false);
});

test('submits again for a genuinely different file even if another one was already sent', () => {
  const fileA = new File([], 'a.jpg');
  const fileB = new File([], 'b.jpg');
  assert.equal(shouldAutoSubmitHandoff({ file: fileB, leaf: null, alreadySubmittedFile: fileA }), true);
});
