process.env.NODE_ENV = 'test';
if (!process.env.GEMINI_API_KEY && !process.env.GEMINI_API_KEYS) {
  process.env.GEMINI_API_KEY = 'test-key';
}

import test from 'node:test';
import assert from 'node:assert/strict';

const { selectDocsForQuery } = await import('../index.js');

test('ignores documentation context for nonsensical prompts', async () => {
  const result = await selectDocsForQuery('.');
  assert.ok(Array.isArray(result.titles), 'titles should be an array');
  assert.deepStrictEqual(result.docs, [], 'no document snippets should be attached');
});
