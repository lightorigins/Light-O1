import { test } from "node:test";
import assert from "node:assert/strict";
import { readHistory, rememberPrompt, HISTORY_KEY } from "../src/usability.js";
test("history is bounded, deduplicated and tolerates invalid/private storage", () => {
  const values = new Map();
  const storage = { getItem: (key) => values.get(key), setItem: (key, value) => values.set(key, value) };
  for (let i = 0; i < 15; i++) rememberPrompt(storage, `action ${i}`);
  assert.equal(readHistory(storage).length, 10);
  rememberPrompt(storage, " action 10 ");
  assert.equal(readHistory(storage)[0], "action 10");
  assert.equal(readHistory(storage).length, 10);
  storage.setItem(HISTORY_KEY, "{}");
  assert.deepEqual(readHistory(storage), []);
  storage.setItem(HISTORY_KEY, '[null,1,"valid","valid"]');
  assert.deepEqual(readHistory(storage), ["valid"]);
  assert.deepEqual(readHistory(undefined), []);
  assert.deepEqual(rememberPrompt(undefined, "wave"), ["wave"]);
});
