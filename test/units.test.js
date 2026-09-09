/**
 * Direct unit coverage for the public helpers the end-to-end suites reach only
 * indirectly, or not at all: the abstract base contract, the two key-packing
 * inverses, the tuple repr, and the compiled-pattern accessor.
 */

import assert from "node:assert/strict";
import test, { describe } from "node:test";

import { Tokenizer } from "../src/base.js";
import {
  bytesToKey,
  keyToBytes,
  pairKey,
  reprPair,
  unpairKey,
} from "../src/bytes.js";
import { GPT4_SPLIT_PATTERN, RegexTokenizer } from "../src/regex.js";

describe("Tokenizer abstract base contract", () => {
  const base = new Tokenizer();

  test("train refuses to run on the base class", () => {
    assert.throws(() => base.train("abc", 300), /NotImplementedError/);
  });

  test("encode refuses to run on the base class", () => {
    assert.throws(() => base.encode("abc"), /NotImplementedError/);
  });

  test("decode refuses to run on the base class", () => {
    assert.throws(() => base.decode([1, 2, 3]), /NotImplementedError/);
  });
});

describe("packed pair keys", () => {
  test("unpairKey inverts pairKey across the id range", () => {
    for (const [a, b] of [
      [0, 0],
      [1, 2],
      [255, 256],
      [100256, 100276],
    ]) {
      assert.deepEqual(unpairKey(pairKey(a, b)), [a, b]);
    }
  });

  test("unpairKey keeps the two halves independent", () => {
    assert.deepEqual(unpairKey(pairKey(7, 9)), [7, 9]);
    assert.deepEqual(unpairKey(pairKey(9, 7)), [9, 7]);
  });
});

describe("byte-sequence keys", () => {
  test("keyToBytes inverts bytesToKey for every byte value", () => {
    const all = Uint8Array.from({ length: 256 }, (_, i) => i);
    assert.deepEqual(keyToBytes(bytesToKey(all)), all);
  });

  test("keyToBytes inverts bytesToKey past the chunking boundary", () => {
    const long = Uint8Array.from({ length: 9000 }, (_, i) => i % 256);
    assert.deepEqual(keyToBytes(bytesToKey(long)), long);
  });
});

describe("Python repr formatting", () => {
  test("reprPair renders a two-element tuple like CPython", () => {
    assert.equal(reprPair(101, 32), "(101, 32)");
    assert.equal(reprPair("b'a'", "b'b'"), "(b'a', b'b')");
  });
});

describe("RegexTokenizer compiledPattern accessor", () => {
  test("exposes a global RegExp compiled from the default split pattern", () => {
    const tok = new RegexTokenizer();
    assert.ok(tok.compiledPattern instanceof RegExp);
    assert.ok(tok.compiledPattern.global);
    assert.equal(tok.pattern, GPT4_SPLIT_PATTERN);
  });

  test("recompiles when the pattern is reassigned", () => {
    const tok = new RegexTokenizer();
    const before = tok.compiledPattern;
    tok.pattern = "\\p{L}+";
    assert.notEqual(tok.compiledPattern, before);
    assert.deepEqual("hi there".match(tok.compiledPattern), ["hi", "there"]);
  });

  test("is null when the pattern is empty", () => {
    const tok = new RegexTokenizer();
    tok.pattern = "";
    assert.equal(tok.compiledPattern, null);
  });
});
