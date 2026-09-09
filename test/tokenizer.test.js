/**
 * Port of tests/test_tokenizer.py.
 *
 * Each pytest function maps to one describe() block; the parametrize decorators
 * become loops that generate one test() per case.
 */

import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test, { describe } from "node:test";

import { BasicTokenizer } from "../src/basic.js";
import { RegexTokenizer } from "../src/regex.js";
import {
  CASE_NAMES,
  TIKTOKEN_IDS,
  llamaText,
  sharedGpt4,
  specialTokens,
  specialsString,
  testStrings,
  unpack,
} from "./common.js";

const factories = [
  ["BasicTokenizer", () => new BasicTokenizer()],
  ["RegexTokenizer", () => new RegexTokenizer()],
  ["GPT4Tokenizer", () => sharedGpt4()],
];

// test encode/decode identity for a few different strings
describe("test_encode_decode_identity", () => {
  for (const [factoryName, factory] of factories) {
    for (let i = 0; i < testStrings.length; i++) {
      test(`${factoryName} :: ${JSON.stringify(testStrings[i])}`, async () => {
        const text = unpack(testStrings[i]);
        const tokenizer = await factory();
        const ids = tokenizer.encode(text);
        const decoded = tokenizer.decode(ids);
        assert.equal(text, decoded);
      });
    }
  }
});

// test that our tokenizer matches the official GPT-4 tokenizer
describe("test_gpt4_tiktoken_equality", () => {
  for (let i = 0; i < testStrings.length; i++) {
    test(`test_gpt4_tiktoken_equality :: ${CASE_NAMES[i]} :: ${JSON.stringify(testStrings[i])}`, async () => {
      const text = unpack(testStrings[i]);
      const tokenizer = await sharedGpt4();
      const tiktokenIds = TIKTOKEN_IDS[CASE_NAMES[i]];
      const gpt4TokenizerIds = tokenizer.encode(text);
      assert.deepEqual(gpt4TokenizerIds, tiktokenIds);
    });
  }
});

// test the handling of special tokens
test("test_gpt4_tiktoken_equality_special_tokens", async () => {
  const tokenizer = await sharedGpt4();
  const tiktokenIds = TIKTOKEN_IDS.specials_all;
  const gpt4TokenizerIds = tokenizer.encode(specialsString, "all");
  assert.deepEqual(gpt4TokenizerIds, tiktokenIds);
});

// reference test to add more tests in the future
describe("test_wikipedia_example", () => {
  /*
   * Quick unit test, following along the Wikipedia example:
   * https://en.wikipedia.org/wiki/Byte_pair_encoding
   *
   * According to Wikipedia, running bpe on the input string:
   * "aaabdaaabac"
   *
   * for 3 merges will result in string:
   * "XdXac"
   *
   * where:
   * X=ZY
   * Y=ab
   * Z=aa
   *
   * Keep in mind that for us a=97, b=98, c=99, d=100 (ASCII values)
   * so Z will be 256, Y will be 257, X will be 258.
   *
   * So we expect the output list of ids to be [258, 100, 258, 97, 99]
   */
  for (const [factoryName, factory] of factories.slice(0, 2)) {
    test(factoryName, () => {
      const tokenizer = factory();
      const text = "aaabdaaabac";
      tokenizer.train(text, 256 + 3);
      const ids = tokenizer.encode(text);
      assert.deepEqual(ids, [258, 100, 258, 97, 99]);
      assert.equal(tokenizer.decode(tokenizer.encode(text)), text);
    });
  }
});

describe("test_save_load", () => {
  for (const [label, specials] of [
    ["no special tokens", {}],
    ["with special tokens", specialTokens],
  ]) {
    test(label, async () => {
      // take a bit more complex piece of text and train the tokenizer, chosen at random
      const text = llamaText;
      // create a Tokenizer and do 64 merges
      let tokenizer = new RegexTokenizer();
      tokenizer.train(text, 256 + 64);
      tokenizer.registerSpecialTokens(specials);
      // verify that decode(encode(x)) == x
      assert.equal(tokenizer.decode(tokenizer.encode(text, "all")), text);
      // verify that save/load work as expected
      const ids = tokenizer.encode(text, "all");
      const dir = fs.mkdtempSync(path.join(os.tmpdir(), "minbpe-test-"));
      const prefix = path.join(dir, "test_tokenizer_tmp");
      try {
        await tokenizer.save(prefix);
        // re-load the tokenizer
        tokenizer = new RegexTokenizer();
        await tokenizer.load(`${prefix}.model`);
        // verify that decode(encode(x)) == x
        assert.equal(tokenizer.decode(ids), text);
        assert.equal(tokenizer.decode(tokenizer.encode(text, "all")), text);
        assert.deepEqual(tokenizer.encode(text, "all"), ids);
      } finally {
        fs.rmSync(dir, { recursive: true, force: true });
      }
    });
  }
});
