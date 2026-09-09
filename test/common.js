/**
 * Common test data, ported from tests/test_tokenizer.py.
 *
 * The Python suite calls tiktoken at runtime to check GPT-4 parity. There is no
 * tiktoken for JS here, so the reference ids were captured once from the Python
 * implementation into test/fixtures/golden/ by tools/gen_golden.py, and the
 * generator asserted that minbpe and tiktoken agreed at capture time.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const TEST_DIR = path.dirname(fileURLToPath(import.meta.url));
export const FIXTURES = path.join(TEST_DIR, "fixtures");
export const GOLDEN = path.join(FIXTURES, "golden");

export function golden(name) {
  return JSON.parse(fs.readFileSync(path.join(GOLDEN, name), "utf-8"));
}

export function goldenRaw(name) {
  return fs.readFileSync(path.join(GOLDEN, name));
}

// a few strings to test the tokenizers on
export const testStrings = [
  "", // empty string
  "?", // single character
  "hello world!!!? (안녕하세요!) lol123 😉", // fun small string
  "FILE:taylorswift.txt", // FILE: is handled as a special string in unpack()
];

/**
 * Python's unpack() avoids printing a whole file into pytest's parametrize
 * output. It is kept here so the test ids stay readable in the same way.
 */
export function unpack(text) {
  if (text.startsWith("FILE:")) {
    return fs.readFileSync(path.join(FIXTURES, text.slice(5)), "utf-8");
  }
  return text;
}

export const specialsString = golden("specials_string.json");
export const specialTokens = {
  "<|endoftext|>": 100257,
  "<|fim_prefix|>": 100258,
  "<|fim_middle|>": 100259,
  "<|fim_suffix|>": 100260,
  "<|endofprompt|>": 100276,
};
export const llamaText = golden("llama_text.json");

/** Golden ids keyed by the same case names gen_golden.py used. */
export const TIKTOKEN_IDS = golden("tiktoken_encoded.json");
export const CASE_NAMES = ["empty", "single", "fun", "taylorswift"];

let gpt4Promise = null;

/** Building a GPT4Tokenizer costs ~0.6s, so every test shares one instance. */
export async function sharedGpt4() {
  if (gpt4Promise === null) {
    const { GPT4Tokenizer } = await import("../src/gpt4.js");
    gpt4Promise = GPT4Tokenizer.create();
  }
  return gpt4Promise;
}
