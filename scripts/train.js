/**
 * Train our Tokenizers on some data, just to see them in action.
 * The whole thing runs in ~2 seconds on my laptop.
 *
 * Port of train.py.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { BasicTokenizer } from "../src/basic.js";
import { RegexTokenizer } from "../src/regex.js";

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");

// open some text and train a vocab of 512 tokens
const text = fs.readFileSync(path.join(root, "test/fixtures/taylorswift.txt"), "utf-8");

// create a directory for models, so we don't pollute the current directory
fs.mkdirSync(path.join(root, "models"), { recursive: true });

const t0 = performance.now();
for (const [TokenizerClass, name] of [
  [BasicTokenizer, "basic"],
  [RegexTokenizer, "regex"],
]) {
  // construct the Tokenizer object and kick off verbose training
  const tokenizer = new TokenizerClass();
  tokenizer.train(text, 512, { verbose: true });
  // writes two files in the models directory: name.model, and name.vocab
  const prefix = path.join(root, "models", name);
  await tokenizer.save(prefix);
}
const t1 = performance.now();

console.log(`Training took ${((t1 - t0) / 1000).toFixed(2)} seconds`);
