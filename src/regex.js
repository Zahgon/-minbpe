/**
 * Minimal (byte-level) Byte Pair Encoding tokenizer.
 *
 * Unlike BasicTokenizer:
 * - RegexTokenizer handles an optional regex splitting pattern.
 * - RegexTokenizer handles optional special tokens.
 *
 * Port of minbpe/regex.py.
 */

import { Tokenizer, getStats, maxPair, merge, minMergePair } from "./base.js";
import {
  concatBytes,
  decodeUtf8,
  encodeUtf8,
  pairFirst,
  pairSecond,
  reprBytes,
  reprPair,
} from "./bytes.js";
import { compilePythonPattern, escapeRegExp } from "./pypattern.js";

/**
 * The main GPT text split patterns, see
 * https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
 *
 * These are stored in their original Python `regex`-module syntax, including
 * possessive quantifiers and (?i:...), because `pattern` is what gets written
 * into .model files and must stay byte-compatible with the Python
 * implementation. They are translated to JS syntax at compile time by
 * pypattern.js.
 */
export const GPT2_SPLIT_PATTERN =
  "'(?:[sdmt]|ll|ve|re)| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
export const GPT4_SPLIT_PATTERN =
  "'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+";

/** Normalise a str->int mapping given as a Map or a plain object. */
function toMap(mapping) {
  if (mapping instanceof Map) return new Map(mapping);
  return new Map(Object.entries(mapping ?? {}));
}

export class RegexTokenizer extends Tokenizer {
  /**
   * @param {string|null} pattern optional override of the default GPT-4 split
   *   pattern, in Python `regex` syntax
   */
  constructor(pattern = null) {
    super();
    this.pattern = pattern === null ? GPT4_SPLIT_PATTERN : pattern;
    this.specialTokens = new Map();
    this.inverseSpecialTokens = new Map();
  }

  /**
   * `pattern` is backed by an accessor so the compiled RegExp is rebuilt
   * whenever the source changes -- notably inside load(), which assigns the
   * pattern read back from the model file.
   */
  get pattern() {
    return this._pattern;
  }

  set pattern(value) {
    this._pattern = value;
    this._compiledPattern = value ? compilePythonPattern(value) : null;
  }

  /** The translated, JS-native RegExp actually used for splitting. */
  get compiledPattern() {
    return this._compiledPattern;
  }

  /** Python: re.findall(self.compiled_pattern, text) */
  _split(text) {
    if (this._compiledPattern === null) return text.length ? [text] : [];
    const out = [];
    for (const m of text.matchAll(this._compiledPattern)) out.push(m[0]);
    return out;
  }

  train(text, vocabSize, { verbose = false } = {}) {
    if (vocabSize < 256) throw new Error("vocabSize must be >= 256");
    const numMerges = vocabSize - 256;

    const textChunks = this._split(text);
    let ids = textChunks.map((ch) => Array.from(encodeUtf8(ch)));

    const merges = new Map();
    const vocab = new Map();
    for (let idx = 0; idx < 256; idx++) vocab.set(idx, Uint8Array.of(idx));

    for (let i = 0; i < numMerges; i++) {
      const stats = new Map();
      for (const chunkIds of ids) getStats(chunkIds, stats);
      const pair = maxPair(stats);
      if (pair === null) {
        throw new Error(
          `cannot train to vocabSize ${vocabSize}: ran out of pairs to merge after ${i} merges`
        );
      }
      const idx = 256 + i;
      ids = ids.map((chunkIds) => merge(chunkIds, pair, idx));
      merges.set(pair, idx);
      vocab.set(idx, concatBytes([vocab.get(pairFirst(pair)), vocab.get(pairSecond(pair))]));
      if (verbose) {
        console.log(
          `merge ${i + 1}/${numMerges}: ${reprPair(pairFirst(pair), pairSecond(pair))} -> ` +
            `${idx} (${reprBytes(vocab.get(idx))}) had ${stats.get(pair)} occurrences`
        );
      }
    }

    this.merges = merges;
    this.vocab = vocab;
  }

  /** @param {Map<string, number>|Record<string, number>} specialTokens */
  registerSpecialTokens(specialTokens) {
    this.specialTokens = toMap(specialTokens);
    this.inverseSpecialTokens = new Map();
    for (const [k, v] of this.specialTokens) this.inverseSpecialTokens.set(v, k);
  }

  decode(ids) {
    const partBytes = [];
    for (const idx of ids) {
      if (this.vocab.has(idx)) {
        partBytes.push(this.vocab.get(idx));
      } else if (this.inverseSpecialTokens.has(idx)) {
        partBytes.push(encodeUtf8(this.inverseSpecialTokens.get(idx)));
      } else {
        throw new Error(`invalid token id: ${idx}`);
      }
    }
    return decodeUtf8(concatBytes(partBytes));
  }

  _encodeChunk(textBytes) {
    let ids = Array.from(textBytes);
    while (ids.length >= 2) {
      const stats = getStats(ids);
      const pair = minMergePair(stats, this.merges);
      if (!this.merges.has(pair)) break;
      ids = merge(ids, pair, this.merges.get(pair));
    }
    return ids;
  }

  /** Encoding that ignores any special tokens. */
  encodeOrdinary(text) {
    const ids = [];
    for (const chunk of this._split(text)) {
      for (const id of this._encodeChunk(encodeUtf8(chunk))) ids.push(id);
    }
    return ids;
  }

  /**
   * Unlike encodeOrdinary, this handles special tokens.
   *
   * @param {string} text
   * @param {"all"|"none"|"none_raise"|Set<string>} allowedSpecial
   *   "none_raise" (the default, matching tiktoken) throws if any special token
   *   appears in the text.
   */
  encode(text, allowedSpecial = "none_raise") {
    let special;
    if (allowedSpecial === "all") {
      special = this.specialTokens;
    } else if (allowedSpecial === "none") {
      special = new Map();
    } else if (allowedSpecial === "none_raise") {
      special = new Map();
      for (const token of this.specialTokens.keys()) {
        if (text.includes(token)) {
          throw new Error(`special token ${token} is not allowed in the input text`);
        }
      }
    } else if (allowedSpecial instanceof Set) {
      special = new Map();
      for (const [k, v] of this.specialTokens) {
        if (allowedSpecial.has(k)) special.set(k, v);
      }
    } else {
      throw new Error(`allowedSpecial=${String(allowedSpecial)} not understood`);
    }

    if (special.size === 0) return this.encodeOrdinary(text);

    // wrapping the alternation in a capturing group makes split() keep the
    // special tokens themselves in the output
    const specialPattern =
      "(" + [...special.keys()].map(escapeRegExp).join("|") + ")";
    const specialChunks = text.split(new RegExp(specialPattern, "u"));

    const ids = [];
    for (const part of specialChunks) {
      if (special.has(part)) {
        ids.push(special.get(part));
      } else {
        for (const id of this.encodeOrdinary(part)) ids.push(id);
      }
    }
    return ids;
  }
}
