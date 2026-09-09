/**
 * Base Tokenizer class and common helpers, including save/load.
 *
 * Port of minbpe/base.py.
 */

import { concatBytes, decodeUtf8, encodeUtf8, pairFirst, pairKey, pairSecond } from "./bytes.js";

/* -------------------------------------------------------------------------- *
 * helpers shared by BasicTokenizer and RegexTokenizer
 * -------------------------------------------------------------------------- */

/**
 * Count consecutive pairs in a list of integers.
 * [1, 2, 3, 1, 2] -> Map { (1,2) => 2, (2,3) => 1, (3,1) => 1 }
 *
 * Keys are packed pairs (see bytes.js). Insertion order is significant: the
 * training loop's `max` and the encoding loop's `min` both break ties by
 * first-seen, matching Python's dict iteration order.
 *
 * @param {number[]} ids
 * @param {Map<number, number>} [counts] updated in place when provided
 * @returns {Map<number, number>}
 */
export function getStats(ids, counts = null) {
  const result = counts === null ? new Map() : counts;
  for (let i = 0; i + 1 < ids.length; i++) {
    const key = pairKey(ids[i], ids[i + 1]);
    result.set(key, (result.get(key) ?? 0) + 1);
  }
  return result;
}

/**
 * Replace every consecutive occurrence of `pair` with `idx`.
 * ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
 *
 * @param {number[]} ids
 * @param {number} pair packed pair key
 * @param {number} idx
 * @returns {number[]}
 */
export function merge(ids, pair, idx) {
  const p0 = pairFirst(pair);
  const p1 = pairSecond(pair);
  const newids = [];
  let i = 0;
  while (i < ids.length) {
    if (ids[i] === p0 && i < ids.length - 1 && ids[i + 1] === p1) {
      newids.push(idx);
      i += 2;
    } else {
      newids.push(ids[i]);
      i += 1;
    }
  }
  return newids;
}

/**
 * Return the packed pair with the highest count, ties broken by first-seen.
 * Mirrors Python's `max(stats, key=stats.get)`.
 */
export function maxPair(stats) {
  let best = null;
  let bestCount = -Infinity;
  for (const [key, count] of stats) {
    if (count > bestCount) {
      bestCount = count;
      best = key;
    }
  }
  return best;
}

/**
 * Return the packed pair with the lowest merge index, ties broken by first-seen.
 * Mirrors Python's `min(stats, key=lambda p: self.merges.get(p, float("inf")))`.
 *
 * When no pair is mergeable every rank is Infinity and Python's `min` returns
 * the first pair arbitrarily; callers detect that case with a membership check
 * on the result. The `first` flag reproduces that behaviour exactly -- without
 * it we would return null and break the caller's termination check.
 */
export function minMergePair(stats, merges) {
  let best = null;
  let bestRank = Infinity;
  let first = true;
  for (const key of stats.keys()) {
    const rank = merges.get(key) ?? Infinity;
    if (first || rank < bestRank) {
      bestRank = rank;
      best = key;
      first = false;
    }
  }
  return best;
}

const CONTROL_RE = /\p{C}/u;

/**
 * Escape control characters so printing a token cannot corrupt the output.
 *
 * Python uses `unicodedata.category(ch)[0] != "C"`; the JS equivalent is the
 * General_Category=Other property escape, which covers the same
 * {Cc, Cf, Cs, Co, Cn} set.
 *
 * Iteration is by code point, not UTF-16 code unit, so astral characters
 * survive intact.
 */
export function replaceControlCharacters(s) {
  let out = "";
  for (const ch of s) {
    if (!CONTROL_RE.test(ch)) {
      out += ch;
    } else {
      out += "\\u" + ch.codePointAt(0).toString(16).padStart(4, "0");
    }
  }
  return out;
}

/** Pretty print a token, escaping control characters. */
export function renderToken(t) {
  return replaceControlCharacters(decodeUtf8(t));
}

/* -------------------------------------------------------------------------- *
 * the base Tokenizer class
 * -------------------------------------------------------------------------- */

export class Tokenizer {
  constructor() {
    /** @type {Map<number, number>} packed pair -> token id */
    this.merges = new Map();
    /** @type {string} */
    this.pattern = "";
    /** @type {Map<string, number>} */
    this.specialTokens = new Map();
    /** @type {Map<number, Uint8Array>} */
    this.vocab = this._buildVocab();
  }

  train(text, vocabSize, options) {
    throw new Error("NotImplementedError: train() is not implemented on the base Tokenizer");
  }

  encode(text) {
    throw new Error("NotImplementedError: encode() is not implemented on the base Tokenizer");
  }

  decode(ids) {
    throw new Error("NotImplementedError: decode() is not implemented on the base Tokenizer");
  }

  /**
   * Vocab is deterministically derived from merges.
   *
   * Correctness depends on `merges` being iterated in ascending-id order so
   * that both parents of every merged token already exist. That holds because
   * merges are inserted in creation order during training and in file order
   * during load().
   */
  _buildVocab() {
    const vocab = new Map();
    for (let idx = 0; idx < 256; idx++) vocab.set(idx, Uint8Array.of(idx));
    for (const [pair, idx] of this.merges) {
      const p0 = vocab.get(pairFirst(pair));
      const p1 = vocab.get(pairSecond(pair));
      if (p0 === undefined || p1 === undefined) {
        throw new Error(
          `cannot build vocab: merge ${pairFirst(pair)},${pairSecond(pair)} -> ${idx} ` +
            `references a token that does not exist yet (merges out of order?)`
        );
      }
      vocab.set(idx, concatBytes([p0, p1]));
    }
    for (const [special, idx] of this.specialTokens) {
      vocab.set(idx, encodeUtf8(special));
    }
    return vocab;
  }

  /**
   * Save `<prefix>.model` (used by load) and `<prefix>.vocab` (human readable).
   *
   * The .model format is byte-compatible with the Python implementation:
   *   minbpe v1\n <pattern>\n <numSpecial>\n <special> <idx>\n... <p0> <p1>\n...
   *
   * Note: Python opens the model file with the platform default encoding while
   * the vocab file is explicitly UTF-8. We write UTF-8 for both, which matches
   * Python on any UTF-8 locale.
   */
  async save(filePrefix) {
    const { writeFile } = await import("node:fs/promises");

    const modelLines = ["minbpe v1", this.pattern, String(this.specialTokens.size)];
    for (const [special, idx] of this.specialTokens) {
      modelLines.push(`${special} ${idx}`);
    }
    for (const pair of this.merges.keys()) {
      modelLines.push(`${pairFirst(pair)} ${pairSecond(pair)}`);
    }
    await writeFile(`${filePrefix}.model`, modelLines.join("\n") + "\n", "utf-8");

    const invertedMerges = new Map();
    for (const [pair, idx] of this.merges) invertedMerges.set(idx, pair);

    const vocabLines = [];
    for (const [idx, token] of this.vocab) {
      // many tokens are partial utf-8 sequences and decode to the replacement
      // character, which is why .vocab can never be used by load()
      const s = renderToken(token);
      if (invertedMerges.has(idx)) {
        const pair = invertedMerges.get(idx);
        const s0 = renderToken(this.vocab.get(pairFirst(pair)));
        const s1 = renderToken(this.vocab.get(pairSecond(pair)));
        vocabLines.push(`[${s0}][${s1}] -> [${s}] ${idx}`);
      } else {
        vocabLines.push(`[${s}] ${idx}`);
      }
    }
    await writeFile(`${filePrefix}.vocab`, vocabLines.join("\n") + "\n", "utf-8");
  }

  /** Inverse of save(), for the model file only. */
  async load(modelFile) {
    if (!modelFile.endsWith(".model")) {
      throw new Error(`model file must end with .model, got: ${modelFile}`);
    }
    const { readFile } = await import("node:fs/promises");
    const raw = await readFile(modelFile, "utf-8");
    const lines = raw.split("\n");

    let cursor = 0;
    const version = lines[cursor++].trim();
    if (version !== "minbpe v1") {
      throw new Error(`unsupported model version: ${JSON.stringify(version)}`);
    }

    // the pattern line is taken verbatim apart from the trailing newline;
    // .trim() would corrupt patterns with meaningful leading/trailing space
    this.pattern = lines[cursor++].replace(/\r$/, "");

    const numSpecial = parseInt(lines[cursor++].trim(), 10);
    const specialTokens = new Map();
    for (let i = 0; i < numSpecial; i++) {
      const parts = lines[cursor++].trim().split(/\s+/);
      specialTokens.set(parts[0], parseInt(parts[1], 10));
    }

    const merges = new Map();
    let idx = 256;
    for (; cursor < lines.length; cursor++) {
      const line = lines[cursor].trim();
      if (line === "") continue;
      const [idx1, idx2] = line.split(/\s+/).map((n) => parseInt(n, 10));
      merges.set(pairKey(idx1, idx2), idx);
      idx += 1;
    }

    this.merges = merges;
    this.specialTokens = specialTokens;
    this.vocab = this._buildVocab();
  }
}
