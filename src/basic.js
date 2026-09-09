/**
 * Minimal (byte-level) Byte Pair Encoding tokenizer.
 *
 * Algorithmically follows along the GPT tokenizer:
 * https://github.com/openai/gpt-2/blob/master/src/encoder.py
 *
 * But:
 * - Does not handle the regular expression splitting pattern.
 * - Does not handle any special tokens.
 *
 * Port of minbpe/basic.py.
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

export class BasicTokenizer extends Tokenizer {
  train(text, vocabSize, { verbose = false } = {}) {
    if (vocabSize < 256) throw new Error("vocabSize must be >= 256");
    const numMerges = vocabSize - 256;

    const textBytes = encodeUtf8(text);
    let ids = Array.from(textBytes);

    const merges = new Map();
    const vocab = new Map();
    for (let idx = 0; idx < 256; idx++) vocab.set(idx, Uint8Array.of(idx));

    for (let i = 0; i < numMerges; i++) {
      const stats = getStats(ids);
      const pair = maxPair(stats);
      if (pair === null) {
        // Python's max() raises ValueError on an empty sequence; preserve the
        // failure rather than silently truncating the vocabulary
        throw new Error(
          `cannot train to vocabSize ${vocabSize}: ran out of pairs to merge after ${i} merges`
        );
      }
      const idx = 256 + i;
      ids = merge(ids, pair, idx);
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

  decode(ids) {
    const parts = [];
    for (const idx of ids) {
      const token = this.vocab.get(idx);
      if (token === undefined) throw new Error(`invalid token id: ${idx}`);
      parts.push(token);
    }
    return decodeUtf8(concatBytes(parts));
  }

  encode(text) {
    let ids = Array.from(encodeUtf8(text));
    while (ids.length >= 2) {
      const stats = getStats(ids);
      const pair = minMergePair(stats, this.merges);
      // when nothing is mergeable minMergePair returns the first pair, so the
      // membership check is what actually terminates the loop
      if (!this.merges.has(pair)) break;
      ids = merge(ids, pair, this.merges.get(pair));
    }
    return ids;
  }
}
