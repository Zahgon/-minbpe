/**
 * Implements the GPT-4 Tokenizer as a light wrapper around the RegexTokenizer.
 * Note that this is a pretrained tokenizer, it does not support training.
 *
 * Port of minbpe/gpt4.py.
 */

import { renderToken } from "./base.js";
import {
  bytesToKey,
  concatBytes,
  decodeUtf8,
  keyToBytes,
  pairFirst,
  pairKey,
  pairSecond,
} from "./bytes.js";
import { GPT4_SPLIT_PATTERN, RegexTokenizer } from "./regex.js";
import { loadCl100kBaseRanks } from "./ranks.js";

/**
 * Recover the merge that produced `token`, by re-running BPE and stopping just
 * before `maxRank` -- i.e. reconstructing the state of the merge list at the
 * moment `token` was created. The result is the pair of parts that were joined.
 *
 * Token parts are latin1 strings rather than byte arrays so that "concatenate
 * two parts and look up their rank" is a plain string concat plus a Map hit.
 *
 * @param {Map<string, number>} mergeableRanks
 * @param {string} token latin1 byte-key
 * @param {number|null} maxRank
 * @returns {string[]}
 */
export function bpe(mergeableRanks, token, maxRank = null) {
  let parts = Array.from(token);
  while (true) {
    let minIdx = null;
    let minRank = null;
    for (let i = 0; i + 1 < parts.length; i++) {
      const rank = mergeableRanks.get(parts[i] + parts[i + 1]);
      if (rank !== undefined && (minRank === null || rank < minRank)) {
        minIdx = i;
        minRank = rank;
      }
    }
    if (minRank === null || (maxRank !== null && minRank >= maxRank)) break;
    parts = [
      ...parts.slice(0, minIdx),
      parts[minIdx] + parts[minIdx + 1],
      ...parts.slice(minIdx + 2),
    ];
  }
  return parts;
}

/**
 * Reverse-engineer tiktoken's flat rank table back into a merge list.
 *
 * tiktoken ships only "token bytes -> rank"; minbpe needs "(id, id) -> id".
 * For every multi-byte token we replay BPE with a rank ceiling to find its two
 * immediate parents, then record their ids as the merge.
 *
 * @param {Map<string, number>} mergeableRanks
 * @returns {Map<number, number>} packed pair -> token id
 */
export function recoverMerges(mergeableRanks) {
  const merges = new Map();
  for (const [token, rank] of mergeableRanks) {
    if (token.length === 1) continue;
    const pair = bpe(mergeableRanks, token, rank);
    if (pair.length !== 2) {
      throw new Error(`expected to recover a pair for rank ${rank}, got ${pair.length} parts`);
    }
    const ix0 = mergeableRanks.get(pair[0]);
    const ix1 = mergeableRanks.get(pair[1]);
    if (ix0 === undefined || ix1 === undefined) {
      throw new Error(`recovered parts for rank ${rank} are not themselves tokens`);
    }
    merges.set(pairKey(ix0, ix1), rank);
  }
  return merges;
}

export const GPT4_SPECIAL_TOKENS = {
  "<|endoftext|>": 100257,
  "<|fim_prefix|>": 100258,
  "<|fim_middle|>": 100259,
  "<|fim_suffix|>": 100260,
  "<|endofprompt|>": 100276,
};

export class GPT4Tokenizer extends RegexTokenizer {
  /**
   * Loading the cl100k ranks is async, so construction goes through the static
   * `create()` factory instead of `new GPT4Tokenizer()` directly.
   */
  constructor(mergeableRanks) {
    super(GPT4_SPLIT_PATTERN);
    if (mergeableRanks === undefined) {
      throw new Error("use `await GPT4Tokenizer.create()` to construct a GPT4Tokenizer");
    }

    this.mergeableRanks = mergeableRanks;
    this.merges = recoverMerges(mergeableRanks);

    const vocab = new Map();
    for (let idx = 0; idx < 256; idx++) vocab.set(idx, Uint8Array.of(idx));
    for (const [pair, idx] of this.merges) {
      vocab.set(idx, concatBytes([vocab.get(pairFirst(pair)), vocab.get(pairSecond(pair))]));
    }
    this.vocab = vocab;

    // the GPT-4 tokenizer permutes the raw bytes before BPE; this historical
    // quirk has to be replayed exactly or every id comes out wrong
    this.byteShuffle = new Map();
    this.inverseByteShuffle = new Map();
    for (let i = 0; i < 256; i++) {
      const rank = mergeableRanks.get(String.fromCharCode(i));
      if (rank === undefined) throw new Error(`byte ${i} is missing from the rank table`);
      this.byteShuffle.set(i, rank);
      this.inverseByteShuffle.set(rank, i);
    }

    this.registerSpecialTokens(GPT4_SPECIAL_TOKENS);
  }

  /** @param {{ ranks?: Map<string, number> }} [options] */
  static async create(options = {}) {
    const ranks = options.ranks ?? (await loadCl100kBaseRanks(options));
    return new GPT4Tokenizer(ranks);
  }

  _encodeChunk(textBytes) {
    const shuffled = new Uint8Array(textBytes.length);
    for (let i = 0; i < textBytes.length; i++) shuffled[i] = this.byteShuffle.get(textBytes[i]);
    return super._encodeChunk(shuffled);
  }

  decode(ids) {
    const parts = [];
    for (const idx of ids) {
      const token = this.vocab.get(idx);
      if (token === undefined) throw new Error(`invalid token id: ${idx}`);
      parts.push(token);
    }
    const joined = concatBytes(parts);
    const unshuffled = new Uint8Array(joined.length);
    for (let i = 0; i < joined.length; i++) unshuffled[i] = this.inverseByteShuffle.get(joined[i]);
    return decodeUtf8(unshuffled);
  }

  train() {
    throw new Error("NotImplementedError: GPT4Tokenizer cannot be trained.");
  }

  save() {
    throw new Error("NotImplementedError: GPT4Tokenizer cannot be saved.");
  }

  load() {
    throw new Error("NotImplementedError: GPT4Tokenizer cannot be loaded.");
  }

  /**
   * Write the GPT-4 tokens in the exact same format the base class would.
   * Simply run as:
   *   node -e "import('./src/index.js').then(async m => (await m.GPT4Tokenizer.create()).saveVocab('gpt4.vocab'))"
   */
  async saveVocab(vocabFile) {
    const { writeFile } = await import("node:fs/promises");

    const vocab = new Map();
    for (let idx = 0; idx < 256; idx++) {
      vocab.set(idx, Uint8Array.of(this.inverseByteShuffle.get(idx)));
    }
    for (const [pair, idx] of this.merges) {
      vocab.set(idx, concatBytes([vocab.get(pairFirst(pair)), vocab.get(pairSecond(pair))]));
    }

    const invertedMerges = new Map();
    for (const [pair, idx] of this.merges) invertedMerges.set(idx, pair);

    const lines = [];
    for (const [idx, token] of vocab) {
      const s = renderToken(token);
      if (invertedMerges.has(idx)) {
        const pair = invertedMerges.get(idx);
        const s0 = renderToken(vocab.get(pairFirst(pair)));
        const s1 = renderToken(vocab.get(pairSecond(pair)));
        lines.push(`[${s0}][${s1}] -> [${s}] ${idx}`);
      } else {
        lines.push(`[${s}] ${idx}`);
      }
    }
    await writeFile(vocabFile, lines.join("\n") + "\n", "utf-8");
  }
}
