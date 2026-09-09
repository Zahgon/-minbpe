/**
 * Byte-for-byte parity against the original Python implementation.
 *
 * Every fixture under test/fixtures/golden/ was produced by running the
 * original minbpe under CPython in the migration's differential harness, which
 * is kept outside this repository so that running the suite needs nothing but
 * Node. These tests are what make "functional equivalence" a measured property
 * rather than a claim.
 */

import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test, { describe } from "node:test";

import { renderToken } from "../src/base.js";
import { BasicTokenizer } from "../src/basic.js";
import { GPT2_SPLIT_PATTERN, GPT4_SPLIT_PATTERN, RegexTokenizer } from "../src/regex.js";
import { pairFirst, pairSecond, reprBytes } from "../src/bytes.js";
import { GOLDEN, golden, llamaText, sharedGpt4, specialTokens, testStrings, unpack } from "./common.js";

const taylorswift = unpack("FILE:taylorswift.txt");

/** Merges as [[p0, p1], idx] pairs, the shape gen_golden.py dumped. */
function mergesAsJson(merges) {
  return [...merges].map(([key, idx]) => [[pairFirst(key), pairSecond(key)], idx]);
}

async function withTempDir(fn) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "minbpe-parity-"));
  try {
    return await fn(dir);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

function assertFileMatchesGolden(actualPath, goldenName) {
  const actual = fs.readFileSync(actualPath);
  const expected = fs.readFileSync(path.join(GOLDEN, goldenName));
  assert.ok(
    actual.equals(expected),
    `${goldenName} differs from the Python output (${actual.length} vs ${expected.length} bytes)`
  );
}

describe("regex pattern translation", () => {
  test("gpt4 pattern chunks taylorswift.txt identically", () => {
    const t = new RegexTokenizer(GPT4_SPLIT_PATTERN);
    assert.deepEqual(t._split(taylorswift), golden("chunks_gpt4.json"));
  });

  test("gpt2 pattern chunks taylorswift.txt identically", () => {
    const t = new RegexTokenizer(GPT2_SPLIT_PATTERN);
    assert.deepEqual(t._split(taylorswift), golden("chunks_gpt2.json"));
  });

  test("gpt4 pattern chunks llama_text identically", () => {
    const t = new RegexTokenizer(GPT4_SPLIT_PATTERN);
    assert.deepEqual(t._split(llamaText), golden("chunks_gpt4_llama.json"));
  });

  test("gpt4 pattern chunks the small test strings identically", () => {
    const t = new RegexTokenizer(GPT4_SPLIT_PATTERN);
    const expected = golden("chunks_gpt4_small.json");
    for (let i = 0; i < 3; i++) {
      assert.deepEqual(t._split(testStrings[i]), expected[i], `case ${i}`);
    }
  });

  // the possessive quantifiers and (?i:...) in the GPT-4 pattern have no direct
  // JS equivalent; this corpus is the evidence that the translation is exact
  for (const [label, pattern, goldenName] of [
    ["gpt4", GPT4_SPLIT_PATTERN, "adversarial_chunks_gpt4.json"],
    ["gpt2", GPT2_SPLIT_PATTERN, "adversarial_chunks_gpt2.json"],
  ]) {
    test(`${label} pattern matches on the full adversarial corpus`, () => {
      const t = new RegexTokenizer(pattern);
      const corpus = golden("adversarial_corpus.json");
      const expected = golden(goldenName);
      assert.equal(corpus.length, expected.length);
      const failures = [];
      for (let i = 0; i < corpus.length; i++) {
        const got = t._split(corpus[i]);
        if (JSON.stringify(got) !== JSON.stringify(expected[i])) {
          failures.push({ i, input: corpus[i], got, want: expected[i] });
        }
      }
      assert.deepEqual(failures, [], `${failures.length}/${corpus.length} chunkings diverged`);
    });
  }
});

describe("BasicTokenizer parity", () => {
  const t = new BasicTokenizer();
  t.train(taylorswift, 512);

  test("merge sequence", () => {
    assert.deepEqual(mergesAsJson(t.merges), golden("basic_merges.json"));
  });

  test("encodes taylorswift.txt identically", () => {
    assert.deepEqual(t.encode(taylorswift), golden("basic_encoded_taylorswift.json"));
  });

  test("save() produces byte-identical .model and .vocab", async () => {
    await withTempDir(async (dir) => {
      await t.save(path.join(dir, "basic"));
      assertFileMatchesGolden(path.join(dir, "basic.model"), "basic.model");
      assertFileMatchesGolden(path.join(dir, "basic.vocab"), "basic.vocab");
    });
  });
});

describe("RegexTokenizer parity", () => {
  const t = new RegexTokenizer();
  t.train(taylorswift, 512);

  test("merge sequence", () => {
    assert.deepEqual(mergesAsJson(t.merges), golden("regex_merges.json"));
  });

  test("encodes taylorswift.txt identically", () => {
    assert.deepEqual(t.encode(taylorswift), golden("regex_encoded_taylorswift.json"));
  });

  test("save() produces byte-identical .model and .vocab", async () => {
    await withTempDir(async (dir) => {
      await t.save(path.join(dir, "regex"));
      assertFileMatchesGolden(path.join(dir, "regex.model"), "regex.model");
      assertFileMatchesGolden(path.join(dir, "regex.vocab"), "regex.vocab");
    });
  });

  test("renderToken matches over the whole vocab", () => {
    const expected = golden("render_token_regex.json");
    for (const [idx, token] of t.vocab) {
      assert.equal(renderToken(token), expected[String(idx)], `token ${idx}`);
    }
  });
});

describe("llama_text save/load parity", () => {
  for (const [label, specials] of [
    ["nospecial", {}],
    ["special", specialTokens],
  ]) {
    test(`${label}: .model, .vocab and ids all match`, async () => {
      const t = new RegexTokenizer();
      t.train(llamaText, 256 + 64);
      t.registerSpecialTokens(specials);
      assert.deepEqual(t.encode(llamaText, "all"), golden(`llama_${label}_ids.json`));
      await withTempDir(async (dir) => {
        const prefix = path.join(dir, `llama_${label}`);
        await t.save(prefix);
        assertFileMatchesGolden(`${prefix}.model`, `llama_${label}.model`);
        assertFileMatchesGolden(`${prefix}.vocab`, `llama_${label}.vocab`);

        const reloaded = new RegexTokenizer();
        await reloaded.load(`${prefix}.model`);
        assert.equal(reloaded.pattern, t.pattern, "pattern must survive a save/load round trip");
        assert.deepEqual(mergesAsJson(reloaded.merges), mergesAsJson(t.merges));
      });
    });
  }
});

describe("GPT4Tokenizer parity", () => {
  test("recoverMerges reproduces all 100000 merges", async () => {
    const t = await sharedGpt4();
    assert.deepEqual(mergesAsJson(t.merges), golden("gpt4_merges.json"));
  });

  test("byte shuffle table", async () => {
    const t = await sharedGpt4();
    const expected = golden("gpt4_byte_shuffle.json");
    for (let i = 0; i < 256; i++) assert.equal(t.byteShuffle.get(i), expected[i], `byte ${i}`);
  });

  test("saveVocab() produces a byte-identical gpt4.vocab", async () => {
    const t = await sharedGpt4();
    await withTempDir(async (dir) => {
      const out = path.join(dir, "gpt4.vocab");
      await t.saveVocab(out);
      assertFileMatchesGolden(out, "gpt4.vocab");
    });
  });

  for (const method of ["train", "save", "load"]) {
    test(`${method}() throws NotImplementedError`, async () => {
      const t = await sharedGpt4();
      assert.throws(() => t[method]("x"), /NotImplementedError/);
    });
  }
});

describe("wikipedia example parity", () => {
  const expected = golden("wikipedia_example.json");
  for (const [name, Cls] of [
    ["basic", BasicTokenizer],
    ["regex", RegexTokenizer],
  ]) {
    test(name, () => {
      const t = new Cls();
      t.train("aaabdaaabac", 256 + 3);
      assert.deepEqual(t.encode("aaabdaaabac"), expected[name].ids);
      assert.deepEqual(mergesAsJson(t.merges), expected[name].merges);
    });
  }
});

describe("allowedSpecial modes", () => {
  const expected = golden("special_modes.json");
  const build = () => {
    const t = new RegexTokenizer();
    t.train(llamaText, 256 + 64);
    t.registerSpecialTokens(specialTokens);
    return t;
  };
  const specialsString = golden("specials_string.json");

  test('"all" encodes every special token as a single id', () => {
    assert.deepEqual(build().encode(specialsString, "all"), expected.all);
  });

  test('"none" encodes special tokens as ordinary text', () => {
    const t = build();
    assert.deepEqual(t.encode(specialsString, "none"), expected.none);
    assert.deepEqual(t.encode(specialsString, "none"), t.encodeOrdinary(specialsString));
  });

  test("a Set restricts which special tokens are honoured", () => {
    const got = build().encode(specialsString, new Set(["<|endoftext|>"]));
    assert.deepEqual(got, expected.subset_endoftext);
  });

  test('"none_raise" throws when a special token appears', () => {
    assert.equal(expected.none_raise_raises, true);
    assert.throws(() => build().encode(specialsString, "none_raise"), /not allowed/);
  });

  test('"none_raise" is the default and passes on clean text', () => {
    const t = build();
    assert.deepEqual(t.encode("no specials here"), expected.none_raise_ok_on_clean_text);
  });
});

test("reprBytes reproduces CPython's repr() of bytes", () => {
  // verbose training output embeds these, so a mismatch would silently change
  // what `npm run train` prints relative to `python train.py`
  const cases = golden("repr_bytes.json");
  for (const [byteList, expected] of cases) {
    assert.equal(reprBytes(Uint8Array.from(byteList)), expected, JSON.stringify(byteList));
  }
});
