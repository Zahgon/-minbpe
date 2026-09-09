# minbpe

Minimal, clean code for the (byte-level) Byte Pair Encoding (BPE) algorithm commonly used in LLM tokenization. The BPE algorithm is "byte-level" because it runs on UTF-8 encoded strings.

This is a JavaScript port of [karpathy/minbpe](https://github.com/karpathy/minbpe), intended to be a functional equivalent of the original Python implementation: same algorithms, same file formats, same tokenizer outputs, byte for byte.

This algorithm was popularized for LLMs by the [GPT-2 paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) and the associated GPT-2 [code release](https://github.com/openai/gpt-2) from OpenAI. [Sennrich et al. 2015](https://arxiv.org/abs/1508.07909) is cited as the original reference for the use of BPE in NLP applications. Today, all modern LLMs (e.g. GPT, Llama, Mistral) use this algorithm to train their tokenizers.

There are two Tokenizers in this repository, both of which can perform the 3 primary functions of a Tokenizer: 1) train the tokenizer vocabulary and merges on a given text, 2) encode from text to tokens, 3) decode from tokens to text. The files of the repo are as follows:

1. [src/base.js](src/base.js): Implements the `Tokenizer` class, which is the base class. It contains the `train`, `encode`, and `decode` stubs, save/load functionality, and there are also a few common utility functions. This class is not meant to be used directly, but rather to be inherited from.
2. [src/basic.js](src/basic.js): Implements the `BasicTokenizer`, the simplest implementation of the BPE algorithm that runs directly on text.
3. [src/regex.js](src/regex.js): Implements the `RegexTokenizer` that further splits the input text by a regex pattern, which is a preprocessing stage that splits up the input text by categories (think: letters, numbers, punctuation) before tokenization. This ensures that no merges will happen across category boundaries. This was introduced in the GPT-2 paper and continues to be in use as of GPT-4. This class also handles special tokens, if any.
4. [src/gpt4.js](src/gpt4.js): Implements the `GPT4Tokenizer`. This class is a light wrapper around the `RegexTokenizer` (2, above) that exactly reproduces the tokenization of GPT-4 in the [tiktoken](https://github.com/openai/tiktoken) library. The wrapping handles some details around recovering the exact merges in the tokenizer, and the handling of some unfortunate (and likely historical?) 1-byte token permutations.

Two supporting files have no counterpart in the Python version, because Python could lean on the `regex` and `tiktoken` packages:

5. [src/pypattern.js](src/pypattern.js): Translates the Python `regex` split patterns into equivalent JavaScript `RegExp` sources (possessive quantifiers, scoped `(?i:...)` groups, and so on). This keeps the pattern strings in saved `.model` files identical to the Python ones.
6. [src/ranks.js](src/ranks.js): Loads the `cl100k_base` mergeable ranks that `GPT4Tokenizer` is built from. A copy of the rank table ships in [data/cl100k_base.tiktoken](data/cl100k_base.tiktoken), so the package has zero runtime dependencies and works offline.

Finally, the script [scripts/train.js](scripts/train.js) trains the two major tokenizers on the input text [test/fixtures/taylorswift.txt](test/fixtures/taylorswift.txt) (this is the Wikipedia entry for her kek) and saves the vocab to disk for visualization. This script runs in about 2.5 seconds on an M1 MacBook.

All of the files above are very short and thoroughly commented, and also contain a usage example on the bottom of the file.

## quick start

As the simplest example, we can reproduce the [Wikipedia article on BPE](https://en.wikipedia.org/wiki/Byte_pair_encoding) as follows:

```js
import { BasicTokenizer } from "minbpe";
const tokenizer = new BasicTokenizer();
const text = "aaabdaaabac";
tokenizer.train(text, 256 + 3); // 256 are the byte tokens, then do 3 merges
console.log(tokenizer.encode(text));
// [258, 100, 258, 97, 99]
console.log(tokenizer.decode([258, 100, 258, 97, 99]));
// aaabdaaabac
await tokenizer.save("toy");
// writes two files: toy.model (for loading) and toy.vocab (for viewing)
```

According to Wikipedia, running bpe on the input string: "aaabdaaabac" for 3 merges results in the string: "XdXac" where  X=ZY, Y=ab, and Z=aa. The tricky thing to note is that minbpe always allocates the 256 individual bytes as tokens, and then merges bytes as needed from there. So for us a=97, b=98, c=99, d=100 (their [ASCII](https://www.asciitable.com) values). Then when (a,a) is merged to Z, Z will become 256. Likewise Y will become 257 and X 258. So we start with the 256 bytes, and do 3 merges to get to the result above, with the expected output of [258, 100, 258, 97, 99].

## inference: GPT-4 comparison

We can verify that the `RegexTokenizer` has feature parity with the GPT-4 tokenizer from [tiktoken](https://github.com/openai/tiktoken) as follows:

```js
const text = "hello123!!!? (안녕하세요!) 😉";

// tiktoken (Python)
// import tiktoken
// enc = tiktoken.get_encoding("cl100k_base")
// print(enc.encode(text))
// [15339, 4513, 12340, 30, 320, 31495, 230, 75265, 243, 92245, 16715, 57037]

// ours
import { GPT4Tokenizer } from "minbpe";
const tokenizer = await GPT4Tokenizer.create();
console.log(tokenizer.encode(text));
// [15339, 4513, 12340, 30, 320, 31495, 230, 75265, 243, 92245, 16715, 57037]
```

Under the hood, the `GPT4Tokenizer` is just a light wrapper around `RegexTokenizer`, passing in the merges and the special tokens of GPT-4. Building it requires reading the `cl100k_base` rank table from disk, which is why it is constructed through the async `GPT4Tokenizer.create()` factory rather than `new GPT4Tokenizer()`. We can also ensure the special tokens are handled correctly:

```js
const text = "<|endoftext|>hello world";

// tiktoken (Python)
// enc.encode(text, allowed_special="all")
// [100257, 15339, 1917]

// ours
import { GPT4Tokenizer } from "minbpe";
const tokenizer = await GPT4Tokenizer.create();
console.log(tokenizer.encode(text, "all"));
// [100257, 15339, 1917]
```

Note that just like tiktoken, we have to explicitly declare our intent to use and parse special tokens in the call to encode. Otherwise this can become a major footgun, unintentionally tokenizing attacker-controlled data (e.g. user prompts) with special tokens. The `allowedSpecial` parameter can be set to "all", "none", "none_raise", or a set of special tokens to allow.

## training

Unlike tiktoken, this code allows you to train your own tokenizer. In principle and to my knowledge, if you train the `RegexTokenizer` on a large dataset with a vocabulary size of 100K, you would reproduce the GPT-4 tokenizer.

There are two paths you can follow. First, you can decide that you don't want the complexity of splitting and preprocessing text with regex patterns, and you also don't care for special tokens. In that case, reach for the `BasicTokenizer`. You can train it, and then encode and decode for example as follows:

```js
import { BasicTokenizer } from "minbpe";
const tokenizer = new BasicTokenizer();
tokenizer.train(veryLongTrainingString, 4096);
tokenizer.encode("hello world"); // string -> tokens
tokenizer.decode([1000, 2000, 3000]); // tokens -> string
await tokenizer.save("mymodel"); // writes mymodel.model and mymodel.vocab
await tokenizer.load("mymodel.model"); // loads the model back, the vocab is just for vis
```

If you instead want to follow along with what OpenAI did for their text tokenizer, it's a good idea to adopt their approach of using a regex pattern to split the text by categories. The GPT-4 pattern is a default with the `RegexTokenizer`, so you'd simply do something like:

```js
import { RegexTokenizer } from "minbpe";
const tokenizer = new RegexTokenizer();
tokenizer.train(veryLongTrainingString, 32768);
tokenizer.encode("hello world"); // string -> tokens
tokenizer.decode([1000, 2000, 3000]); // tokens -> string
await tokenizer.save("tok32k"); // writes tok32k.model and tok32k.vocab
await tokenizer.load("tok32k.model"); // loads the model back from disk
```

Where, of course, you'd want to change around the vocabulary size depending on the size of your dataset.

`save` and `load` are the only async methods on the tokenizers; training, encoding and decoding are all synchronous, exactly as in the Python version.

**Special tokens**. Finally, you might wish to add special tokens to your tokenizer. Register these using the `registerSpecialTokens` function. For example if you train with a vocab size of 32768, then the first 256 tokens are raw byte tokens, the next 32768-256 are merge tokens, and after those you can add the special tokens. The last "real" merge token will have id of 32767 (vocabSize - 1), so your first special token should come right after that, with an id of exactly 32768. So:

```js
import { RegexTokenizer } from "minbpe";
const tokenizer = new RegexTokenizer();
tokenizer.train(veryLongTrainingString, 32768);
tokenizer.registerSpecialTokens({ "<|endoftext|>": 32768 });
tokenizer.encode("<|endoftext|>hello world", "all");
```

You can of course add more tokens after that as well, as you like. Finally, I'd like to stress that I tried hard to keep the code itself clean, readable and hackable. You should not have to feel scared to read the code and understand how it works. The tests are also a nice place to look for more usage examples. That reminds me:

## tests

We use the built-in Node test runner, so there is nothing to install. All tests are located in the `test/` directory:

```bash
$ npm test
```

[test/tokenizer.test.js](test/tokenizer.test.js) is a direct port of the original pytest suite. [test/parity.test.js](test/parity.test.js) goes further and asserts byte-for-byte equality against golden fixtures recorded from the original implementation, covering the regex split patterns, the merge sequences, the encoded ids, and the exact contents of the `.model` and `.vocab` files.

The fixtures under `test/fixtures/golden/` are checked in, so the test suite needs nothing but Node. They were produced by the migration's differential verification harness, which lives outside this repository so that the repository itself has no dependency on the original toolchain.

## training on the sample text

```bash
$ npm run train
```

This trains both tokenizers on `test/fixtures/taylorswift.txt` and writes `models/basic.model`, `models/basic.vocab`, `models/regex.model` and `models/regex.vocab`.

## community extensions

* [gnp/minbpe-rs](https://github.com/gnp/minbpe-rs): A Rust implementation of `minbpe` providing (near) one-to-one correspondence with the Python version

## exercise

For those trying to study BPE, here is the advised progression exercise for how you can build your own minbpe step by step. See [exercise.md](exercise.md).

## lecture

The code in the original repository was built in this [YouTube video](https://www.youtube.com/watch?v=zduSFxRajkE). You can also find this lecture in text form in [lecture.md](lecture.md).

## todos

- write a more optimized version that could run over large files and big vocabs
- write an even more optimized C or Rust version (think through)
- rename GPT4Tokenizer to GPTTokenizer and support GPT-2/GPT-3/GPT-3.5 as well?
- write a LlamaTokenizer similar to GPT4Tokenizer (i.e. attempt sentencepiece equivalent)

## License

MIT
