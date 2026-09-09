/**
 * Byte-level helpers.
 *
 * Python's `bytes` maps to `Uint8Array` here, and Python's tuple-keyed dicts
 * map to `Map` with an encoded scalar key. These helpers exist so that the
 * encoding choice lives in exactly one place.
 */

const encoder = new TextEncoder();
// Python's bytes.decode("utf-8", errors="replace") does NOT strip a BOM, so
// neither do we. `fatal: false` is the default and yields U+FFFD on bad input.
const decoder = new TextDecoder("utf-8", { ignoreBOM: true, fatal: false });

/** Python: text.encode("utf-8") */
export function encodeUtf8(text) {
  return encoder.encode(text);
}

/** Python: text_bytes.decode("utf-8", errors="replace") */
export function decodeUtf8(bytes) {
  return decoder.decode(bytes instanceof Uint8Array ? bytes : Uint8Array.from(bytes));
}

/** Python: b"".join(parts) */
export function concatBytes(parts) {
  let total = 0;
  for (const p of parts) total += p.length;
  const out = new Uint8Array(total);
  let off = 0;
  for (const p of parts) {
    out.set(p, off);
    off += p.length;
  }
  return out;
}

/* -------------------------------------------------------------------------- *
 * Pair keys
 *
 * `merges` is dict[(int, int), int] and `stats` is dict[(int, int), int] in
 * Python. JS Map keys use SameValueZero, so arrays compare by reference and are
 * unusable. We pack the pair into a single Number.
 *
 * Token ids reach ~100_277 for cl100k, so a shift of 2**20 (1_048_576) leaves
 * an order of magnitude of headroom while keeping the product far below
 * Number.MAX_SAFE_INTEGER (2**53). This is meaningfully faster than string keys
 * on the hot training loop.
 * -------------------------------------------------------------------------- */

const PAIR_SHIFT = 1048576; // 2**20

export function pairKey(a, b) {
  return a * PAIR_SHIFT + b;
}

export function pairFirst(key) {
  return Math.floor(key / PAIR_SHIFT);
}

export function pairSecond(key) {
  return key % PAIR_SHIFT;
}

export function unpairKey(key) {
  return [pairFirst(key), pairSecond(key)];
}

/* -------------------------------------------------------------------------- *
 * Byte-sequence keys (used by the GPT-4 merge recovery, which keys a Map by
 * byte strings). latin1 gives one char per byte: compact and lossless.
 * -------------------------------------------------------------------------- */

export function bytesToKey(bytes) {
  let s = "";
  // chunked to avoid blowing the argument limit on long tokens
  for (let i = 0; i < bytes.length; i += 4096) {
    s += String.fromCharCode.apply(null, bytes.subarray(i, i + 4096));
  }
  return s;
}

export function keyToBytes(key) {
  const out = new Uint8Array(key.length);
  for (let i = 0; i < key.length; i++) out[i] = key.charCodeAt(i);
  return out;
}

/* -------------------------------------------------------------------------- *
 * Python repr formatting
 *
 * train.py's verbose output interpolates a tuple and a bytes object directly.
 * To keep training logs diffable against the original we reproduce CPython's
 * repr exactly.
 * -------------------------------------------------------------------------- */

/** Python: repr((a, b)) -> "(97, 98)" */
export function reprPair(a, b) {
  return `(${a}, ${b})`;
}

/** Python: repr(b'...') */
export function reprBytes(bytes) {
  // CPython prefers single quotes, switching to double only when the content
  // contains a single quote but no double quote.
  const hasSingle = bytes.includes(0x27);
  const hasDouble = bytes.includes(0x22);
  const quote = hasSingle && !hasDouble ? '"' : "'";

  let body = "";
  for (const b of bytes) {
    if (b === 0x5c) body += "\\\\";
    else if (b === quote.charCodeAt(0)) body += "\\" + quote;
    else if (b === 0x0a) body += "\\n";
    else if (b === 0x0d) body += "\\r";
    else if (b === 0x09) body += "\\t";
    else if (b >= 0x20 && b < 0x7f) body += String.fromCharCode(b);
    else body += "\\x" + b.toString(16).padStart(2, "0");
  }
  return `b${quote}${body}${quote}`;
}
