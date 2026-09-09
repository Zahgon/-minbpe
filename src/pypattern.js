/**
 * Translate a Python `regex`-module pattern into an equivalent JS RegExp source.
 *
 * The original minbpe uses the third-party `regex` module, whose GPT split
 * patterns rely on two features JS RegExp does not have:
 *
 *   1. Possessive quantifiers   `?+`  `++`  `*+`  `{n,m}+`
 *   2. Scoped inline flags      `(?i:...)`
 *
 * Rather than hand-maintaining a second copy of each pattern, we translate.
 * This keeps `tokenizer.pattern` holding the *Python* source string, which is
 * what gets written to `.model` files -- required for byte-exact save/load
 * parity with the Python implementation.
 *
 * ---------------------------------------------------------------------------
 * On dropping possessiveness
 *
 * `X*+` is atomic: once matched it never gives characters back. Translating it
 * to plain greedy `X*` permits backtracking that the original forbids, so the
 * two are equivalent only when no backtracking path could ever alter the match.
 *
 * For both shipped GPT patterns that holds, because every possessive atom is a
 * character class disjoint from whatever follows it:
 *
 *   [^\r\n\p{L}\p{N}]?+\p{L}+   the optional atom explicitly excludes \p{L},
 *                              so giving it back can never help \p{L}+ match.
 *   [^\s\p{L}\p{N}]++[\r\n]*    the repeated atom excludes \s, and [\r\n] is
 *                              a subset of \s, so the trailing part can never
 *                              reclaim a character the greedy run consumed.
 *
 * This is not taken on faith: test/parity.test.js diffs the JS chunk output
 * against Python's over the full 185KB corpus plus a purpose-built adversarial
 * corpus that hammers exactly these boundaries.
 *
 * `translatePythonPattern` throws on possessive constructs it cannot prove are
 * safe only in the sense that it does not attempt atomic-group emulation; for
 * arbitrary user patterns the caller is responsible for verifying equivalence.
 * ---------------------------------------------------------------------------
 */

/**
 * The exact set of code points Python's `regex` module matches with `\s`,
 * expressed as JS character-class members.
 *
 * JS `\s` is NOT equivalent. Determined empirically by sweeping all 0x110000
 * code points through both runtimes; the sets differ on exactly two:
 *
 *   U+0085 (NEL)          matched by Python `\s`, NOT by JS `\s`
 *   U+FEFF (ZWNBSP / BOM) matched by JS `\s`, NOT by Python `\s`
 *
 * Both appear in the GPT split patterns via `\s+`, `\s*` and `[^\s\p{L}\p{N}]`,
 * so using the JS default silently mis-chunks any text containing a BOM or a
 * NEL. Do not "simplify" this back to `\s`.
 */
const PY_SPACE_MEMBERS =
  "\\t\\n\\v\\f\\r \\u0085\\u00a0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";

/** Is `c` an ASCII letter? */
function isAsciiLetter(c) {
  return (c >= "a" && c <= "z") || (c >= "A" && c <= "Z");
}

function swapCase(c) {
  return c >= "a" && c <= "z" ? c.toUpperCase() : c.toLowerCase();
}

/**
 * Consume one escape sequence starting at `i` (which points at the backslash).
 * Returns the end index (exclusive). Handles \p{...} / \P{...} so that the
 * braces are not mistaken for a quantifier.
 */
function escapeEnd(src, i) {
  const next = src[i + 1];
  if (next === "p" || next === "P") {
    if (src[i + 2] === "{") {
      const close = src.indexOf("}", i + 3);
      if (close === -1) throw new SyntaxError(`unterminated \\p{...} at ${i}`);
      return close + 1;
    }
    return i + 3; // \pL shorthand
  }
  if (next === "x") return i + 4;
  if (next === "u") return src[i + 2] === "{" ? src.indexOf("}", i + 2) + 1 : i + 6;
  return i + 2;
}

/**
 * Find the index of the `)` closing the group whose `(` is at `open`.
 * Respects escapes and character classes.
 */
function groupEnd(src, open) {
  let depth = 0;
  let inClass = false;
  for (let i = open; i < src.length; i++) {
    const c = src[i];
    if (c === "\\") {
      i = escapeEnd(src, i) - 1;
      continue;
    }
    if (inClass) {
      if (c === "]") inClass = false;
      continue;
    }
    if (c === "[") inClass = true;
    else if (c === "(") depth++;
    else if (c === ")") {
      depth--;
      if (depth === 0) return i;
    }
  }
  throw new SyntaxError(`unbalanced group starting at ${open}`);
}

/**
 * Core translation pass.
 * @param {string} src        pattern source
 * @param {boolean} caseFold  expand ASCII letters to case-insensitive classes
 */
function translate(src, caseFold) {
  let out = "";
  let i = 0;

  // True when the previously emitted token was a quantifier, meaning a
  // following '+' is a possessive marker rather than a quantifier of its own.
  let afterQuantifier = false;

  while (i < src.length) {
    const c = src[i];

    // ---- escape sequences -------------------------------------------------
    if (c === "\\") {
      const end = escapeEnd(src, i);
      const seq = src.slice(i, end);
      if (seq === "\\s") out += `[${PY_SPACE_MEMBERS}]`;
      else if (seq === "\\S") out += `[^${PY_SPACE_MEMBERS}]`;
      else out += seq;
      i = end;
      afterQuantifier = false;
      continue;
    }

    // ---- character class --------------------------------------------------
    if (c === "[") {
      let j = i + 1;
      let body = "";
      if (src[j] === "^") {
        body += "^";
        j++;
      }
      if (src[j] === "]") {
        // a literal ']' as the first member
        body += "\\]";
        j++;
      }
      while (j < src.length && src[j] !== "]") {
        if (src[j] === "\\") {
          const end = escapeEnd(src, j);
          const seq = src.slice(j, end);
          if (seq === "\\s") {
            body += PY_SPACE_MEMBERS;
          } else if (seq === "\\S") {
            throw new SyntaxError(
              "\\S inside a character class cannot be expressed as a member list"
            );
          } else {
            body += seq;
          }
          j = end;
          continue;
        }
        const ch = src[j];
        if (caseFold && isAsciiLetter(ch)) {
          // inside a class, case folding means adding the counterpart
          body += ch + swapCase(ch);
        } else {
          body += ch;
        }
        j++;
      }
      if (j >= src.length) throw new SyntaxError(`unterminated class at ${i}`);
      out += "[" + body + "]";
      i = j + 1;
      afterQuantifier = false;
      continue;
    }

    // ---- groups -----------------------------------------------------------
    if (c === "(") {
      // scoped inline flags: (?i:...) / (?i)... is not supported by JS
      const m = /^\(\?([a-zA-Z]+):/.exec(src.slice(i));
      if (m) {
        const flags = m[1];
        if (flags.replace(/[im]/g, "") !== "") {
          throw new SyntaxError(
            `unsupported inline flags "(?${flags}:" -- only 'i' is translated`
          );
        }
        const close = groupEnd(src, i);
        const inner = src.slice(i + m[0].length, close);
        out += "(?:" + translate(inner, caseFold || flags.includes("i")) + ")";
        i = close + 1;
        afterQuantifier = false;
        continue;
      }
      // ordinary group: recurse so nested content is translated too
      const close = groupEnd(src, i);
      const prefix = /^\(\?[:=!<]?[a-zA-Z<>=!]*/.exec(src.slice(i, close + 1));
      // Preserve the group opener verbatim (handles (?:, (?=, (?!, (?<=, (?<!,
      // (?<name>) and plain '(' ), translating only the body.
      let openerLen = 1;
      if (src[i + 1] === "?") {
        const om = /^\(\?(?:<[A-Za-z_$][A-Za-z0-9_$]*>|<[=!]|[:=!>])/.exec(
          src.slice(i)
        );
        if (!om) throw new SyntaxError(`unsupported group syntax at ${i}`);
        openerLen = om[0].length;
        if (om[0] === "(?>") {
          // atomic group -- JS has none; emit a non-capturing group and rely on
          // the caller's equivalence testing (same caveat as possessive).
          out += "(?:" + translate(src.slice(i + 3, close), caseFold) + ")";
          i = close + 1;
          afterQuantifier = false;
          continue;
        }
      }
      void prefix;
      out +=
        src.slice(i, i + openerLen) +
        translate(src.slice(i + openerLen, close), caseFold) +
        ")";
      i = close + 1;
      afterQuantifier = false;
      continue;
    }

    // ---- quantifiers ------------------------------------------------------
    if (c === "*" || c === "+" || c === "?") {
      if (afterQuantifier && c === "+") {
        // possessive marker -> drop (see the module header for why this is safe
        // for the shipped patterns, and how it is verified)
        i++;
        afterQuantifier = false;
        continue;
      }
      if (afterQuantifier && c === "?") {
        // lazy marker: `*?` etc. Valid in JS, emit as-is.
        out += c;
        i++;
        afterQuantifier = false;
        continue;
      }
      out += c;
      i++;
      afterQuantifier = true;
      continue;
    }

    // ---- {n,m} ------------------------------------------------------------
    if (c === "{") {
      const m = /^\{\d*(?:,\d*)?\}/.exec(src.slice(i));
      if (m) {
        out += m[0];
        i += m[0].length;
        afterQuantifier = true;
        continue;
      }
      out += "\\{";
      i++;
      afterQuantifier = false;
      continue;
    }

    // ---- literal ----------------------------------------------------------
    if (caseFold && isAsciiLetter(c)) {
      out += "[" + c + swapCase(c) + "]";
    } else {
      out += c;
    }
    i++;
    afterQuantifier = false;
  }

  return out;
}

/**
 * Translate a Python `regex` pattern to JS RegExp source.
 * @param {string} pattern
 * @returns {string} JS-compatible source
 */
export function translatePythonPattern(pattern) {
  return translate(pattern, false);
}

/**
 * Compile a Python pattern into a sticky-free global RegExp suitable for
 * `matchAll`. Always returns a fresh object so `lastIndex` is never shared.
 */
export function compilePythonPattern(pattern) {
  return new RegExp(translatePythonPattern(pattern), "gu");
}

/** Python: re.escape(s) -- equivalent matching behaviour, not identical output. */
export function escapeRegExp(s) {
  return s.replace(/[.*+?^${}()|[\]\\\-\/]/g, "\\$&");
}
