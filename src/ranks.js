/**
 * Loader for tiktoken's cl100k_base mergeable ranks.
 *
 * The Python original gets these from `tiktoken.get_encoding("cl100k_base")`
 * and reads the private `_mergeable_ranks` attribute. There is no equivalent
 * dependency in JS, so we read the same canonical artifact tiktoken itself
 * downloads: the official `cl100k_base.tiktoken` file, whose every line is
 * "<base64 token bytes> <rank>".
 *
 * Resolution order:
 *   1. $MINBPE_CL100K_PATH             -- explicit override
 *   2. <package>/data/cl100k_base.tiktoken  -- vendored, so tests run offline
 *   3. download from the official URL and cache it under $MINBPE_CACHE_DIR
 */

import { bytesToKey } from "./bytes.js";

export const CL100K_BASE_URL =
  "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken";

/** SHA-256 of the official artifact; a mismatch means a corrupt or wrong file. */
export const CL100K_BASE_SHA256 =
  "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7";

export const CL100K_BASE_SIZE = 100256;

function decodeBase64(s) {
  if (typeof Buffer !== "undefined") return new Uint8Array(Buffer.from(s, "base64"));
  const bin = atob(s);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/**
 * Parse the .tiktoken wire format into a rank map.
 *
 * Entries are inserted in ascending rank order, which recoverMerges() relies on
 * when it reconstructs merges: a token's two parents must already be known by
 * the time the token itself is processed.
 *
 * @param {string} content
 * @returns {Map<string, number>} latin1 byte-key -> rank
 */
export function parseTiktokenFile(content) {
  const byRank = [];
  for (const line of content.split("\n")) {
    if (line === "") continue;
    const sp = line.indexOf(" ");
    if (sp === -1) throw new Error(`malformed .tiktoken line: ${JSON.stringify(line)}`);
    const rank = Number(line.slice(sp + 1));
    if (!Number.isInteger(rank)) {
      throw new Error(`malformed rank in .tiktoken line: ${JSON.stringify(line)}`);
    }
    byRank[rank] = decodeBase64(line.slice(0, sp));
  }
  const ranks = new Map();
  for (let rank = 0; rank < byRank.length; rank++) {
    if (byRank[rank] === undefined) throw new Error(`.tiktoken file is missing rank ${rank}`);
    ranks.set(bytesToKey(byRank[rank]), rank);
  }
  return ranks;
}

async function sha256Hex(bytes) {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

let cached = null;

/**
 * Return cl100k_base's mergeable ranks, memoised for the process lifetime.
 *
 * @param {{ path?: string, allowDownload?: boolean }} [options]
 * @returns {Promise<Map<string, number>>}
 */
export async function loadCl100kBaseRanks(options = {}) {
  if (cached !== null && options.path === undefined) return cached;

  const fs = await import("node:fs/promises");
  const { fileURLToPath } = await import("node:url");
  const path = await import("node:path");

  const here = path.dirname(fileURLToPath(import.meta.url));
  const vendored = path.join(here, "..", "data", "cl100k_base.tiktoken");
  const explicit = options.path ?? process.env.MINBPE_CL100K_PATH;

  let raw = null;
  for (const candidate of [explicit, vendored]) {
    if (!candidate) continue;
    try {
      raw = await fs.readFile(candidate);
      break;
    } catch (err) {
      if (candidate === explicit) throw err;
    }
  }

  if (raw === null) {
    if (options.allowDownload === false) {
      throw new Error(
        `cl100k_base.tiktoken not found at ${vendored}. Run "npm run fetch-ranks" or set MINBPE_CL100K_PATH.`
      );
    }
    const res = await fetch(CL100K_BASE_URL);
    if (!res.ok) throw new Error(`failed to download ${CL100K_BASE_URL}: HTTP ${res.status}`);
    raw = new Uint8Array(await res.arrayBuffer());
    await fs.mkdir(path.dirname(vendored), { recursive: true });
    await fs.writeFile(vendored, raw);
  }

  const actual = await sha256Hex(raw);
  if (actual !== CL100K_BASE_SHA256) {
    throw new Error(
      `cl100k_base.tiktoken failed its integrity check\n  expected sha256 ${CL100K_BASE_SHA256}\n  actual   sha256 ${actual}`
    );
  }

  const ranks = parseTiktokenFile(new TextDecoder().decode(raw));
  if (ranks.size !== CL100K_BASE_SIZE) {
    throw new Error(`expected ${CL100K_BASE_SIZE} mergeable ranks, got ${ranks.size}`);
  }
  if (options.path === undefined) cached = ranks;
  return ranks;
}
