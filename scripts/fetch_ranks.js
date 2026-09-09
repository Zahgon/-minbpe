/**
 * Download the official cl100k_base.tiktoken rank table into data/.
 *
 * The file is vendored in the repository so tests run offline; this script only
 * matters for a fresh checkout that stripped it, or to refresh it deliberately.
 */

import { loadCl100kBaseRanks, CL100K_BASE_URL } from "../src/ranks.js";

console.log(`fetching ${CL100K_BASE_URL} ...`);
const ranks = await loadCl100kBaseRanks();
console.log(`ok: ${ranks.size} mergeable ranks available`);
