// The program the harness runs: node /bot/main.ts (Node 22). It reads every
// case from /input/cases.json and writes /output/answers.json with the answers
// in the cases' order. Outside a container MAKEFUN_INPUT_DIR and
// MAKEFUN_OUTPUT_DIR stand in for /input and /output.
//
// The fast path hands the whole cases.json to solve.js's transduce(), which
// unescapes the texts and answers them inside the WebAssembly module and
// writes answers.json straight out. The driver never JSON.parses the cases
// and never JSON.stringifies the answers, which is where the time used to go.
// When transduce() cannot answer (a lone surrogate somewhere in the file,
// which has no UTF-8 form), the driver falls back to answering each case
// alone with run(), which handles that case in JavaScript.

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { setFlagsFromString } from 'node:v8';

// Incremental marking off before solve.js is imported, so all of its work runs
// under it, as the house baseline's driver does.
setFlagsFromString('--no-incremental-marking');

const solve = (await import(new URL('./solve.js', import.meta.url).href)) as {
  run: (input: unknown) => unknown;
  transduce: (casesBuf: Buffer) => Buffer | null;
};

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// The output directory is the harness's to create, but making sure it is
// there costs nothing and keeps a local run from failing on it.
mkdirSync(OUTPUT_DIR, { recursive: true });

const casesBuf = readFileSync(join(INPUT_DIR, 'cases.json'));

// Fast path: the whole file transduced at once, answers written as produced.
const fast = solve.transduce(casesBuf);
if (fast !== null) {
  writeFileSync(join(OUTPUT_DIR, 'answers.json'), fast);
} else {
  // Fallback: answer each case alone, in order, and stringify the answers.
  const cases = JSON.parse(casesBuf.toString('utf8')) as {
    id: string;
    input: unknown;
  }[];
  const answers = cases.map((c) => solve.run(c.input));
  writeFileSync(join(OUTPUT_DIR, 'answers.json'), JSON.stringify(answers));
}
