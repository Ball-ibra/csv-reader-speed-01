// Fast drop-in for csv-reader 1.0.0: run(input) returns exactly {columns, rows, errors}
// for {text, header}, with the same broken-row policy, but the work is done by a
// small WebAssembly module (parser.wasm, built from parser.rs in this repo)
// instead of by walking the text one character at a time in JavaScript.
//
// The module holds the text as UTF-8 bytes. ASCII delimiters are single bytes
// in UTF-8, so one SIMD step covers 16 bytes, a field's bytes are the answer's
// bytes (no widening, no packing), and the JSON unescape is a byte copy. The
// parser is event-driven: one SIMD pass lists every structural byte, then the
// parser walks the events instead of re-scanning the text field by field.
// Two entry points share that core: csv_to_json for one {text, header}, and
// cases_to_answers, which takes the whole cases.json and writes answers.json
// straight out, so the driver never JSON.parses the cases or JSON.stringifies
// the answers.
//
// A lone surrogate in the input has no UTF-8 form, so the module cannot carry
// it. run() checks for one first and falls back to the plain JavaScript reader
// at the bottom of this file when it finds one (the drawn sets never hold one,
// but the answer must still be right if one ever turns up). The module also
// reports it through had_lone_surrogate, and transduce() returns null then,
// so the driver falls back to answering case by case.

import { readFileSync } from 'node:fs';

// The module is instantiated synchronously (WebAssembly.Module + new
// WebAssembly.Instance) because run() is synchronous. If the module cannot be
// loaded, every call falls back to the JavaScript reader below.
let wasm = null;
try {
  const bytes = readFileSync(new URL('./parser.wasm', import.meta.url));
  const mod = new WebAssembly.Module(bytes);
  wasm = new WebAssembly.Instance(mod, {}).exports;
} catch {
  wasm = null;
}

const mem = () => new Uint8Array(wasm.memory.buffer);

/** A lone surrogate (high not followed by low, or low not preceded by high). */
const LONE = /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/;

/**
 * Answer one {text, header} with the module. Returns the answer object, or
 * null when the caller should fall back (no module, or a lone surrogate).
 */
function runWasm(text, header) {
  if (wasm === null) return null;
  if (LONE.test(text)) return null;
  const bytes = Buffer.from(text, 'utf8');
  const ptr = wasm.input_ptr(bytes.length);
  mem().set(bytes, ptr);
  wasm.csv_to_json(ptr, bytes.length, header === true ? 1 : 0);
  if (wasm.had_lone_surrogate() !== 0) return null;
  const out = Buffer.from(wasm.memory.buffer, wasm.out_ptr(), wasm.out_len());
  return JSON.parse(out.toString('utf8'));
}

/**
 * Transduce a whole cases.json (a Buffer of its UTF-8 bytes) into the
 * answers.json bytes. Returns a Buffer, or null when the caller should fall
 * back to answering case by case (a lone surrogate somewhere in the file).
 */
export function transduce(casesBuf) {
  if (wasm === null) return null;
  const ptr = wasm.input_ptr(casesBuf.length);
  mem().set(casesBuf, ptr);
  const p = wasm.cases_to_answers(ptr, casesBuf.length);
  if (p === 0 || wasm.had_lone_surrogate() !== 0) return null;
  return Buffer.from(wasm.memory.buffer, wasm.out_ptr(), wasm.out_len());
}

/** run(input) returns csv-reader 1.0.0's answer for {text, header}. */
export function run(input) {
  const text = String(input.text);
  const header = input.header === true;
  const fast = runWasm(text, header);
  if (fast !== null) return fast;
  return readCsvJs(text, header);
}

/**
 * The plain JavaScript reader, used only when the module cannot answer: a
 * lone surrogate in the text, or the module failing to load. It follows
 * csv-reader 1.0.0's reading exactly, one field at a time, so the answer is
 * right even though it is slow. Kept separate from runWasm so the fast path
 * never pays for it.
 */
function readCsvJs(text, header) {
  const errors = [];
  const rows = [];
  let columns = null;
  let width = -1;
  const n = text.length;
  let pos = text.charCodeAt(0) === 0xfeff ? 1 : 0;
  let line = 1;
  let recordLine = 1;
  let recordRow = null;

  const at = (i) => (i < n ? text[i] : '');

  function error(code) {
    errors.push({ code, line: recordLine, row: recordRow });
  }

  // Read an unquoted field onto `value` up to the next comma or line break.
  // A CR not followed by LF is dropped (stray_cr). Returns [value, ended,
  // kept], kept counting the characters this scan kept.
  function readUnquoted(value) {
    let kept = 0;
    while (pos < n) {
      const c = text[pos];
      if (c === ',') {
        pos += 1;
        return [value, 0, kept];
      }
      if (c === '\n') {
        pos += 1;
        line += 1;
        return [value, 1, kept];
      }
      if (c === '\r') {
        if (at(pos + 1) === '\n') {
          pos += 2;
          line += 1;
          return [value, 1, kept];
        }
        error('stray_cr');
        pos += 1;
        continue;
      }
      value += c;
      kept += 1;
      pos += 1;
    }
    return [value, 2, kept];
  }

  // Read a quoted field whose opening quote is at pos. "" inside is one
  // quote; a CRLF or lone CR inside reads as LF and counts one line. Text
  // after the closing quote is read as unquoted junk (quote_junk when any of
  // it is kept). A quote that never closes is a literal character and the
  // text after it is read again as unquoted (unterminated_quote).
  function readQuoted() {
    const open = pos;
    const openLine = line;
    let value = '';
    pos += 1;
    let closed = false;
    while (pos < n) {
      const c = text[pos];
      if (c === '"') {
        if (at(pos + 1) === '"') {
          value += '"';
          pos += 2;
          continue;
        }
        pos += 1;
        closed = true;
        break;
      }
      if (c === '\r') {
        value += '\n';
        line += 1;
        pos += at(pos + 1) === '\n' ? 2 : 1;
        continue;
      }
      if (c === '\n') line += 1;
      value += c;
      pos += 1;
    }
    if (!closed) {
      pos = open + 1;
      line = openLine;
      error('unterminated_quote');
      return readUnquoted('"');
    }
    if (pos >= n) return [value, 2, 0];
    const c = text[pos];
    if (c === ',') {
      pos += 1;
      return [value, 0, 0];
    }
    if (c === '\n') {
      pos += 1;
      line += 1;
      return [value, 1, 0];
    }
    if (c === '\r' && at(pos + 1) === '\n') {
      pos += 2;
      line += 1;
      return [value, 1, 0];
    }
    const junk = readUnquoted(value);
    if (junk[2] > 0) error('quote_junk');
    return junk;
  }

  for (;;) {
    // A line left with no characters is no record.
    while (pos < n) {
      if (text[pos] === '\n') {
        pos += 1;
        line += 1;
      } else if (text[pos] === '\r' && at(pos + 1) === '\n') {
        pos += 2;
        line += 1;
      } else {
        break;
      }
    }
    if (pos >= n) break;

    recordLine = line;
    recordRow = header && columns === null ? null : rows.length;

    const fields = [];
    let field;
    let quoted = false;
    do {
      quoted = text[pos] === '"';
      field = quoted ? readQuoted() : readUnquoted('');
      fields.push(field[0]);
    } while (field[1] === 0);

    // A line that held only stray CRs is left with no characters: no record.
    if (fields.length === 1 && fields[0] === '' && !quoted) continue;

    if (width < 0) width = fields.length;
    if (header && columns === null) {
      columns = fields;
      continue;
    }
    if (fields.length < width) {
      errors.push({ code: 'short_row', line: recordLine, row: recordRow, fields: fields.length });
      while (fields.length < width) fields.push('');
    } else if (fields.length > width) {
      errors.push({ code: 'long_row', line: recordLine, row: recordRow, extra: fields.splice(width) });
    }
    rows.push(fields);
  }
  return { columns: header ? (columns ?? []) : null, rows, errors };
}
