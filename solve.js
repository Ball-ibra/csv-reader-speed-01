// Fast drop-in for csv-reader 1.0.0: run(input) returns exactly {columns, rows, errors}
// for {text, header}, with the same broken-row policy, but without walking the text
// one character at a time.
//
// Speed ideas, in the order they pay:
// - A record whose line holds no quote at all is split straight from that line:
//   one slice for the line, one indexOf per comma, no per-field line-feed
//   search and no function call per field. Most records of a drawn set are
//   plain like this; quoted fields go through the general path below.
// - In the general path a field is taken with String.slice between delimiters
//   found by indexOf. The searches stay inside the field: the comma and
//   line-feed searches stop at the field's own end, and the carriage return
//   check runs only on the field's slice, never on the rest of the text (an
//   unbound '\r' search would walk to the end of the file on every field of
//   a mostly LF file). Only a field that actually holds a quote, a stray
//   carriage return, or a line break inside quotes is walked more closely,
//   and those are rare in the drawn sets (about one record in a hundred is
//   broken).

/** ended: 0 at a comma, 1 at a line break, 2 at the end of the text. */
export function run(input) {
  const text = String(input.text);
  const header = input.header === true;
  const errors = [];
  const rows = [];
  let columns = null;
  let width = -1;
  const n = text.length;
  let pos = text.charCodeAt(0) === 0xfeff ? 1 : 0;
  let line = 1;
  let recordLine = 1;
  let recordRow = null;

  // Reads an unquoted field from pos, starting from `base` ('' for a fresh field,
  // a quoted value when this is the junk after a closing quote). Returns
  // [value, ended, kept], kept counting the characters this scan kept, so the
  // caller can tell junk from nothing. A CR not followed by LF is dropped and
  // recorded as stray_cr; the scan then carries on from the next character.
  // A CR that sits right before the LF ending the field is the CRLF half of
  // that line break, not a stray CR.
  function readUnquoted(base) {
    let value = base;
    let kept = 0;
    while (pos < n) {
      const comma = text.indexOf(',', pos);
      const lf = text.indexOf('\n', pos);
      let end = n;
      let kind = 2;
      if (comma !== -1 && comma < end) { end = comma; kind = 0; }
      if (lf !== -1 && lf < end) { end = lf; kind = 1; }
      const seg = text.slice(pos, end);
      const tailCr = kind === 1 && seg.charCodeAt(seg.length - 1) === 13;
      const limit = tailCr ? seg.length - 1 : seg.length;
      const cr = seg.indexOf('\r');
      if (cr !== -1 && cr < limit) {
        errors.push({ code: 'stray_cr', line: recordLine, row: recordRow });
        value += seg.slice(0, cr);
        kept += cr;
        pos = pos + cr + 1;
        continue;
      }
      value += tailCr ? seg.slice(0, limit) : seg;
      kept += limit;
      if (kind === 0) { pos = end + 1; return [value, 0, kept]; }
      if (kind === 1) { pos = end + 1; line += 1; return [value, 1, kept]; }
      pos = n;
      return [value, 2, kept];
    }
    return [value, 2, kept];
  }

  // Reads a quoted field whose opening quote is at pos. Returns [value, ended, 0].
  // "" inside is one quote; a CRLF or lone CR inside reads as LF and counts one
  // line. Text after the closing quote is read as unquoted junk (quote_junk when
  // any of it is kept). A quote that never closes is a literal character and the
  // text after it is read again as unquoted (unterminated_quote).
  function readQuoted() {
    const open = pos;
    const openLine = line;
    let cur = pos + 1;
    const parts = [];
    let q = text.indexOf('"', cur);
    while (q !== -1 && text.charCodeAt(q + 1) === 34) {
      parts.push(text.slice(cur, q));
      parts.push('"');
      cur = q + 2;
      q = text.indexOf('"', cur);
    }
    if (q === -1) {
      errors.push({ code: 'unterminated_quote', line: recordLine, row: recordRow });
      pos = open + 1;
      line = openLine;
      return readUnquoted('"');
    }
    parts.push(text.slice(cur, q));
    let content = parts.length === 1 ? parts[0] : parts.join('');
    const nlIn = text.indexOf('\n', open + 1);
    const crIn = text.indexOf('\r', open + 1);
    if ((nlIn !== -1 && nlIn < q) || (crIn !== -1 && crIn < q)) {
      let fixed = '';
      let p = 0;
      while (p < content.length) {
        const nl = content.indexOf('\n', p);
        const cr = content.indexOf('\r', p);
        let cut = content.length;
        let isCr = false;
        if (nl !== -1 && nl < cut) cut = nl;
        if (cr !== -1 && cr < cut) { cut = cr; isCr = true; }
        fixed += content.slice(p, cut);
        if (cut >= content.length) break;
        fixed += '\n';
        line += 1;
        p = isCr && content.charCodeAt(cut + 1) === 10 ? cut + 2 : cut + 1;
      }
      content = fixed;
    }
    const after = q + 1;
    if (after >= n) { pos = n; return [content, 2, 0]; }
    const c = text.charCodeAt(after);
    if (c === 44) { pos = after + 1; return [content, 0, 0]; }
    if (c === 10) { pos = after + 1; line += 1; return [content, 1, 0]; }
    if (c === 13 && text.charCodeAt(after + 1) === 10) { pos = after + 2; line += 1; return [content, 1, 0]; }
    pos = after;
    const junk = readUnquoted(content);
    if (junk[2] > 0) errors.push({ code: 'quote_junk', line: recordLine, row: recordRow });
    return junk;
  }

  // Files a finished record: a line that held only stray CRs is no record, the
  // first record sets the width (and becomes columns when header is true), and
  // later records are padded (short_row) or cut (long_row) to that width.
  function finishRecord(fields, lastQuoted) {
    if (fields.length === 1 && fields[0] === '' && !lastQuoted) return;
    if (width < 0) width = fields.length;
    if (header && columns === null) { columns = fields; return; }
    if (fields.length < width) {
      errors.push({ code: 'short_row', line: recordLine, row: recordRow, fields: fields.length });
      while (fields.length < width) fields.push('');
    } else if (fields.length > width) {
      errors.push({ code: 'long_row', line: recordLine, row: recordRow, extra: fields.splice(width) });
    }
    rows.push(fields);
  }

  for (;;) {
    // A line left with no characters is no record.
    while (pos < n) {
      const c = text.charCodeAt(pos);
      if (c === 10) { pos += 1; line += 1; }
      else if (c === 13 && text.charCodeAt(pos + 1) === 10) { pos += 2; line += 1; }
      else break;
    }
    if (pos >= n) break;

    recordLine = line;
    recordRow = header && columns === null ? null : rows.length;

    const fields = [];
    let ended = 1;
    let lastQuoted = false;
    do {
      let field;
      if (text.charCodeAt(pos) === 34) { field = readQuoted(); lastQuoted = true; }
      else { field = readUnquoted(''); lastQuoted = false; }
      fields.push(field[0]);
      ended = field[1];
    } while (ended === 0);
    finishRecord(fields, lastQuoted);
  }
  return { columns: header ? (columns ?? []) : null, rows, errors };
}
