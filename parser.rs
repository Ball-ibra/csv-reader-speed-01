//! csv-wasm: a WebAssembly SIMD CSV reader for make.fun's csv-reader-speed-01.
//!
//! It returns exactly what csv-reader 1.0.0 returns for every {text, header}:
//! {columns, rows, errors}, with the same broken-row policy (BOM dropped, LF/CRLF
//! records, stray_cr / quote_junk / unterminated_quote / short_row / long_row).
//!
//! Design, in short:
//! - Text is kept as UTF-8 bytes. ASCII delimiters are single bytes in UTF-8,
//!   so one SIMD step covers 16 bytes (twice the UTF-16 reach), the JSON
//!   unescape is a byte copy instead of a widen, and a field's bytes are the
//!   output's bytes: no packing step at all.
//! - Two entry points share one parser core:
//!   * csv_to_json: one CSV text (UTF-8 bytes) -> the JSON of its answer.
//!   * cases_to_answers: the whole cases.json (UTF-8 bytes) -> answers.json.
//!     The JSON strings are unescaped in here and the answers are written as
//!     JSON straight out, so the driver never JSON.parses the cases or
//!     JSON.stringifies the answers; that is where the time used to go.
//! - The parser is event-driven. One SIMD pass over the text lists every
//!   structural byte (comma, quote, LF, CR, and anything that needs JSON
//!   escaping) as an event; the parser then walks the events instead of
//!   re-scanning the text field by field. A field's bytes are copied to the
//!   output as they stand unless an event inside it says otherwise.
//! - A lone surrogate in a \uXXXX escape has no UTF-8 form; the unescape
//!   reports it and the caller falls back to the UTF-16 path for that case
//!   (the generator's vocabulary never produces one, but the fallback keeps
//!   the answer right if it ever did).

use core::arch::wasm32::*;

/// One field of a record: a span of the input text, or a range of the scratch
/// buffer where a cooked (quote-unescaped / CR-dropped / junk-appended) value
/// was materialised. `esc` says the value holds a byte that needs JSON
/// escaping (a quote, a backslash, or a control character).
#[derive(Clone, Copy)]
struct Fld {
    cooked: bool,
    esc: bool,
    a: usize,
    b: usize,
}

/// Event kinds, stored in the low 3 bits of an event word (position << 3).
const EV_COMMA: u32 = 0;
const EV_QUOTE: u32 = 1;
const EV_LF: u32 = 2;
const EV_CR: u32 = 3;
const EV_ESC: u32 = 4;

/// Write a u32 as decimal digits, no allocation, no formatting machinery.
fn push_u32(out: &mut Vec<u8>, mut v: u32) {
    if v == 0 {
        out.push(b'0');
        return;
    }
    let mut buf = [0u8; 10];
    let mut i = 10;
    while v > 0 {
        i -= 1;
        buf[i] = b'0' + (v % 10) as u8;
        v /= 10;
    }
    out.extend_from_slice(&buf[i..]);
}

/// Emit one byte with JSON escaping (for a value that needs it).
#[inline(always)]
fn emit_byte_esc(out: &mut Vec<u8>, b: u8) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    match b {
        b'"' => out.extend_from_slice(b"\\\""),
        b'\\' => out.extend_from_slice(b"\\\\"),
        b'\n' => out.extend_from_slice(b"\\n"),
        b'\r' => out.extend_from_slice(b"\\r"),
        b'\t' => out.extend_from_slice(b"\\t"),
        0x08 => out.extend_from_slice(b"\\b"),
        0x0C => out.extend_from_slice(b"\\f"),
        0x00..=0x1F => {
            out.extend_from_slice(b"\\u00");
            out.push(HEX[(b >> 4) as usize]);
            out.push(HEX[(b & 0xF) as usize]);
        }
        _ => out.push(b),
    }
}

/// Does a byte run hold anything that needs JSON escaping: a quote, a
/// backslash, or a control character (LF and CR count here too, since a
/// cooked value can hold a literal LF that must come out as \n)?
fn needs_escape(bytes: &[u8]) -> bool {
    unsafe {
        let sp_q = i8x16_splat(b'"' as i8);
        let sp_bs = i8x16_splat(b'\\' as i8);
        let sp_20 = i8x16_splat(0x20);
        let sp_0 = i8x16_splat(0i8);
        let mut i = 0;
        let n = bytes.len();
        while i + 16 <= n {
            let v = v128_load(bytes.as_ptr().add(i) as *const v128);
            // A byte under 0x20 (unsigned) is lt 0x20 and ge 0 as signed.
            let ctl = v128_and(i8x16_lt(v, sp_20), i8x16_ge(v, sp_0));
            let m = i8x16_bitmask(i8x16_eq(v, sp_q))
                | i8x16_bitmask(i8x16_eq(v, sp_bs))
                | i8x16_bitmask(ctl);
            if m != 0 {
                return true;
            }
            i += 16;
        }
        while i < n {
            let b = bytes[i];
            if b == b'"' || b == b'\\' || b < 0x20 {
                return true;
            }
            i += 1;
        }
    }
    false
}

/// Emit one JSON string value (with its quotes) for a run of UTF-8 bytes.
/// A clean run is copied as it stands; a run that needs escaping goes out
/// byte by byte.
#[inline(always)]
fn emit_str(out: &mut Vec<u8>, bytes: &[u8], esc: bool) {
    out.push(b'"');
    if esc {
        for &b in bytes {
            emit_byte_esc(out, b);
        }
    } else {
        out.extend_from_slice(bytes);
    }
    out.push(b'"');
}

/// Find a short byte needle in a haystack: SIMD scan for its first byte, then
/// a plain compare at each candidate. The needles used here (`"text"`,
/// `"header"`) cannot occur inside a JSON string value, because a literal
/// quote there is always escaped.
fn find_bytes(hay: &[u8], mut i: usize, needle: &[u8]) -> Option<usize> {
    let n = hay.len();
    if needle.is_empty() || n < needle.len() {
        return None;
    }
    let first = needle[0];
    unsafe {
        let sp = i8x16_splat(first as i8);
        while i + 16 <= n {
            let v = v128_load(hay.as_ptr().add(i) as *const v128);
            let m = i8x16_bitmask(i8x16_eq(v, sp)) as u32;
            if m != 0 {
                let mut mm = m;
                while mm != 0 {
                    let j = mm.trailing_zeros() as usize;
                    mm &= mm - 1;
                    let cand = i + j;
                    if cand + needle.len() <= n && &hay[cand..cand + needle.len()] == needle {
                        return Some(cand);
                    }
                }
            }
            i += 16;
        }
    }
    while i + needle.len() <= n {
        if hay[i] == first && &hay[i..i + needle.len()] == needle {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Skip JSON whitespace (space, tab, LF, CR) from `i`.
fn skip_ws(b: &[u8], mut i: usize) -> usize {
    while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\r') {
        i += 1;
    }
    i
}

/// List every structural byte of the text as an event word (position << 3) |
/// kind: comma, quote, LF, CR, or anything that needs JSON escaping (a
/// backslash or a control character other than LF and CR). One SIMD pass,
/// 16 bytes at a time; the scalar tail covers the last few bytes.
fn build_events(text: &[u8], n: usize, ev: &mut Vec<u32>) {
    ev.clear();
    unsafe {
        let sp_comma = i8x16_splat(b',' as i8);
        let sp_quote = i8x16_splat(b'"' as i8);
        let sp_lf = i8x16_splat(b'\n' as i8);
        let sp_cr = i8x16_splat(b'\r' as i8);
        let sp_bs = i8x16_splat(b'\\' as i8);
        let sp_20 = i8x16_splat(0x20);
        let sp_0 = i8x16_splat(0i8);
        let mut i = 0;
        while i + 16 <= n {
            let v = v128_load(text.as_ptr().add(i) as *const v128);
            let m_comma = i8x16_bitmask(i8x16_eq(v, sp_comma)) as u32;
            let m_quote = i8x16_bitmask(i8x16_eq(v, sp_quote)) as u32;
            let m_lf = i8x16_bitmask(i8x16_eq(v, sp_lf)) as u32;
            let m_cr = i8x16_bitmask(i8x16_eq(v, sp_cr)) as u32;
            let m_bs = i8x16_bitmask(i8x16_eq(v, sp_bs)) as u32;
            // Control bytes other than LF and CR: under 0x20 unsigned, which
            // is lt 0x20 and ge 0 as signed, minus the LF and CR lanes.
            let m_ctl = (i8x16_bitmask(v128_and(i8x16_lt(v, sp_20), i8x16_ge(v, sp_0))) as u32)
                & !m_lf
                & !m_cr;
            let m_esc = m_bs | m_ctl;
            let mut m = m_comma | m_quote | m_lf | m_cr | m_esc;
            while m != 0 {
                let j = m.trailing_zeros() as usize;
                m &= m - 1;
                let bit = 1u32 << j;
                let kind = if m_comma & bit != 0 {
                    EV_COMMA
                } else if m_quote & bit != 0 {
                    EV_QUOTE
                } else if m_lf & bit != 0 {
                    EV_LF
                } else if m_cr & bit != 0 {
                    EV_CR
                } else {
                    EV_ESC
                };
                ev.push((((i + j) as u32) << 3) | kind);
            }
            i += 16;
        }
        while i < n {
            let b = text[i];
            let kind = match b {
                b',' => Some(EV_COMMA),
                b'"' => Some(EV_QUOTE),
                b'\n' => Some(EV_LF),
                b'\r' => Some(EV_CR),
                b'\\' => Some(EV_ESC),
                0x00..=0x1F => Some(EV_ESC),
                _ => None,
            };
            if let Some(k) = kind {
                ev.push(((i as u32) << 3) | k);
            }
            i += 1;
        }
    }
}

/// Write one Unicode code point as UTF-8 at base+*dst, advancing *dst.
#[inline(always)]
unsafe fn write_utf8(base: *mut u8, dst: &mut usize, cp: u32) {
    if cp < 0x80 {
        *base.add(*dst) = cp as u8;
        *dst += 1;
    } else if cp < 0x800 {
        *base.add(*dst) = 0xC0 | ((cp >> 6) as u8);
        *base.add(*dst + 1) = 0x80 | ((cp & 0x3F) as u8);
        *dst += 2;
    } else if cp < 0x10000 {
        *base.add(*dst) = 0xE0 | ((cp >> 12) as u8);
        *base.add(*dst + 1) = 0x80 | (((cp >> 6) & 0x3F) as u8);
        *base.add(*dst + 2) = 0x80 | ((cp & 0x3F) as u8);
        *dst += 3;
    } else {
        *base.add(*dst) = 0xF0 | ((cp >> 18) as u8);
        *base.add(*dst + 1) = 0x80 | (((cp >> 12) & 0x3F) as u8);
        *base.add(*dst + 2) = 0x80 | (((cp >> 6) & 0x3F) as u8);
        *base.add(*dst + 3) = 0x80 | ((cp & 0x3F) as u8);
        *dst += 4;
    }
}

/// Parse a JSON string starting at bytes[start] == b'"'. The unescaped UTF-8
/// bytes are written to `text` (cleared first); 16 zero bytes are left after
/// the content so the scanner can read a little past the end. Returns the
/// index just past the closing quote and the number of content bytes. A lone
/// surrogate in a \uXXXX escape has no UTF-8 form: it comes out as U+FFFD
/// and sets `lone`, so the caller can fall back for that case.
///
/// The string is walked in 16-byte chunks. A chunk with no special byte (a
/// quote, a backslash, or a byte >= 0x80) is copied with a single SIMD store.
/// A chunk that holds a special byte is walked byte by byte, decoding the
/// escapes it holds. Most bytes live in clean chunks, so most of the copy is
/// one SIMD store per 16 bytes.
fn parse_json_string(
    text: &mut Vec<u8>,
    bytes: &[u8],
    start: usize,
    lone: &mut bool,
) -> (usize, usize) {
    text.clear();
    text.reserve(bytes.len() + 16);
    let n = bytes.len();
    let mut i = start + 1;
    let mut dst = 0usize;
    unsafe {
        let base = text.as_mut_ptr();
        let src = bytes.as_ptr();
        let sp_q = i8x16_splat(b'"' as i8);
        let sp_bs = i8x16_splat(b'\\' as i8);
        let sp_0 = i8x16_splat(0i8);
        while i + 16 <= n {
            let v = v128_load(src.add(i) as *const v128);
            let m = (i8x16_bitmask(i8x16_eq(v, sp_q))
                | i8x16_bitmask(i8x16_eq(v, sp_bs))
                | i8x16_bitmask(i8x16_lt(v, sp_0))) as u32;
            if m == 0 {
                v128_store(base.add(dst) as *mut v128, v);
                dst += 16;
                i += 16;
                continue;
            }
            // This chunk holds a special byte. Walk it byte by byte.
            let end = i + 16;
            while i < end {
                let b = *src.add(i);
                if b == b'"' {
                    for k in 0..16 {
                        *base.add(dst + k) = 0;
                    }
                    text.set_len(dst + 16);
                    return (i + 1, dst);
                }
                if b == b'\\' {
                    if i + 1 >= n {
                        i = end;
                        break;
                    }
                    match *src.add(i + 1) {
                        b'"' => {
                            *base.add(dst) = b'"';
                            dst += 1;
                            i += 2;
                        }
                        b'\\' => {
                            *base.add(dst) = b'\\';
                            dst += 1;
                            i += 2;
                        }
                        b'/' => {
                            *base.add(dst) = b'/';
                            dst += 1;
                            i += 2;
                        }
                        b'b' => {
                            *base.add(dst) = 0x08;
                            dst += 1;
                            i += 2;
                        }
                        b'f' => {
                            *base.add(dst) = 0x0C;
                            dst += 1;
                            i += 2;
                        }
                        b'n' => {
                            *base.add(dst) = b'\n';
                            dst += 1;
                            i += 2;
                        }
                        b'r' => {
                            *base.add(dst) = b'\r';
                            dst += 1;
                            i += 2;
                        }
                        b't' => {
                            *base.add(dst) = b'\t';
                            dst += 1;
                            i += 2;
                        }
                        b'u' => {
                            if i + 5 >= n {
                                i = end;
                                break;
                            }
                            let mut v: u32 = 0;
                            for k in 0..4 {
                                let h = *src.add(i + 2 + k);
                                let d = match h {
                                    b'0'..=b'9' => (h - b'0') as u32,
                                    b'a'..=b'f' => (h - b'a' + 10) as u32,
                                    b'A'..=b'F' => (h - b'A' + 10) as u32,
                                    _ => 0,
                                };
                                v = (v << 4) | d;
                            }
                            if (0xD800..0xDC00).contains(&v) {
                                let mut paired = false;
                                if i + 11 < n
                                    && *src.add(i + 6) == b'\\'
                                    && *src.add(i + 7) == b'u'
                                {
                                    let mut v2: u32 = 0;
                                    for k in 0..4 {
                                        let h = *src.add(i + 8 + k);
                                        let d = match h {
                                            b'0'..=b'9' => (h - b'0') as u32,
                                            b'a'..=b'f' => (h - b'a' + 10) as u32,
                                            b'A'..=b'F' => (h - b'A' + 10) as u32,
                                            _ => 0,
                                        };
                                        v2 = (v2 << 4) | d;
                                    }
                                    if (0xDC00..0xE000).contains(&v2) {
                                        let cp =
                                            0x10000 + ((v - 0xD800) << 10) + (v2 - 0xDC00);
                                        write_utf8(base, &mut dst, cp);
                                        i += 12;
                                        paired = true;
                                    }
                                }
                                if !paired {
                                    *lone = true;
                                    write_utf8(base, &mut dst, 0xFFFD);
                                    i += 6;
                                }
                            } else if (0xDC00..0xE000).contains(&v) {
                                *lone = true;
                                write_utf8(base, &mut dst, 0xFFFD);
                                i += 6;
                            } else {
                                write_utf8(base, &mut dst, v);
                                i += 6;
                            }
                        }
                        _ => {
                            *base.add(dst) = *src.add(i + 1);
                            dst += 1;
                            i += 2;
                        }
                    }
                    continue;
                }
                if b >= 0x80 {
                    let len = if b >= 0xF0 {
                        4
                    } else if b >= 0xE0 {
                        3
                    } else {
                        2
                    };
                    // Copy the whole sequence even if it runs past this
                    // chunk's end; a chunk end is a processing boundary, not
                    // a data boundary, and capping here would split it.
                    let len = len.min(n - i);
                    core::ptr::copy_nonoverlapping(src.add(i), base.add(dst), len);
                    dst += len;
                    i += len;
                    continue;
                }
                *base.add(dst) = b;
                dst += 1;
                i += 1;
            }
        }
        // The tail (fewer than 16 bytes left), byte by byte.
        while i < n {
            let b = *src.add(i);
            if b == b'"' {
                for k in 0..16 {
                    *base.add(dst + k) = 0;
                }
                text.set_len(dst + 16);
                return (i + 1, dst);
            }
            if b == b'\\' {
                if i + 1 >= n {
                    break;
                }
                match *src.add(i + 1) {
                    b'"' => {
                        *base.add(dst) = b'"';
                        dst += 1;
                    }
                    b'\\' => {
                        *base.add(dst) = b'\\';
                        dst += 1;
                    }
                    b'/' => {
                        *base.add(dst) = b'/';
                        dst += 1;
                    }
                    b'b' => {
                        *base.add(dst) = 0x08;
                        dst += 1;
                    }
                    b'f' => {
                        *base.add(dst) = 0x0C;
                        dst += 1;
                    }
                    b'n' => {
                        *base.add(dst) = b'\n';
                        dst += 1;
                    }
                    b'r' => {
                        *base.add(dst) = b'\r';
                        dst += 1;
                    }
                    b't' => {
                        *base.add(dst) = b'\t';
                        dst += 1;
                    }
                    b'u' => {
                        if i + 5 >= n {
                            break;
                        }
                        let mut v: u32 = 0;
                        for k in 0..4 {
                            let h = *src.add(i + 2 + k);
                            let d = match h {
                                b'0'..=b'9' => (h - b'0') as u32,
                                b'a'..=b'f' => (h - b'a' + 10) as u32,
                                b'A'..=b'F' => (h - b'A' + 10) as u32,
                                _ => 0,
                            };
                            v = (v << 4) | d;
                        }
                        if (0xD800..0xDC00).contains(&v) {
                            let mut paired = false;
                            if i + 11 < n
                                && *src.add(i + 6) == b'\\'
                                && *src.add(i + 7) == b'u'
                            {
                                let mut v2: u32 = 0;
                                for k in 0..4 {
                                    let h = *src.add(i + 8 + k);
                                    let d = match h {
                                        b'0'..=b'9' => (h - b'0') as u32,
                                        b'a'..=b'f' => (h - b'a' + 10) as u32,
                                        b'A'..=b'F' => (h - b'A' + 10) as u32,
                                        _ => 0,
                                    };
                                    v2 = (v2 << 4) | d;
                                }
                                if (0xDC00..0xE000).contains(&v2) {
                                    let cp = 0x10000 + ((v - 0xD800) << 10) + (v2 - 0xDC00);
                                    write_utf8(base, &mut dst, cp);
                                    i += 12;
                                    paired = true;
                                }
                            }
                            if !paired {
                                *lone = true;
                                write_utf8(base, &mut dst, 0xFFFD);
                                i += 6;
                            }
                            continue;
                        }
                        if (0xDC00..0xE000).contains(&v) {
                            *lone = true;
                            write_utf8(base, &mut dst, 0xFFFD);
                            i += 6;
                            continue;
                        }
                        write_utf8(base, &mut dst, v);
                        i += 6;
                        continue;
                    }
                    _ => {
                        *base.add(dst) = *src.add(i + 1);
                        dst += 1;
                    }
                }
                i += 2;
                continue;
            }
            if b >= 0x80 {
                let len = if b >= 0xF0 {
                    4
                } else if b >= 0xE0 {
                    3
                } else {
                    2
                };
                let len = len.min(n - i);
                core::ptr::copy_nonoverlapping(src.add(i), base.add(dst), len);
                dst += len;
                i += len;
                continue;
            }
            *base.add(dst) = b;
            dst += 1;
            i += 1;
        }
        for k in 0..16 {
            *base.add(dst + k) = 0;
        }
        text.set_len(dst + 16);
        (n, dst)
    }
}

/// The module's working state. WebAssembly here is single-threaded, so one
/// static instance is enough; every buffer is reused from case to case.
pub struct State {
    input: Vec<u8>,
    text: Vec<u8>,
    text_len: usize,
    ev: Vec<u32>,
    scratch: Vec<u8>,
    col: Vec<u8>,
    rows: Vec<u8>,
    errs: Vec<u8>,
    stage: Vec<u8>,
    out: Vec<u8>,
    ans: Vec<u8>,
    fields: Vec<Fld>,
    lone: bool,
    // Parse state, reset for every csv_answer call.
    ei: usize,
    pos: usize,
    line: u32,
    rec_line: u32,
    rec_row: Option<u32>,
    width: i32,
    rows_n: u32,
    cols_done: bool,
    row_first: bool,
    err_first: bool,
    header: bool,
}

/// Emit a comma-separated list of field JSON strings (no brackets) for
/// fields[from..to]. A cooked field's escaping is decided by scanning its
/// bytes; a span field carries the flag the parser tracked while walking its
/// events.
fn emit_field_range(
    text: &[u8],
    scratch: &[u8],
    out: &mut Vec<u8>,
    fields: &[Fld],
    from: usize,
    to: usize,
) {
    for i in from..to {
        if i > from {
            out.push(b',');
        }
        let f = fields[i];
        if f.cooked {
            let bytes = &scratch[f.a..f.b];
            emit_str(out, bytes, needs_escape(bytes));
        } else {
            emit_str(out, &text[f.a..f.b], f.esc);
        }
    }
}

/// Emit a comma-separated list of field JSON strings (no brackets) for the
/// first `count` fields.
fn emit_field_list(
    text: &[u8],
    scratch: &[u8],
    out: &mut Vec<u8>,
    fields: &[Fld],
    count: usize,
) {
    emit_field_range(text, scratch, out, fields, 0, count);
}

impl State {
    fn new() -> Self {
        State {
            input: Vec::new(),
            text: Vec::new(),
            text_len: 0,
            ev: Vec::new(),
            scratch: Vec::new(),
            col: Vec::new(),
            rows: Vec::new(),
            errs: Vec::new(),
            stage: Vec::new(),
            out: Vec::new(),
            ans: Vec::new(),
            fields: Vec::new(),
            lone: false,
            ei: 0,
            pos: 0,
            line: 1,
            rec_line: 1,
            rec_row: None,
            width: -1,
            rows_n: 0,
            cols_done: false,
            row_first: true,
            err_first: true,
            header: false,
        }
    }

    /// The next event at or after the current position, if any. Events are
    /// consumed in order and positions only move forward, so this is the
    /// first event the parser has not dealt with yet.
    fn peek_event(&self) -> Option<(usize, u32)> {
        if self.ei < self.ev.len() {
            let w = self.ev[self.ei];
            Some(((w >> 3) as usize, w & 7))
        } else {
            None
        }
    }

    /// Move the event cursor to the first event at or after `pos`.
    fn rewind_ei(&mut self, pos: usize) {
        self.ei = self.ev.partition_point(|&w| ((w >> 3) as usize) < pos);
    }

    /// Record an error for the record being read: {"code","line","row"}.
    fn err_prefix(&mut self, code: &str) {
        if !self.err_first {
            self.errs.push(b',');
        }
        self.err_first = false;
        self.errs.extend_from_slice(b"{\"code\":\"");
        self.errs.extend_from_slice(code.as_bytes());
        self.errs.extend_from_slice(b"\",\"line\":");
        push_u32(&mut self.errs, self.rec_line);
        self.errs.extend_from_slice(b",\"row\":");
        match self.rec_row {
            Some(r) => push_u32(&mut self.errs, r),
            None => self.errs.extend_from_slice(b"null"),
        }
    }

    fn err(&mut self, code: &str) {
        self.err_prefix(code);
        self.errs.push(b'}');
    }

    /// Materialise a quoted field's content (open+1 .. close) into the scratch
    /// buffer: "" becomes one quote, and each CRLF or lone CR inside becomes
    /// one LF, counting one line each. Only used when the content actually
    /// holds one of those; the common case stays a plain span.
    fn cook_quoted(&mut self, open: usize, close: usize) -> (usize, usize) {
        let start = self.scratch.len();
        let mut i = open + 1;
        while i < close {
            let b = self.text[i];
            if b == b'"' {
                self.scratch.push(b'"');
                i += 2;
            } else if b == b'\r' {
                self.scratch.push(b'\n');
                self.line += 1;
                i += if self.text[i + 1] == b'\n' { 2 } else { 1 };
            } else if b == b'\n' {
                self.scratch.push(b'\n');
                self.line += 1;
                i += 1;
            } else {
                self.scratch.push(b);
                i += 1;
            }
        }
        (start, self.scratch.len())
    }

    /// Finish a field that ends at `end`, starting its last segment at `seg`.
    /// A field that met a stray CR was cooked into the scratch buffer; a clean
    /// one stays a span of the text.
    fn finish_field(
        &mut self,
        cooked: Option<usize>,
        seg: usize,
        end: usize,
        added: &mut usize,
        esc: bool,
    ) -> Fld {
        match cooked {
            Some(cs) => {
                self.scratch.extend_from_slice(&self.text[seg..end]);
                *added += end - seg;
                Fld {
                    cooked: true,
                    esc: false,
                    a: cs,
                    b: self.scratch.len(),
                }
            }
            None => Fld {
                cooked: false,
                esc,
                a: seg,
                b: end,
            },
        }
    }
}

impl State {
    /// Read an unquoted field starting at self.pos, driven by the event list.
    /// Returns the field, a terminator code (0 comma, 1 line break, 2 end of
    /// text), and the number of bytes kept. With a prefix (junk after a
    /// closing quote) the field is cooked into the scratch buffer and the
    /// kept count decides whether quote_junk applies. A quote or backslash
    /// inside an unquoted field is literal content; a stray CR is dropped and
    /// reported; CR LF ends the record as one line break.
    fn scan_unquoted(&mut self, prefix: Option<Fld>) -> (Fld, u8, usize) {
        let mut cooked: Option<usize> = None;
        if let Some(p) = prefix {
            if p.cooked {
                cooked = Some(p.a);
            } else {
                let cs = self.scratch.len();
                self.scratch.extend_from_slice(&self.text[p.a..p.b]);
                cooked = Some(cs);
            }
        }
        let mut added = 0usize;
        let mut seg = self.pos;
        let n = self.text_len;
        let mut esc = false;
        loop {
            match self.peek_event() {
                None => {
                    let f = self.finish_field(cooked, seg, n, &mut added, esc);
                    self.pos = n;
                    return (f, 2, added);
                }
                Some((epos, ekind)) => match ekind {
                    EV_COMMA => {
                        let f = self.finish_field(cooked, seg, epos, &mut added, esc);
                        self.pos = epos + 1;
                        self.ei += 1;
                        return (f, 0, added);
                    }
                    EV_LF => {
                        let f = self.finish_field(cooked, seg, epos, &mut added, esc);
                        self.pos = epos + 1;
                        self.line += 1;
                        self.ei += 1;
                        return (f, 1, added);
                    }
                    EV_CR => {
                        if self.text[epos + 1] == b'\n' {
                            let f = self.finish_field(cooked, seg, epos, &mut added, esc);
                            self.pos = epos + 2;
                            self.line += 1;
                            self.ei += 2;
                            return (f, 1, added);
                        }
                        self.err("stray_cr");
                        if cooked.is_none() {
                            cooked = Some(self.scratch.len());
                        }
                        self.scratch.extend_from_slice(&self.text[seg..epos]);
                        added += epos - seg;
                        seg = epos + 1;
                        self.ei += 1;
                    }
                    EV_QUOTE | EV_ESC => {
                        // Literal content of an unquoted field.
                        esc = true;
                        self.ei += 1;
                    }
                    _ => unreachable!(),
                },
            }
        }
    }

    /// Read a quoted field starting at self.pos (which holds a `"`).
    /// Returns the field and a terminator code (0 comma, 1 line break, 2 end
    /// of text). The walk finds the closing quote by doubling: a quote
    /// followed by another quote is an escaped quote, not a close, so both
    /// events of the pair are consumed. A field with no closing quote is
    /// re-read as an unquoted field starting just after the opening quote,
    /// with the opening quote kept as a literal prefix.
    fn read_quoted(&mut self) -> (Fld, u8) {
        let open = self.pos;
        let open_line = self.line;
        let n = self.text_len;
        // The event cursor sits on the opening quote itself; consume it so
        // the walk below only sees events inside the field.
        self.ei += 1;
        let mut crlf = false;
        let mut escaped = false;
        let mut esc = false;
        let mut close: Option<usize> = None;
        while let Some((epos, ekind)) = self.peek_event() {
            match ekind {
                EV_QUOTE => {
                    if self.text[epos + 1] == b'"' {
                        escaped = true;
                        self.ei += 2;
                        continue;
                    }
                    close = Some(epos);
                    self.ei += 1;
                    break;
                }
                EV_LF | EV_CR => {
                    crlf = true;
                    self.ei += 1;
                }
                EV_ESC => {
                    esc = true;
                    self.ei += 1;
                }
                EV_COMMA => {
                    self.ei += 1;
                }
                _ => unreachable!(),
            }
        }
        let close = match close {
            Some(c) => c,
            None => {
                self.err("unterminated_quote");
                self.pos = open + 1;
                self.line = open_line;
                self.rewind_ei(open + 1);
                let cs = self.scratch.len();
                self.scratch.push(b'"');
                let (f, term, _) = self.scan_unquoted(Some(Fld {
                    cooked: true,
                    esc: false,
                    a: cs,
                    b: cs + 1,
                }));
                return (f, term);
            }
        };
        let fld = if !crlf && !escaped {
            Fld {
                cooked: false,
                esc,
                a: open + 1,
                b: close,
            }
        } else {
            let (a, b) = self.cook_quoted(open, close);
            Fld {
                cooked: true,
                esc: false,
                a,
                b,
            }
        };
        let after = close + 1;
        if after >= n {
            self.pos = n;
            return (fld, 2);
        }
        let c = self.text[after];
        if c == b',' {
            self.pos = after + 1;
            self.ei += 1;
            return (fld, 0);
        }
        if c == b'\n' {
            self.pos = after + 1;
            self.line += 1;
            self.ei += 1;
            return (fld, 1);
        }
        if c == b'\r' && self.text[after + 1] == b'\n' {
            self.pos = after + 2;
            self.line += 1;
            self.ei += 2;
            return (fld, 1);
        }
        // Junk after the closing quote: it joins the field as literal text,
        // and anything at all kept makes it a quote_junk error.
        self.pos = after;
        let (f, term, added) = self.scan_unquoted(Some(fld));
        if added > 0 {
            self.err("quote_junk");
        }
        (f, term)
    }

    /// Parse the CSV text in self.text (length self.text_len) for one
    /// {text, header} case and write the JSON of its answer to self.out.
    /// The events are built first; the parser then walks them, so each byte
    /// of the text is looked at once.
    fn csv_answer(&mut self, header: bool) {
        self.out.clear();
        self.scratch.clear();
        self.col.clear();
        self.rows.clear();
        self.errs.clear();
        self.fields.clear();
        // The answer is about the size of the text; reserving up front keeps
        // the output buffers from reallocating and copying as they grow.
        let n0 = self.text_len;
        self.rows.reserve(n0);
        self.out.reserve(n0);
        self.ev.reserve(n0 / 8);
        build_events(&self.text, self.text_len, &mut self.ev);
        let n = self.text_len;
        self.pos = if n >= 3 && self.text[0] == 0xEF && self.text[1] == 0xBB && self.text[2] == 0xBF {
            3
        } else {
            0
        };
        self.line = 1;
        self.width = -1;
        self.rows_n = 0;
        self.cols_done = false;
        self.row_first = true;
        self.err_first = true;
        self.header = header;
        self.rewind_ei(self.pos);
        loop {
            // Skip empty lines.
            while self.pos < n {
                let c = self.text[self.pos];
                if c == b'\n' {
                    self.pos += 1;
                    self.line += 1;
                    self.ei += 1;
                } else if c == b'\r' && self.text[self.pos + 1] == b'\n' {
                    self.pos += 2;
                    self.line += 1;
                    self.ei += 2;
                } else {
                    break;
                }
            }
            if self.pos >= n {
                break;
            }
            self.rec_line = self.line;
            // The header record (header is true and no columns yet) has a
            // null row; every other record takes its index in rows. This is
            // set before the fields are read, so errors met mid-record
            // (stray_cr, quote_junk) carry the right row.
            self.rec_row = if self.header && !self.cols_done {
                None
            } else {
                Some(self.rows_n)
            };
            self.fields.clear();
            let mut term = 2u8;
            loop {
                let (f, t) = if self.text[self.pos] == b'"' {
                    self.read_quoted()
                } else {
                    let (f, t, _) = self.scan_unquoted(None);
                    (f, t)
                };
                self.fields.push(f);
                term = t;
                if t != 0 {
                    break;
                }
            }
            if self.fields.len() == 1 {
                let f = self.fields[0];
                let empty = if f.cooked {
                    f.a == f.b
                } else {
                    f.a == f.b
                };
                if empty {
                    continue;
                }
            }
            if !self.cols_done {
                self.cols_done = true;
                if header {
                    self.col.push(b'[');
                    emit_field_list(
                        &self.text,
                        &self.scratch,
                        &mut self.col,
                        &self.fields,
                        self.fields.len(),
                    );
                    self.col.push(b']');
                    self.width = self.fields.len() as i32;
                    continue;
                }
                // header is false: columns stays null, but the first record
                // still sets the width.
                self.width = self.fields.len() as i32;
            }
            if (self.fields.len() as i32) < self.width {
                self.err_prefix("short_row");
                self.errs.extend_from_slice(b",\"fields\":");
                push_u32(&mut self.errs, self.fields.len() as u32);
                self.errs.push(b'}');
            } else if (self.fields.len() as i32) > self.width {
                self.err_prefix("long_row");
                self.errs.extend_from_slice(b",\"extra\":[");
                emit_field_range(
                    &self.text,
                    &self.scratch,
                    &mut self.errs,
                    &self.fields,
                    self.width as usize,
                    self.fields.len(),
                );
                self.errs.extend_from_slice(b"]}");
            }
            if !self.row_first {
                self.rows.push(b',');
            }
            self.row_first = false;
            self.rows.push(b'[');
            let take = self.fields.len().min(self.width as usize);
            emit_field_list(
                &self.text,
                &self.scratch,
                &mut self.rows,
                &self.fields,
                take,
            );
            for _ in take..self.width as usize {
                self.rows.extend_from_slice(b",\"\"");
            }
            self.rows.push(b']');
            self.rows_n += 1;
            if term == 2 {
                break;
            }
        }
        self.out.extend_from_slice(b"{\"columns\":");
        if self.col.is_empty() {
            // No header record was read: [] when a header was expected,
            // null when it was not.
            if self.header {
                self.out.extend_from_slice(b"[]");
            } else {
                self.out.extend_from_slice(b"null");
            }
        } else {
            self.out.extend_from_slice(&self.col);
        }
        self.out.extend_from_slice(b",\"rows\":[");
        self.out.extend_from_slice(&self.rows);
        self.out.extend_from_slice(b"],\"errors\":[");
        self.out.extend_from_slice(&self.errs);
        self.out.extend_from_slice(b"]}");
    }

    /// Transduce the whole cases.json (UTF-8 bytes) into answers.json, in
    /// one pass. Each case's text is unescaped here and parsed straight from
    /// the UTF-8 bytes; the answers are written as JSON as they are produced.
    /// The array is built in self.ans and moved into self.out at the end.
    /// Returns false if a lone surrogate was seen (the caller falls back for
    /// that case, since a lone surrogate has no UTF-8 form).
    fn cases_to_answers(&mut self, bytes: &[u8]) -> bool {
        self.ans.clear();
        self.ans.extend_from_slice(b"[");
        let n = bytes.len();
        let mut i = 0usize;
        let mut first = true;
        loop {
            let t = find_bytes(bytes, i, b"\"text\"");
            let t = match t {
                Some(v) => v,
                None => break,
            };
            let mut j = skip_ws(bytes, t + 6);
            if j >= n || bytes[j] != b':' {
                break;
            }
            j = skip_ws(bytes, j + 1);
            if j >= n || bytes[j] != b'"' {
                break;
            }
            let (end, len) = parse_json_string(&mut self.text, bytes, j, &mut self.lone);
            self.text_len = len;
            i = end;
            let h = find_bytes(bytes, i, b"\"header\"");
            let h = match h {
                Some(v) => v,
                None => break,
            };
            let mut k = skip_ws(bytes, h + 8);
            if k >= n || bytes[k] != b':' {
                break;
            }
            k = skip_ws(bytes, k + 1);
            let header = k < n && bytes[k] == b't';
            if !first {
                self.ans.push(b',');
            }
            first = false;
            self.csv_answer(header);
            self.ans.extend_from_slice(&self.out);
            // Find the end of this case object so the next search starts
            // after it. The text and header keys were found in order, so the
            // next case starts after the header value.
            let mut m = k;
            while m < n && bytes[m] != b'}' {
                m += 1;
            }
            i = m + 1;
        }
        self.ans.push(b']');
        self.out = core::mem::take(&mut self.ans);
        !self.lone
    }
}

static mut STATE: Option<State> = None;

fn state() -> &'static mut State {
    unsafe {
        if STATE.is_none() {
            STATE = Some(State::new());
        }
        STATE.as_mut().unwrap()
    }
}

/// Reserve `len` bytes in the input scratch buffer and return a pointer to
/// them. The caller writes the input there, then calls csv_to_json or
/// cases_to_answers with the same pointer and length.
#[no_mangle]
pub extern "C" fn input_ptr(len: usize) -> usize {
    let st = state();
    st.input.clear();
    st.input.reserve(len);
    st.input.as_mut_ptr() as usize
}

/// A pointer to the last answer written by csv_to_json or cases_to_answers.
#[no_mangle]
pub extern "C" fn out_ptr() -> usize {
    state().out.as_ptr() as usize
}

/// One CSV text (UTF-8 bytes) -> the JSON of its answer. The `header`
/// argument is 1 for a header row, 0 otherwise. Returns a pointer into the
/// module's memory holding the answer bytes; the length is read with out_len.
/// The text is copied once into the state's text buffer (with padding) and
/// parsed there; the caller's buffer is only read, never written.
#[no_mangle]
pub extern "C" fn csv_to_json(ptr: usize, len: usize, header: u32) -> usize {
    let st = state();
    let bytes = unsafe { core::slice::from_raw_parts(ptr as *const u8, len) };
    st.text.clear();
    st.text.reserve(len + 16);
    st.text.extend_from_slice(bytes);
    for _ in 0..16 {
        st.text.push(0);
    }
    st.text_len = len;
    st.lone = false;
    st.csv_answer(header != 0);
    st.out.as_ptr() as usize
}

/// The whole cases.json (UTF-8 bytes) -> answers.json. Returns a pointer to
/// the answer bytes; the length is read with out_len. Returns 0 if a lone
/// surrogate was seen anywhere (the caller falls back, per case). The input
/// is read in place from the caller's buffer; only the unescaped texts and
/// the answers are copied.
#[no_mangle]
pub extern "C" fn cases_to_answers(ptr: usize, len: usize) -> usize {
    let st = state();
    let bytes = unsafe { core::slice::from_raw_parts(ptr as *const u8, len) };
    st.lone = false;
    if !st.cases_to_answers(bytes) {
        return 0;
    }
    st.out.as_ptr() as usize
}

/// The length of the last answer written by csv_to_json or cases_to_answers.
#[no_mangle]
pub extern "C" fn out_len() -> usize {
    state().out.len()
}

/// 1 if the last unescape saw a lone surrogate (so its answer is not to be
/// trusted), 0 otherwise.
#[no_mangle]
pub extern "C" fn had_lone_surrogate() -> u32 {
    state().lone as u32
}

/// Benchmark only: unescape every "text" string in the cases.json at ptr,
/// without parsing any CSV. Returns the total unescaped text bytes. Used to
/// see how much of the transduce time is the JSON unescape.
#[no_mangle]
pub extern "C" fn bench_unescape(ptr: usize, len: usize) -> usize {
    let st = state();
    let bytes = unsafe { core::slice::from_raw_parts(ptr as *const u8, len) };
    let n = bytes.len();
    let mut i = 0usize;
    let mut total = 0usize;
    st.lone = false;
    loop {
        let t = match find_bytes(bytes, i, b"\"text\"") {
            Some(v) => v,
            None => break,
        };
        let mut j = skip_ws(bytes, t + 6);
        if j >= n || bytes[j] != b':' {
            break;
        }
        j = skip_ws(bytes, j + 1);
        if j >= n || bytes[j] != b'"' {
            break;
        }
        let (end, tlen) = parse_json_string(&mut st.text, bytes, j, &mut st.lone);
        total += tlen;
        i = end;
    }
    total
}

/// Benchmark only: unescape every "text" string and build its event list,
/// without parsing any CSV. Returns the total number of events. Used to see
/// how much of the transduce time is the unescape plus the event scan.
#[no_mangle]
pub extern "C" fn bench_events(ptr: usize, len: usize) -> usize {
    let st = state();
    let bytes = unsafe { core::slice::from_raw_parts(ptr as *const u8, len) };
    let n = bytes.len();
    let mut i = 0usize;
    let mut total = 0usize;
    st.lone = false;
    loop {
        let t = match find_bytes(bytes, i, b"\"text\"") {
            Some(v) => v,
            None => break,
        };
        let mut j = skip_ws(bytes, t + 6);
        if j >= n || bytes[j] != b':' {
            break;
        }
        j = skip_ws(bytes, j + 1);
        if j >= n || bytes[j] != b'"' {
            break;
        }
        let (end, tlen) = parse_json_string(&mut st.text, bytes, j, &mut st.lone);
        st.text_len = tlen;
        build_events(&st.text, tlen, &mut st.ev);
        total += st.ev.len();
        i = end;
    }
    total
}
