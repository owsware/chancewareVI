//! `SGD5`: the engine's shader database. Shader objects, each holding one
//! deflate-packed shader container, and effects whose passes combine them.
//! Little-endian; `str` is a u8 length (with the NUL), the bytes, a NUL:
//!
//! ```text
//! file:    "SGD5", u64, u32 n, n × { u32 size, size bytes }  shared parameter blocks
//!          u32 size, size bytes                              string table
//!          6 × { u32 n, n × shader }                         one list per stage slot
//!          u32 n, n × effect
//! shader:  str name, u8 kind, u32 n, n bytes record,
//!          u64 hash, 0x90 bytes, u32 m, m bytes
//! record:  u32 0x000DDCE7, u32 n, u32 unpacked size, raw deflate (n - 12 bytes)
//! effect:  str name, u32 size once loaded, 6 × { u32 n, n × u32 shader index },
//!          0x50 bytes, u32 n, n × 16 B rasterizer, u32 n, n × 24 B depth-stencil,
//!          u32 n, n × 68 B blend, u32 n, n × 36 B sampler, u32 n, n × u32 technique,
//!          u32 n, n × 24 B pass, u32 n, n × u32, u32 strings (0x8000_0000: the
//!          file's table), u32 n, n B, u32 n, n B, u32 n (bits 0-23), n × block,
//!          u32 n, n B technique names
//! block:   u32 size, size bytes | u32 0x8000_0000, u32 hash of a shared block
//! technique: bits 0-14 name offset, 15-23 first pass, 24-31 pass count
//! pass:    +0..+5 1-based index into the effect's list of each stage slot
//!          (0 none), +6/+7/+8 rasterizer, depth-stencil, blend index, +9
//!          stencil ref, +0xA flags: bits 0-3 which of those four the pass
//!          sets, bit 6/7 bit 8 of the depth-stencil/blend index
//! ```
//!
//! A pass that does not set a state draws with whatever state is current.

use super::{MAGIC, MARKER, MAX_UNPACKED, Record};

const SHARED: u32 = 0x8000_0000;

/// The stage slots, in file order. The vertex slot also holds mesh shaders.
/// Slots 3 and 4 (tessellation) are empty in every database seen; which of
/// them is hull and which domain is not established.
pub const STAGES: [&str; 6] = ["vertex", "pixel", "geometry", "slot3", "slot4", "compute"];

/// A shader object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shader<'a> {
    pub record: Record<'a>,
    /// The engine's identity of the shader: programs (and the title's
    /// pipeline cache) are keyed by these.
    pub hash: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Database<'a> {
    /// The shader objects of each stage slot (`STAGES`), in file order.
    pub shaders: [Vec<Shader<'a>>; 6],
    pub effects: Vec<Effect<'a>>,
    /// File offsets of the magic and of the byte after the last effect.
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Effect<'a> {
    pub name: &'a str,
    pub techniques: Vec<Technique<'a>>,
    /// Every state description the effect holds (its passes pick from
    /// these), as compact descriptions.
    pub rasterizers: Vec<&'a [u8]>,
    pub depth_stencils: Vec<&'a [u8]>,
    pub blends: Vec<&'a [u8]>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Technique<'a> {
    pub name: &'a str,
    pub passes: Vec<Pass<'a>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pass<'a> {
    /// Per stage slot, an index into `Database::shaders` of that slot.
    pub shaders: [Option<usize>; 6],
    /// The state the pass sets, as the engine's compact descriptions
    /// (D3D12 enum values); `None`: the engine's current state.
    pub rasterizer: Option<&'a [u8]>,
    pub depth_stencil: Option<&'a [u8]>,
    pub blend: Option<&'a [u8]>,
    pub stencil_ref: Option<u8>,
}

/// Every database whose tables parse, in file order.
pub fn databases(file: &[u8]) -> Vec<Database<'_>> {
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(at) = memchr::memmem::find(&file[from..], MAGIC).map(|p| from + p) {
        match parse(file, at) {
            Some(db) => {
                from = db.end;
                out.push(db);
            }
            None => from = at + 1,
        }
    }
    out
}

struct Cursor<'a> {
    file: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn bytes(&mut self, n: usize) -> Option<&'a [u8]> {
        let b = self.file.get(self.at..self.at.checked_add(n)?)?;
        self.at += n;
        Some(b)
    }
    fn u8(&mut self) -> Option<u8> {
        Some(self.bytes(1)?[0])
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.bytes(4)?.try_into().ok()?))
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.bytes(8)?.try_into().ok()?))
    }
    fn str(&mut self) -> Option<&'a str> {
        let n = self.u8()?;
        let (&nul, s) = self.bytes(n.into())?.split_last()?;
        (nul == 0).then_some(())?;
        std::str::from_utf8(s).ok()
    }
    /// `u32 n`, then `n` items of `size` bytes.
    fn items(&mut self, size: usize) -> Option<Vec<&'a [u8]>> {
        let n = self.u32()? as usize;
        let all = self.bytes(n.checked_mul(size)?)?;
        Some(all.chunks_exact(size).collect())
    }
    fn u32s(&mut self) -> Option<Vec<u32>> {
        let items = self.items(4)?;
        Some(
            items
                .iter()
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
        )
    }
}

fn parse(file: &[u8], start: usize) -> Option<Database<'_>> {
    let mut c = Cursor { file, at: start };
    (c.bytes(4)? == MAGIC).then_some(())?;
    c.u64()?;
    for _ in 0..c.u32()? {
        let size = c.u32()? as usize;
        c.bytes(size)?;
    }
    let size = c.u32()? as usize;
    c.bytes(size)?;
    let mut shaders: [Vec<Shader<'_>>; 6] = Default::default();
    for list in &mut shaders {
        for _ in 0..c.u32()? {
            list.push(shader(&mut c)?);
        }
    }
    let mut effects = Vec::new();
    for _ in 0..c.u32()? {
        effects.push(effect(&mut c, &shaders)?);
    }
    Some(Database {
        shaders,
        effects,
        start,
        end: c.at,
    })
}

fn shader<'a>(c: &mut Cursor<'a>) -> Option<Shader<'a>> {
    let name = c.str()?;
    let kind = c.u8()?;
    let n = c.u32()? as usize;
    let offset = c.at;
    let mut r = Cursor {
        file: c.bytes(n)?,
        at: 0,
    };
    (r.bytes(4)? == MARKER && r.u32()? as usize == n).then_some(())?;
    let unpacked_size = r.u32()? as usize;
    (unpacked_size <= MAX_UNPACKED).then_some(())?;
    let hash = c.u64()?;
    c.bytes(0x90)?;
    let m = c.u32()? as usize;
    c.bytes(m)?;
    Some(Shader {
        record: Record {
            offset,
            name: Some(name),
            kind,
            packed: &r.file[12..],
            unpacked_size,
        },
        hash,
    })
}

fn effect<'a>(c: &mut Cursor<'a>, shaders: &[Vec<Shader<'a>>; 6]) -> Option<Effect<'a>> {
    let name = c.str()?;
    let size = c.u32()? as usize;
    let mut stages: [Vec<u32>; 6] = Default::default();
    for (list, all) in stages.iter_mut().zip(shaders) {
        *list = c.u32s()?;
        list.iter()
            .all(|&i| (i as usize) < all.len())
            .then_some(())?;
    }
    c.bytes(0x50)?;
    let rasterizers = c.items(16)?;
    let depth_stencils = c.items(24)?;
    let blends = c.items(0x44)?;
    c.items(0x24)?;
    let techniques = c.u32s()?;
    let passes = c.items(24)?;
    let extra = c.u32s()?;
    // `size` is what the loader allocates for all this: a check on the parse.
    let mut loaded = stages.iter().map(|s| s.len() + 1).sum::<usize>() * 8
        + techniques.len() * 8
        + passes.len() * 32
        + extra.len() * 4;
    let strings = c.u32()?;
    if strings != SHARED {
        c.bytes(strings as usize)?;
        loaded += strings as usize;
    }
    for _ in 0..2 {
        let n = c.u32()? as usize;
        c.bytes(n)?;
        loaded += n;
    }
    let blocks = c.u32()? & 0x00FF_FFFF;
    if blocks != 0 {
        loaded = loaded.next_multiple_of(8) + blocks as usize * 8;
    }
    for _ in 0..blocks {
        match c.u32()? {
            SHARED => {
                c.u32()?;
            }
            n => {
                c.bytes(n as usize)?;
                loaded += n as usize;
            }
        }
    }
    let n = c.u32()? as usize;
    let names = c.bytes(n)?;
    (loaded + names.len() == size).then_some(())?;

    let state = |list: &[&'a [u8]], i: usize| list.get(i).copied();
    let pass = |p: &[u8]| -> Option<Pass<'a>> {
        let flags = p[10];
        let mut out = Pass {
            shaders: [None; 6],
            rasterizer: None,
            depth_stencil: None,
            blend: None,
            stencil_ref: (flags & 8 != 0).then_some(p[9]),
        };
        for (slot, list) in stages.iter().enumerate() {
            if p[slot] != 0 {
                out.shaders[slot] = Some(*list.get(usize::from(p[slot]) - 1)? as usize);
            }
        }
        if flags & 1 != 0 {
            out.rasterizer = Some(state(&rasterizers, p[6].into())?);
        }
        if flags & 2 != 0 {
            let i = usize::from(p[7]) + usize::from(flags >> 6 & 1) * 256;
            out.depth_stencil = Some(state(&depth_stencils, i)?);
        }
        if flags & 4 != 0 {
            let i = usize::from(p[8]) + usize::from(flags >> 7) * 256;
            out.blend = Some(state(&blends, i)?);
        }
        Some(out)
    };
    let techniques = techniques
        .iter()
        .map(|&t| {
            let at = (t & 0x7FFF) as usize;
            let tail = names.get(at..)?;
            let name = std::str::from_utf8(&tail[..memchr::memchr(0, tail)?]).ok()?;
            let first = (t >> 15 & 0x1FF) as usize;
            let list = passes.get(first..first + (t >> 24) as usize)?;
            Some(Technique {
                name,
                passes: list.iter().map(|p| pass(p)).collect::<Option<_>>()?,
            })
        })
        .collect::<Option<_>>()?;
    Some(Effect {
        name,
        techniques,
        rasterizers,
        depth_stencils,
        blends,
    })
}
