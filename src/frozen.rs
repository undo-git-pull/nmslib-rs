//! Saved-index file format, and the read-only index that uses it: either a
//! memory-mapped file or an in-memory copy in the same layout. Built indexes
//! are converted to this layout after each `add` (see `Hnsw::freeze`).
//!
//! The format is nmslib-rs's own (not NMSLIB's), little-endian and versioned.
//! Every section starts on a 64-byte boundary:
//!
//! | section      | contents                                                        |
//! |--------------|-----------------------------------------------------------------|
//! | header       | `HEADER_LEN` bytes, see [`Header`]                              |
//! | levels       | one `u8` per node                                               |
//! | upper_offset | one `u64` per node: start of its upper lists in `upper`, in words |
//! | upper        | `u32` words; per node and level `1..=level`: count, then `M` ids |
//! | level0       | one block of `stride` words per node: count, `2M` ids, padding, then the vector, padding |
//!
//! Level 0 interleaves each node's list with its vector, as NMSLIB's optimized
//! index does: expanding a node reads a list that sits next to the vector
//! already fetched to compute that node's distance. Cosine indexes store
//! normalized vectors.
//!
//! The header ends with an FNV-1a checksum of its other bytes, so accidental
//! header corruption is caught. Loading validates the header and the small
//! per-node arrays (levels and offsets), so section bounds are trusted
//! afterwards. Neighbor lists are not
//! scanned at load, which would read most of the file; instead every read caps
//! the count and skips ids that are out of range, so a corrupt list cannot
//! cause an out-of-bounds access.

use std::fmt;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use fearless_simd::Simd;
use memmap2::Mmap;

use crate::hnsw::{Entry, Space, View};
use crate::simd::prefetch_in;

const MAGIC: [u8; 8] = *b"RHNSWIDX";
pub const VERSION: u32 = 1;
const HEADER_LEN: usize = 256;
/// The last 8 header bytes hold the checksum of the bytes before them.
const CHECKSUM_AT: usize = HEADER_LEN - 8;
const ALIGN: u64 = 64;
const NO_ENTRY: u32 = u32::MAX;
/// Words per cache line; lists and vectors are padded to whole lines.
const LINE_WORDS: usize = 16;

#[derive(Debug)]
pub enum LoadError {
    Io(io::Error),
    /// The file is not a valid nmslib-rs index (or not one this build can read).
    Format(String),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Io(e) => write!(f, "{e}"),
            LoadError::Format(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for LoadError {}

impl From<io::Error> for LoadError {
    fn from(e: io::Error) -> Self {
        LoadError::Io(e)
    }
}

fn bad(msg: impl Into<String>) -> LoadError {
    LoadError::Format(msg.into())
}

/// FNV-1a over `bytes`.
pub(crate) fn header_checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3))
}

fn pad_words(words: usize) -> usize {
    words.div_ceil(LINE_WORDS) * LINE_WORDS
}

fn align_up(x: u64) -> u64 {
    x.div_ceil(ALIGN) * ALIGN
}

/// Byte range of one section.
#[derive(Clone, Copy, Debug, Default)]
struct Section {
    offset: u64,
    len: u64,
}

/// Fixed-size header at the start of the file.
#[derive(Clone, Debug)]
pub(crate) struct Header {
    pub dim: usize,
    pub space: Space,
    pub m: usize,
    pub max_m0: usize,
    pub ef_construction: usize,
    pub post: bool,
    pub seed: u64,
    pub n: usize,
    pub entry: Entry,
    /// Words in a level-0 list block (count plus `max_m0` ids, padded).
    list_words: usize,
    /// Words per level-0 node block (list block plus padded vector).
    stride: usize,
    levels: Section,
    upper_offset: Section,
    upper: Section,
    level0: Section,
    file_len: u64,
}

/// Index settings stored with a saved index.
pub(crate) struct Meta {
    pub dim: usize,
    pub space: Space,
    pub m: usize,
    pub ef_construction: usize,
    pub post: bool,
    pub seed: u64,
}

impl Header {
    fn layout(meta: &Meta, n: usize, entry: Entry, upper_words: u64) -> Header {
        let max_m0 = 2 * meta.m;
        let list_words = pad_words(1 + max_m0);
        let stride = list_words + pad_words(meta.dim);
        let levels = Section { offset: HEADER_LEN as u64, len: n as u64 };
        let upper_offset = Section { offset: align_up(levels.offset + levels.len), len: 8 * n as u64 };
        let upper = Section { offset: align_up(upper_offset.offset + upper_offset.len), len: 4 * upper_words };
        let level0 = Section { offset: align_up(upper.offset + upper.len), len: 4 * (n * stride) as u64 };
        Header {
            dim: meta.dim,
            space: meta.space,
            m: meta.m,
            max_m0,
            ef_construction: meta.ef_construction,
            post: meta.post,
            seed: meta.seed,
            n,
            entry,
            list_words,
            stride,
            levels,
            upper_offset,
            upper,
            level0,
            file_len: level0.offset + level0.len,
        }
    }

    fn encode(&self) -> [u8; HEADER_LEN] {
        let mut b = [0u8; HEADER_LEN];
        let mut at = 0;
        let mut put = |bytes: &[u8]| {
            b[at..at + bytes.len()].copy_from_slice(bytes);
            at += bytes.len();
        };
        put(&MAGIC);
        put(&VERSION.to_le_bytes());
        put(&u32::from(self.post).to_le_bytes());
        put(&(self.dim as u32).to_le_bytes());
        put(&(match self.space { Space::L2 => 0u32, Space::Cosine => 1 }).to_le_bytes());
        put(&(self.m as u32).to_le_bytes());
        put(&(self.max_m0 as u32).to_le_bytes());
        put(&(self.ef_construction as u32).to_le_bytes());
        put(&self.entry.node.unwrap_or(NO_ENTRY).to_le_bytes());
        put(&(self.n as u64).to_le_bytes());
        put(&(self.entry.max_level as u32).to_le_bytes());
        put(&(self.list_words as u32).to_le_bytes());
        put(&(self.stride as u32).to_le_bytes());
        put(&0u32.to_le_bytes());
        put(&self.seed.to_le_bytes());
        for s in [self.levels, self.upper_offset, self.upper, self.level0] {
            put(&s.offset.to_le_bytes());
            put(&s.len.to_le_bytes());
        }
        put(&self.file_len.to_le_bytes());
        let sum = header_checksum(&b[..CHECKSUM_AT]);
        b[CHECKSUM_AT..].copy_from_slice(&sum.to_le_bytes());
        b
    }

    /// Parses and checks a header against the actual file length. After this,
    /// every section lies inside the file, is 64-byte aligned and has the
    /// length the node count and sizes imply.
    fn decode(b: &[u8], actual_len: u64) -> Result<Header, LoadError> {
        if b.len() < HEADER_LEN {
            return Err(bad(format!("file is {} bytes, shorter than an nmslib-rs header", b.len())));
        }
        if b[..8] != MAGIC {
            return Err(bad("not an nmslib-rs index file (bad magic bytes)"));
        }
        let u32_at = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().unwrap());
        let u64_at = |i: usize| u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        let version = u32_at(8);
        if version != VERSION {
            return Err(bad(format!("index file format version {version}; this build reads version {VERSION}")));
        }
        if u64_at(CHECKSUM_AT) != header_checksum(&b[..CHECKSUM_AT]) {
            return Err(bad("header checksum mismatch: the file header is corrupt"));
        }
        let usize_of = |x: u64, what: &str| {
            usize::try_from(x).map_err(|_| bad(format!("{what} {x} does not fit this platform")))
        };
        let post = match u32_at(12) {
            0 => false,
            1 => true,
            f => return Err(bad(format!("unknown flags {f:#x}"))),
        };
        let dim = u32_at(16) as usize;
        let space = match u32_at(20) {
            0 => Space::L2,
            1 => Space::Cosine,
            s => return Err(bad(format!("unknown space code {s}"))),
        };
        let m = u32_at(24) as usize;
        let max_m0 = u32_at(28) as usize;
        let ef_construction = u32_at(32) as usize;
        let entry_node = u32_at(36);
        let n = usize_of(u64_at(40), "node count")?;
        let max_level = u32_at(48) as usize;
        let list_words = u32_at(52) as usize;
        let stride = u32_at(56) as usize;
        let seed = u64_at(64);
        let section = |i: usize| Section { offset: u64_at(72 + 16 * i), len: u64_at(80 + 16 * i) };
        let (levels, upper_offset, upper, level0) = (section(0), section(1), section(2), section(3));
        let file_len = u64_at(136);

        if dim == 0 || dim > 1 << 20 {
            return Err(bad(format!("dim {dim} out of range")));
        }
        if !(2..=1 << 16).contains(&m) || max_m0 != 2 * m {
            return Err(bad(format!("M {m} / maxM0 {max_m0} out of range")));
        }
        if n >= NO_ENTRY as usize {
            return Err(bad(format!("node count {n} out of range")));
        }
        if list_words != pad_words(1 + max_m0) || stride != list_words + pad_words(dim) {
            return Err(bad("level-0 block layout does not match dim and M"));
        }
        if max_level > u8::MAX as usize {
            return Err(bad(format!("max level {max_level} out of range")));
        }
        let entry = match (n, entry_node) {
            (0, NO_ENTRY) => Entry::default(),
            (0, _) => return Err(bad("empty index with an entry point")),
            (_, e) if (e as usize) < n => Entry { node: Some(e), max_level },
            _ => return Err(bad(format!("entry point {entry_node} out of range"))),
        };
        if file_len != actual_len {
            return Err(bad(format!("file is {actual_len} bytes but the header says {file_len}")));
        }
        let level0_len = (n as u64)
            .checked_mul(stride as u64)
            .and_then(|w| w.checked_mul(4))
            .ok_or_else(|| bad("level-0 section size overflows"))?;
        for (name, s, expected) in [
            ("levels", levels, Some(n as u64)),
            ("upper_offset", upper_offset, Some(8 * n as u64)),
            ("upper", upper, None),
            ("level0", level0, Some(level0_len)),
        ] {
            let end = s.offset.checked_add(s.len).ok_or_else(|| bad(format!("{name} section overflows")))?;
            if s.offset < HEADER_LEN as u64 || s.offset % ALIGN != 0 || end > actual_len {
                return Err(bad(format!("{name} section [{}, {end}) is misplaced in a {actual_len}-byte file", s.offset)));
            }
            if expected.is_some_and(|e| e != s.len) || s.len % 4 != 0 && name == "upper" {
                return Err(bad(format!("{name} section has length {}", s.len)));
            }
        }
        Ok(Header {
            dim,
            space,
            m,
            max_m0,
            ef_construction,
            post,
            seed,
            n,
            entry,
            list_words,
            stride,
            levels,
            upper_offset,
            upper,
            level0,
            file_len,
        })
    }
}

fn range(s: Section) -> std::ops::Range<usize> {
    // Sections were checked to lie inside the mapped file, so they fit usize.
    s.offset as usize..(s.offset + s.len) as usize
}

/// Memory holding an index in the file layout.
enum Backing {
    /// A saved file mapped read-only; shared with other processes.
    Mapped(Mmap),
    /// An owned copy. `words[start..]` begins on a 64-byte boundary, so
    /// level-0 blocks sit on cache lines exactly as they do in a mapped file.
    Owned { words: Vec<u64>, start: usize, len: usize },
}

/// An index in the interleaved file layout, read-only: either a mapped file
/// or an in-memory copy (a loaded file, or a freshly built index after
/// `freeze`).
pub struct Frozen {
    backing: Backing,
    header: Header,
}

/// Zeroed memory of `len` bytes starting on a 64-byte boundary.
fn aligned_zeroed(len: usize) -> Backing {
    let words = vec![0u64; len.div_ceil(8) + 8];
    let start = words.as_ptr().align_offset(ALIGN as usize);
    Backing::Owned { words, start, len }
}

fn owned_bytes_mut(backing: &mut Backing) -> &mut [u8] {
    match backing {
        Backing::Owned { words, start, len } => &mut bytemuck::cast_slice_mut(&mut words[*start..])[..*len],
        Backing::Mapped(_) => unreachable!("mapped memory is read-only"),
    }
}

impl Frozen {
    /// Maps and validates the index file at `path`.
    pub fn map(path: &Path) -> Result<Frozen, LoadError> {
        let file = open_checked(path)?;
        // SAFETY: the map is read-only, so this process never writes through
        // it. Rust cannot rule out another process truncating or rewriting the
        // file while it is mapped; reads would then fault (SIGBUS) or see
        // changing bytes. Callers must not modify a saved index file while it
        // is loaded with mmap. `save` replaces files by renaming a new file
        // over them, which leaves existing mappings of the old file intact.
        let map = unsafe { Mmap::map(&file)? };
        Frozen::validated(Backing::Mapped(map))
    }

    /// Reads and validates the index file at `path` into owned memory.
    pub fn read(path: &Path) -> Result<Frozen, LoadError> {
        let mut file = open_checked(path)?;
        let len = usize::try_from(file.metadata()?.len())
            .map_err(|_| bad("index file is too large for this platform"))?;
        let mut backing = aligned_zeroed(len);
        io::Read::read_exact(&mut file, owned_bytes_mut(&mut backing))?;
        Frozen::validated(backing)
    }

    /// Lays out `view` in owned memory, in the file layout.
    pub(crate) fn from_view<V: View>(view: &V, meta: &Meta) -> Frozen {
        let header = layout_for(view, meta);
        let mut backing = aligned_zeroed(header.file_len as usize);
        let out = io::Cursor::new(owned_bytes_mut(&mut backing));
        write_index(view, &header, out).expect("the buffer is sized for the index");
        Frozen { backing, header }
    }

    fn validated(backing: Backing) -> Result<Frozen, LoadError> {
        let bytes = bytes_of(&backing);
        let header = Header::decode(&bytes[..HEADER_LEN.min(bytes.len())], bytes.len() as u64)?;
        let frozen = Frozen { backing, header };
        frozen.check_levels_and_offsets()?;
        Ok(frozen)
    }

    pub fn is_mapped(&self) -> bool {
        matches!(self.backing, Backing::Mapped(_))
    }

    pub(crate) fn header(&self) -> &Header {
        &self.header
    }

    fn check_levels_and_offsets(&self) -> Result<(), LoadError> {
        let h = &self.header;
        let v = self.view();
        let upper_words = v.upper.len() as u64;
        let per_level = 1 + h.m as u64;
        for (id, (&level, &offset)) in v.levels.iter().zip(v.upper_offset).enumerate() {
            if level as usize > h.entry.max_level {
                return Err(bad(format!("node {id} has level {level}, above the max level {}", h.entry.max_level)));
            }
            let end = offset.checked_add(level as u64 * per_level);
            if level > 0 && !end.is_some_and(|e| e <= upper_words) {
                return Err(bad(format!("node {id}: upper-level lists run past the upper section")));
            }
        }
        if let Some(e) = h.entry.node
            && v.levels[e as usize] as usize != h.entry.max_level
        {
            return Err(bad("entry point is not on the max level"));
        }
        Ok(())
    }

    pub(crate) fn view(&self) -> FrozenView<'_> {
        let h = &self.header;
        let m = bytes_of(&self.backing);
        // Sections are 64-byte aligned and the memory starts on a page
        // (mapped) or 64-byte (owned) boundary, so these casts cannot fail.
        let level0 = &m[range(h.level0)];
        FrozenView {
            header: h,
            levels: &m[range(h.levels)],
            upper_offset: bytemuck::cast_slice(&m[range(h.upper_offset)]),
            upper: bytemuck::cast_slice(&m[range(h.upper)]),
            level0_f32: bytemuck::cast_slice(level0),
            level0_u32: bytemuck::cast_slice(level0),
        }
    }
}

fn bytes_of(backing: &Backing) -> &[u8] {
    match backing {
        Backing::Mapped(map) => map,
        Backing::Owned { words, start, len } => &bytemuck::cast_slice(&words[*start..])[..*len],
    }
}

fn open_checked(path: &Path) -> Result<File, LoadError> {
    if cfg!(target_endian = "big") {
        return Err(bad("nmslib-rs index files are little-endian; big-endian hosts are not supported"));
    }
    let file = File::open(path)?;
    let len = file.metadata()?.len();
    if len < HEADER_LEN as u64 {
        return Err(bad(format!("file is {len} bytes, shorter than an nmslib-rs header")));
    }
    Ok(file)
}

/// Read access to a frozen index.
pub(crate) struct FrozenView<'a> {
    header: &'a Header,
    levels: &'a [u8],
    upper_offset: &'a [u64],
    upper: &'a [u32],
    level0_f32: &'a [f32],
    level0_u32: &'a [u32],
}

impl FrozenView<'_> {
    fn list(&self, id: u32, level: usize) -> &[u32] {
        let h = self.header;
        let i = id as usize;
        if level == 0 {
            let block = &self.level0_u32[i * h.stride..i * h.stride + 1 + h.max_m0];
            let count = (block[0] as usize).min(h.max_m0);
            &block[1..1 + count]
        } else if level <= self.levels[i] as usize {
            // In range: checked at open against the upper section's length.
            let start = self.upper_offset[i] as usize + (level - 1) * (1 + h.m);
            let count = (self.upper[start] as usize).min(h.m);
            &self.upper[start + 1..start + 1 + count]
        } else {
            &[]
        }
    }
}

impl View for FrozenView<'_> {
    fn space(&self) -> Space {
        self.header.space
    }

    fn len(&self) -> usize {
        self.header.n
    }

    fn entry(&self) -> Entry {
        self.header.entry
    }

    fn level(&self, id: u32) -> usize {
        self.levels[id as usize] as usize
    }

    #[inline(always)]
    fn vector(&self, id: u32) -> &[f32] {
        let h = self.header;
        let start = id as usize * h.stride + h.list_words;
        &self.level0_f32[start..start + h.dim]
    }

    #[inline(always)]
    fn neighbors(&self, id: u32, level: usize) -> impl Iterator<Item = u32> + '_ {
        let n = self.header.n as u32;
        // Skip ids out of range, so a corrupt file cannot index past the end.
        self.list(id, level).iter().copied().filter(move |&x| x < n)
    }

    #[inline(always)]
    fn prefetch_list<S: Simd>(&self, simd: S, id: u32, level: usize) {
        if level == 0 {
            let start = id as usize * self.header.stride;
            prefetch_in(simd, &self.level0_u32[start..start + 1]);
        }
    }
}

fn layout_for<V: View>(view: &V, meta: &Meta) -> Header {
    let per_level = 1 + meta.m;
    let upper_words: u64 = (0..view.len() as u32).map(|id| (view.level(id) * per_level) as u64).sum();
    Header::layout(meta, view.len(), view.entry(), upper_words)
}

/// Writes `view` as an index file at `path`: to a temporary file in the same
/// directory first, then renamed over `path`, so a process that has the old
/// file mapped keeps a valid mapping.
pub(crate) fn save<V: View>(view: &V, meta: &Meta, path: &Path) -> io::Result<()> {
    if cfg!(target_endian = "big") {
        return Err(io::Error::other("saving is only supported on little-endian hosts"));
    }
    let header = layout_for(view, meta);
    let tmp = temp_path(path);
    let result = (|| {
        let mut file = BufWriter::with_capacity(8 << 20, File::create(&tmp)?);
        write_index(view, &header, &mut file)?;
        let file = file.into_inner().map_err(|e| e.into_error())?;
        file.sync_all()?;
        fs::rename(&tmp, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

/// Writes the whole index in the file layout described by `header`.
fn write_index<V: View, W: Write>(view: &V, header: &Header, w: W) -> io::Result<()> {
    let n = view.len();
    let per_level = 1 + header.m;
    let mut w = Out { w, pos: 0 };
    w.write(&header.encode())?;
    let levels: Vec<u8> = (0..n as u32).map(|id| view.level(id) as u8).collect();
    w.write(&levels)?;
    w.pad_to(header.upper_offset.offset)?;

    let mut next = 0u64;
    let mut offsets = Vec::with_capacity(n);
    for &level in &levels {
        offsets.push(if level > 0 { next } else { 0 });
        next += (level as usize * per_level) as u64;
    }
    w.write(bytemuck::cast_slice(&offsets))?;
    w.pad_to(header.upper.offset)?;

    let mut words = vec![0u32; per_level];
    for (id, &level) in levels.iter().enumerate() {
        for l in 1..=level as usize {
            words.fill(0);
            let mut count = 0;
            for nb in view.neighbors(id as u32, l) {
                count += 1;
                words[count] = nb;
            }
            words[0] = count as u32;
            w.write(bytemuck::cast_slice(&words))?;
        }
    }
    w.pad_to(header.level0.offset)?;

    let mut block = vec![0u32; header.stride];
    for id in 0..n as u32 {
        block.fill(0);
        let mut count = 0;
        for nb in view.neighbors(id, 0) {
            count += 1;
            block[count] = nb;
        }
        block[0] = count as u32;
        for (slot, x) in block[header.list_words..].iter_mut().zip(view.vector(id)) {
            *slot = x.to_bits();
        }
        w.write(bytemuck::cast_slice(&block))?;
    }
    debug_assert_eq!(w.pos, header.file_len);
    w.w.flush()
}

/// A writer that tracks its position, for padding to section offsets.
struct Out<W: Write> {
    w: W,
    pos: u64,
}

impl<W: Write> Out<W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.w.write_all(bytes)?;
        self.pos += bytes.len() as u64;
        Ok(())
    }

    fn pad_to(&mut self, offset: u64) -> io::Result<()> {
        let gap = (offset - self.pos) as usize;
        self.write(&[0u8; ALIGN as usize][..gap])
    }
}

fn temp_path(path: &Path) -> PathBuf {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    path.with_file_name(format!(".{name}.{}.tmp", std::process::id()))
}

#[cfg(test)]
mod tests;
