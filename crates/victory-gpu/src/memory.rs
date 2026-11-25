//! Title memory by address, for translated shaders that read and write
//! through raw 40-bit addresses (`victory_shader::emit::SLOT_MEMORY`: the
//! console's ray-tracing structure builder and the programs that walk its
//! structures; the structures themselves lie in plain title memory, in no
//! buffer object).
//!
//! The copies are the ones every other access uses ([`crate::shadow`]: a
//! title buffer's copy, or the 4 MiB window of memory outside buffers), so
//! a root view and a raw address reach the same host bytes. This module
//! keeps what a shader needs to find them:
//!
//! - the **directory**: per 64 KiB page of title address space the number
//!   of the copy that holds the page (0: none) and the page's byte offset
//!   in it; changing a mapped page's owner gives subsequent bindings a new
//!   directory while recorded command lists keep their original directory;
//! - per view heap a run of slots with a raw unordered access view of each
//!   numbered copy (slot 0 of the run: a null view, for pages without a
//!   copy);
//! - the **faults**: the page numbers shaders found empty, read back here
//!   once a frame, so the copies exist from the next frame on. A pointer a
//!   shader takes out of memory cannot be known before it runs; the
//!   addresses a dispatch is given as root views are made ready before it.
//!
//! **A page that is not there when a shader touches it is a lost access**:
//! the read gives 0, the store is dropped, and the fault note only helps
//! the next frame. For the structure builder that is the same as a dropped
//! stage, so before a dispatch the host makes ready everything it can see
//! the dispatch name: the whole copy around each root view, around each
//! pair of shader-visible root constants that is a title address and around
//! each such constant that is one in 256-byte units (the builder passes
//! structures all three ways). A structure may cross a window's end, so
//! roots also name the next cell, and constant candidates do when their own
//! page exists.
//! shortcut: what a stage reaches only through pointers it reads out of
//! memory (InitHeaders' table of structure addresses, LoadInstances'
//! instance records) is only made ready if another source names it. A
//! live acceleration-structure SRV names its header for the builder and
//! traversal even when the shader obtains the pointer from a GPU-built table;
//! the fault count in the log says whether that happens. The whole answer
//! is the extent of the title's allocation around an address, which the
//! memory module of the layer knows and this one does not ask yet.
//!
//! shortcut: a copy that is numbered is kept for the rest of the run, and
//! each view heap grows its descriptor run as more copies are needed.
//! Reads see all addressable copies. Writers publish all visible copies unless
//! the direct builder's bound constants identify every possible raw store.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};
use std::sync::atomic::{AtomicU64, Ordering};

use victory_shader::emit::{MEMORY_FAULTS, MEMORY_PAGES};
use windows::Win32::Graphics::Direct3D12::*;

use crate::descriptor::host::HostHeap;
use crate::host::{HostList, host};
use crate::shadow::Shadow;

#[cfg(feature = "diagnostics")]
pub(crate) mod ray_probe;

/// A window of memory outside buffers (`shadow.rs`).
const CELL: u64 = 4 << 20;
/// A directory page.
const PAGE: u64 = 1 << 16;
// Slot zero is the null descriptor. Only real addresses may grow the first
// run; a scalar or a neighbor guess must not retain another 4 MiB window.
const SPECULATIVE_COPIES: usize = crate::descriptor::host::FIRST_RAW_PAGE_SLOTS as usize - 1;

// A title may register buffers while another thread is recording GPU work.
// Record only the changed spans. A lagging reader that misses journal history
// rechecks its resident pages. Plain rendering pays no journal cost until a
// shader has used the raw-address directory.
static CHANGES: OnceLock<Mutex<VecDeque<(u64, u64)>>> = OnceLock::new();
static GENERATION: AtomicU64 = AtomicU64::new(0);

pub(crate) fn ownership_changed(start: u64, len: u64) {
    if let Some(changes) = CHANGES.get() {
        let mut changes = changes.lock().unwrap_or_else(|e| e.into_inner());
        if changes.len() == 256 {
            changes.pop_front();
        }
        changes.push_back((start / PAGE, start.saturating_add(len).div_ceil(PAGE)));
        GENERATION.fetch_add(1, Ordering::Release);
    }
}

/// Byte index used by translated shaders in a raw UAV starting at byte zero
/// of the Windows resource. An in-place resource can begin before the guest
/// allocation, and a directory entry must cover its complete 64 KiB page.
fn page_byte(page: u64, start: u64, len: u64, prefix: u64) -> Option<u32> {
    let at = page.checked_sub(start)?;
    if at.checked_add(PAGE)? > len {
        return None;
    }
    let byte = prefix.checked_add(at)?;
    u32::try_from(byte.checked_add(PAGE - 1)?).ok()?;
    Some(byte as u32)
}

/// Which bytes a shader can change through the raw directory. An exact
/// declaration contains at most the builder header and a centroid box.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Writes {
    None,
    All,
    Ranges([(u64, u64); 2]),
}

impl Writes {
    /// Visit the parts of each declaration addressable by the shader. The
    /// lookup masks the high dword to 8 bits, and its clamped last directory
    /// entry is always a null UAV. Split at each page, including 40-bit wrap.
    fn pages(&self, mut visit: impl FnMut(u64, u64)) {
        const MASK: u64 = (1 << 40) - 1;
        let Self::Ranges(ranges) = self else { return };
        for &(address, mut len) in ranges {
            let mut at = address & MASK;
            while len != 0 {
                let part = len.min(PAGE - at % PAGE);
                if at / PAGE < u64::from(MEMORY_PAGES - 1) {
                    visit(at, part);
                }
                at = (at + part) & MASK;
                len -= part;
            }
        }
    }

    fn owned(&self, pages: &BTreeMap<u64, usize>, mut visit: impl FnMut(usize, u64, u64)) {
        self.pages(|at, len| {
            if let Some(&owner) = pages.get(&(at / PAGE)) {
                visit(owner, at, len);
            }
        });
    }
}

struct Memory {
    /// Buffer, its mapping, its GPU address.
    directory: (ID3D12Resource, *mut u32, u64),
    faults: (ID3D12Resource, *mut u32, u64),
    /// Copy `n` is `copies[n - 1]`.
    copies: Vec<Arc<Shadow>>,
    /// Shadow IDs are unique for the lifetime of a backing, including after
    /// the guest reuses its address.
    numbers: HashMap<u64, usize>,
    pages: BTreeMap<u64, usize>,
    ranges: Vec<Vec<(u64, u64)>>,
    recalculate: BTreeSet<usize>,
    generation: u64,
    coverage_revision: u64,
    /// The frame the faults were last read in.
    frame: u64,
    /// Pages asked for this frame that are not the title's memory.
    refused: std::collections::HashSet<u64>,
    /// Typed locations already considered for this heap snapshot this frame.
    acceleration: Option<Arc<[u64]>>,
    /// What first caused each copy to receive a descriptor. Replacements
    /// during reconciliation account for the remainder.
    from_roots: usize,
    from_neighbors: usize,
    from_constants: usize,
    from_faults: usize,
    speculative_skips: u64,
    /// A speculative expansion can stop mid-buffer once, when the initial
    /// slots fill. A subsequent confirmed access completes that expansion.
    partial_copy: Option<usize>,
}

// SAFETY: the mappings are of buffers the struct owns; all access is under
// the mutex.
unsafe impl Send for Memory {}

static MEMORY: Mutex<Option<Memory>> = Mutex::new(None);

/// What a draw binds.
pub(crate) struct Ready {
    pub(crate) directory: u64,
    pub(crate) faults: u64,
    /// The view heap slot of the null view; copy `n`'s view is `n` further.
    pub(crate) base: u32,
    /// Last numbered copy this draw prepared. Other recording threads can
    /// subsequently add mappings to the same directory.
    pub(crate) count: u32,
}

impl Memory {
    fn new() -> Option<Memory> {
        CHANGES.get_or_init(Default::default);
        tracing::info!("title memory by address: a shader reads or writes through raw addresses");
        let mut fault_dwords = MEMORY_FAULTS;
        if victory_shader::emit::ray_probe_enabled() {
            fault_dwords += victory_shader::emit::RAY_PROBE_DWORDS;
        }
        Some(Memory {
            directory: Self::table(u64::from(MEMORY_PAGES) * 2, "directory")?,
            faults: Self::table(u64::from(fault_dwords), "faults")?,
            copies: Vec::new(),
            numbers: HashMap::new(),
            pages: BTreeMap::new(),
            ranges: Vec::new(),
            recalculate: BTreeSet::new(),
            generation: GENERATION.load(Ordering::Acquire),
            coverage_revision: 0,
            frame: u64::MAX,
            refused: Default::default(),
            acceleration: None,
            from_roots: 0,
            from_neighbors: 0,
            from_constants: 0,
            from_faults: 0,
            speculative_skips: 0,
            partial_copy: None,
        })
    }

    fn table(dwords: u64, what: &str) -> Option<(ID3D12Resource, *mut u32, u64)> {
        let (buffer, data) = host()?.shared_buffer(dwords * 4)?;
        crate::host::name(&buffer, || format!("title memory by address: {what}"));
        // SAFETY: a fresh mapped buffer of the requested length.
        unsafe { std::ptr::write_bytes(data, 0, (dwords * 4) as usize) };
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        Some((buffer, data.cast(), gpu))
    }

    fn page_owner(start: u64) -> Option<(Arc<Shadow>, u32)> {
        if !crate::texture::committed(start as *const u8, PAGE as usize) {
            return None;
        }
        let whole = |s: &Arc<Shadow>| s.offset_in(start).is_some_and(|_| s.left(start) >= PAGE);
        let shadow = crate::shadow::find(start, 1)
            .filter(whole)
            .or_else(|| crate::shadow::find(start, PAGE).filter(whole))?;
        let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
        if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
            return None;
        }
        let (first, len) = shadow.span();
        let offset = page_byte(start, first, len, shadow.offset(first))?;
        Some((shadow, offset))
    }

    /// Assign stable slots. A previously recorded list may still reference any
    /// old slot, even after the corresponding guest address has been reused.
    fn number(&mut self, shadow: &Arc<Shadow>) -> Option<usize> {
        if let Some(&index) = self.numbers.get(&shadow.id()) {
            return Some(index);
        }
        // The directory stores a one-based u32 slot; zero is the null view.
        u32::try_from(self.copies.len()).ok()?.checked_add(1)?;
        self.numbers.insert(shadow.id(), self.copies.len());
        self.copies.push(shadow.clone());
        self.ranges.push(Vec::new());
        Some(self.copies.len() - 1)
    }

    fn set_page(&mut self, page: u64, mapping: Option<(usize, u32)>) {
        let (slot, offset) = if let Some((index, offset)) = mapping {
            self.pages.insert(page, index);
            self.recalculate.insert(index);
            (index as u32 + 1, offset)
        } else {
            self.pages.remove(&page);
            (0, 0)
        };
        // SAFETY: page is inside the table. New assignments replace existing
        // ones only in a private directory; first-time assignments publish
        // the offset before the nonzero slot.
        unsafe {
            let entry = self.directory.1.add(page as usize * 2);
            entry.add(1).write_volatile(offset);
            entry.write_volatile(slot);
        }
    }

    fn reconcile(&mut self) -> Option<()> {
        if self.generation == GENERATION.load(Ordering::Acquire) {
            return Some(());
        }
        let mut check = BTreeSet::new();
        let generation = {
            let changes = CHANGES.get()?.lock().unwrap_or_else(|e| e.into_inner());
            let now = GENERATION.load(Ordering::Relaxed);
            let count = now - self.generation;
            if count > changes.len() as u64 {
                check.extend(self.pages.keys().copied());
            } else {
                for &(first, end) in changes.iter().rev().take(count as usize) {
                    check.extend(self.pages.range(first..end).map(|(&page, _)| page));
                }
            }
            now
        };
        let mut replaced = false;
        for page in check {
            let old = self.pages[&page];
            let owner = Self::page_owner(page * PAGE);
            if owner.as_ref().is_some_and(|(s, _)| Arc::ptr_eq(s, &self.copies[old])) {
                continue;
            }
            if !replaced {
                let next = Self::table(u64::from(MEMORY_PAGES) * 2, "directory")?;
                // SAFETY: equal-sized disjoint tables. The CPU records under
                // the memory mutex and all GPU directory accesses are reads.
                unsafe { std::ptr::copy_nonoverlapping(self.directory.1, next.1, MEMORY_PAGES as usize * 2) };
                self.directory = next;
                replaced = true;
                self.coverage_revision += 1;
            }
            let mapping = owner.and_then(|(s, offset)| self.number(&s).map(|n| (n, offset)));
            self.recalculate.insert(old);
            self.set_page(page, mapping);
        }
        self.generation = generation;
        self.refused.clear();
        self.acceleration = None;
        Some(())
    }

    fn update_ranges(&mut self) {
        for index in std::mem::take(&mut self.recalculate) {
            let (start, len) = self.copies[index].span();
            let ranges = &mut self.ranges[index];
            ranges.clear();
            for (&page, &owner) in self.pages.range(start.div_ceil(PAGE)..(start + len) / PAGE) {
                if owner != index {
                    continue;
                }
                let at = page * PAGE;
                if let Some((first, bytes)) = ranges.last_mut()
                    && *first + *bytes == at
                {
                    *bytes += PAGE;
                } else {
                    ranges.push((at, PAGE));
                }
            }
        }
    }

    /// Makes the page of `address` available and reports whether it has a copy.
    fn map(&mut self, address: u64) -> bool {
        self.map_page(address, false)
    }

    fn map_page(&mut self, address: u64, speculative: bool) -> bool {
        let page = address / PAGE;
        // The last entry stays empty: addresses past the directory read it.
        if page >= u64::from(MEMORY_PAGES) - 1 {
            return false;
        }
        // SAFETY: inside the directory's mapping.
        let entry = unsafe { self.directory.1.add(page as usize * 2) };
        // SAFETY: as above.
        let slot = unsafe { entry.read() };
        if slot != 0 {
            let index = slot as usize - 1;
            if !speculative && self.partial_copy == Some(index) {
                self.coverage_revision += 1;
                self.populate(index, false);
            }
            return true;
        }
        let start = page * PAGE;
        // A shader's pointer, or a constant taken for an address, may be
        // anything: only memory the title has.
        if self.refused.contains(&page) {
            return false;
        }
        if speculative && self.copies.len() >= SPECULATIVE_COPIES {
            // Do not call page_owner: it can allocate a shadow immediately.
            // Leave the page eligible for a later root, AS descriptor or fault.
            self.speculative_skips += 1;
            return false;
        }
        let Some((shadow, offset)) = Self::page_owner(start) else {
            self.refused.insert(page);
            return false;
        };
        let count = self.copies.len();
        let Some(n) = self.number(&shadow) else { return false };
        if n < count {
            // An existing copy regained a page that earlier binds did not cover.
            self.coverage_revision += 1;
        }
        self.set_page(page, Some((n, offset)));
        if n == count || (!speculative && self.partial_copy == Some(n)) {
            self.populate(n, speculative);
        }
        true
    }

    fn populate(&mut self, index: usize, speculative: bool) {
        // A builder can follow an indirect address to any full page in this
        // buffer. Prefer a nested resource already registered at that page.
        let (first, len) = self.copies[index].span();
        let end = (first + len) / PAGE;
        if self.partial_copy == Some(index) {
            self.partial_copy = None;
        }
        for p in first.div_ceil(PAGE)..end.min(u64::from(MEMORY_PAGES) - 1) {
            if self.pages.contains_key(&p) {
                continue;
            }
            // A nested resource in this span may need its own copy. At the
            // speculation limit leave other pages for a confirmed access.
            if speculative && self.copies.len() >= SPECULATIVE_COPIES {
                self.speculative_skips += 1;
                self.partial_copy = Some(index);
                break;
            }
            if let Some((owner, off)) = Self::page_owner(p * PAGE)
                && let Some(number) = self.number(&owner)
            {
                self.set_page(p, Some((number, off)));
            }
        }
    }

    /// The pages shaders found empty since the last frame.
    fn take_faults(&mut self) {
        let mut pages = Vec::new();
        for i in 1..MEMORY_FAULTS as usize {
            // SAFETY: inside the faults' mapping; the GPU may be writing,
            // a number missed now is written again next frame.
            let at = unsafe { self.faults.1.add(i) };
            // SAFETY: as above.
            let page = unsafe { at.read_volatile() };
            if page != 0 {
                pages.push(page);
                // SAFETY: as above.
                unsafe { at.write_volatile(0) };
            }
        }
        if !pages.is_empty() {
            // Each is an access that read 0 or was dropped.
            static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 50 {
                tracing::warn!(
                    pages = pages.len(),
                    first = format_args!("{:#x}", u64::from(pages[0]) * PAGE),
                    "title memory by address: shaders touched pages that had no copy (those accesses read 0 or were dropped); the copies are made now"
                );
            }
        }
        let before = self.copies.len();
        for page in pages {
            self.map(u64::from(page) * PAGE);
        }
        self.from_faults += self.copies.len() - before;
    }

    /// What a dispatch names: the copy around `address` and the 4 MiB
    /// after it.
    fn name(&mut self, address: u64) {
        let before = self.copies.len();
        self.map(address);
        let after_root = self.copies.len();
        self.from_roots += after_root - before;
        self.map_page((address & !(CELL - 1)) + CELL, true);
        self.from_neighbors += self.copies.len() - after_root;
    }

    fn name_constant(&mut self, address: u64) {
        // A scalar that happens to precede a live allocation does not
        // establish that the shader uses any part of that allocation.
        if self.map_page(address, true) {
            self.map_page((address & !(CELL - 1)) + CELL, true);
        }
    }

    /// A null-resource acceleration-structure SRV supplies a definite guest
    /// address even when no shader root directly names the structure. It
    /// gives no size, so request only the cell containing its header.
    fn map_acceleration_structures(&mut self, heap: &HostHeap) {
        let locations = heap.acceleration_addresses();
        if self.acceleration.as_ref().is_some_and(|last| Arc::ptr_eq(last, &locations)) {
            return;
        }
        for &address in locations.iter() {
            self.map(address);
        }
        self.acceleration = Some(locations);
    }

    fn prepare_addresses(&mut self, heap: &HostHeap, addresses: &[u64], constants: &[u32]) {
        let before = self.copies.len();
        self.map_acceleration_structures(heap);
        self.from_roots += self.copies.len() - before;
        for &address in addresses {
            self.name(address);
        }
        let after_roots = self.copies.len();
        // A title address in two constants (low dword, then bits 32 to 39), or
        // in one in 256-byte units. A constant that only looks like one names
        // memory nothing reads; one that is not the title's memory is refused.
        for (i, &c) in constants.iter().enumerate() {
            if let Some(&high) = constants.get(i + 1)
                && (1..=0xFF).contains(&high)
                && c >= 0x1_0000
            {
                self.name_constant(u64::from(c) | u64::from(high) << 32);
            }
            if c >= 0x0100_0000 {
                self.name_constant(u64::from(c) << 8);
            }
        }
        self.from_constants += self.copies.len() - after_roots;
    }
}

/// Before a draw whose shader reads title memory by address: what its
/// root arguments name gets its copies (`addresses`: its root views;
/// `constants`: its root constants, see the module), every numbered copy
/// has its view in `heap`'s run and is brought up to date when `list` is
/// submitted (and its declared writes copied back after). None if the host
/// cannot hold it; the draw is then dropped.
pub(crate) fn ready(
    list: &HostList,
    heap: &HostHeap,
    addresses: &[u64],
    constants: &[u32],
    writes: Writes,
) -> Option<Ready> {
    let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        *guard = Memory::new();
    }
    let m = guard.as_mut()?;
    let before = m.copies.len();
    let skipped_before = m.speculative_skips;
    m.reconcile()?;
    let frame = crate::frame::current();
    if m.frame != frame {
        m.frame = frame;
        m.refused.clear();
        m.acceleration = None;
        m.take_faults();
    }
    m.prepare_addresses(heap, addresses, constants);
    // Resolve the precise destinations even when an address has tag bits or
    // its range crosses into a page the root prefetch did not discover.
    let before_writes = m.copies.len();
    writes.pages(|at, _| { m.map(at); });
    m.from_roots += m.copies.len() - before_writes;
    let copies = m.copies.len();
    if copies / 256 != before / 256 || (skipped_before == 0 && m.speculative_skips != 0) {
        tracing::info!(frame, copies, roots_and_as=m.from_roots, constants=m.from_constants,
            root_neighbors=m.from_neighbors, shader_faults=m.from_faults, deferred_speculation=m.speculative_skips,
            ownership_changes=copies - m.from_roots - m.from_constants - m.from_faults - m.from_neighbors,
            "raw memory copies by first mapping source");
    }
    m.update_ranges();
    let (base, count) = heap.raw_page_views(&m.copies)?;
    // A list may submit before the other recording list that numbered a
    // copy. Each must prepare all copies visible at its own bind.
    const READ: usize = usize::MAX - 0x6d;
    // A bounded writer still reads every mapped copy. It must not advance
    // the coverage of a later writer with indirect destinations.
    let (reads, written) = list.raw_memory_ranges(
        &m.directory.0, m.coverage_revision, count as usize, writes == Writes::All,
    );
    for i in reads {
        for &(start, len) in &m.ranges[i] {
            m.copies[i].refresh(list, start, len);
        }
    }
    for i in written {
        for &(start, len) in &m.ranges[i] {
            m.copies[i].gpu_writes(list, start, len);
        }
    }
    writes.owned(&m.pages, |owner, at, len| {
        if owner < count as usize {
            m.copies[owner].gpu_writes(list, at, len);
        }
    });
    #[cfg(feature = "diagnostics")]
    if let Writes::Ranges(ranges) = writes {
        static BOUNDED: AtomicU64 = AtomicU64::new(0);
        let dispatches = BOUNDED.fetch_add(1, Ordering::Relaxed) + 1;
        if dispatches.is_power_of_two() {
            let bytes = ranges.iter().map(|range| range.1).sum::<u64>();
            tracing::info!(frame, dispatches, bytes, visible_copies = count, "bounded direct builder raw write declarations");
        }
    }
    if list.first_use(READ) {
        list.keep(&m.faults.0);
    }
    Some(Ready {
        directory: m.directory.2,
        faults: m.faults.2,
        base,
        count,
    })
}

#[cfg(test)]
#[path = "memory/producer_tests.rs"]
mod producer_tests;

#[cfg(test)]
mod tests {
    use super::{page_byte, Memory, CELL, PAGE, SPECULATIVE_COPIES};

    #[test]
    fn speculation_stops_before_heap_growth_and_confirmed_accesses_can_still_grow() {
        use crate::descriptor::{create_srv, host::HostHeap};
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            assert!(std::env::var_os("VICTORY_D3D12_WARP").is_none());
            return;
        };
        let pages = SPECULATIVE_COPIES as u64 + 6;
        let bytes = (pages * PAGE) as usize;
        let start = (0x300..0x340).map(|i| i * 32 * CELL).find(|&at| unsafe {
            !VirtualAlloc(at as _, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE).is_null()
        }).unwrap();
        let mut memory = Memory::new().unwrap();
        let slot = |m: &Memory, at| unsafe { m.directory.1.add((at / PAGE * 2) as usize).read() };
        for i in 0..SPECULATIVE_COPIES as u64 - 1 {
            let at = start + i * PAGE;
            crate::shadow::register(at as usize, at, PAGE);
            assert!(memory.map(at));
        }
        let last = start + (SPECULATIVE_COPIES as u64 - 1) * PAGE;
        crate::shadow::register(last as usize, last, 3 * PAGE);
        // A nested owner of page 1, plus separate definitely needed allocations.
        for i in [1, 3, 4, 5] {
            let at = last + i * PAGE;
            crate::shadow::register(at as usize, at, PAGE);
        }
        memory.name_constant(last);
        assert_eq!(memory.copies.len(), SPECULATIVE_COPIES);
        assert_eq!(slot(&memory, last + PAGE), 0);
        let initial = heap.raw_page_views(&memory.copies).unwrap();
        let fault = last + 3 * PAGE;
        memory.name_constant(fault);
        assert_eq!(memory.copies.len(), SPECULATIVE_COPIES);
        assert_eq!(slot(&memory, fault), 0);
        assert!(!memory.refused.contains(&(fault / PAGE)));
        assert!(memory.speculative_skips > 0);
        // Run an actual GPU lookup in the production directory and write the
        // missing page to the same feedback bucket that translated shaders use.
        let host = crate::host::host().unwrap();
        let shader = r#"
            ByteAddressBuffer directory : register(t0);
            RWByteAddressBuffer faults : register(u0);
            cbuffer Args : register(b0) { uint page; uint bucket; };
            [RootSignature("SRV(t0), UAV(u0), RootConstants(num32BitConstants=2,b0)")]
            [numthreads(1,1,1)] void main() {
                if (!directory.Load(page * 8)) faults.Store(bucket * 4, page);
            }"#;
        let code = crate::host::compile(shader, windows::core::s!("main"), windows::core::s!("cs_5_1")).unwrap();
        let root: windows::Win32::Graphics::Direct3D12::ID3D12RootSignature =
            unsafe { host.device.CreateRootSignature(0, &code) }.unwrap();
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_COMPUTE_PIPELINE_STATE_DESC, D3D12_SHADER_BYTECODE, ID3D12PipelineState,
        };
        let pso: ID3D12PipelineState = unsafe {
            host.device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: code.as_ptr().cast(), BytecodeLength: code.len() },
                ..Default::default()
            })
        }.unwrap();
        let page = (fault / PAGE) as u32;
        let bucket = 1 + page % (victory_shader::emit::MEMORY_FAULTS - 1);
        let lookup = |m: &Memory| {
            let list = host.new_list().unwrap();
            unsafe {
                list.list.SetComputeRootSignature(&root);
                list.list.SetPipelineState(&pso);
                list.list.SetComputeRootShaderResourceView(0, m.directory.2);
                list.list.SetComputeRootUnorderedAccessView(1, m.faults.2);
                list.list.SetComputeRoot32BitConstants(2, 2, [page, bucket].as_ptr().cast(), 0);
                list.list.Dispatch(1, 1, 1);
            }
            list.close();
            host.execute(&[&list]);
            unsafe { m.faults.1.add(bucket as usize).read_volatile() }
        };
        assert_eq!(lookup(&memory), page);
        memory.take_faults();
        assert_eq!(slot(&memory, fault) as usize, SPECULATIVE_COPIES + 1);
        assert_eq!(memory.from_faults, 1);
        assert_eq!(lookup(&memory), 0);
        let typed = last + 4 * PAGE;
        let mut descriptor = [0u32; 8];
        let handle = descriptor.as_mut_ptr() as u64;
        let view = [0, 11, 5768, 0, typed as u32, (typed >> 32) as u32, 0, 0, 0, 0];
        create_srv(std::ptr::null(), std::ptr::null(), &view, handle);
        heap.mirror(1, handle, false).unwrap();
        let direct = last + 5 * PAGE;
        memory.prepare_addresses(&heap, &[last, direct], &[]);
        assert_ne!(slot(&memory, typed), 0);
        assert_ne!(slot(&memory, direct), 0);
        assert_ne!(slot(&memory, last + PAGE), slot(&memory, last));
        assert_ne!(slot(&memory, last + PAGE), 0);
        assert_eq!(slot(&memory, last + 2 * PAGE), slot(&memory, last));
        let grown = heap.raw_page_views(&memory.copies).unwrap();
        assert_ne!(grown.0, initial.0);
        assert!(grown.1 as usize > SPECULATIVE_COPIES);
        drop(memory);
        drop(heap);
        for i in 0..pages {
            let at = start + i * PAGE;
            crate::shadow::unregister(at as usize, at);
        }
        assert_ne!(unsafe { VirtualFree(start as _, 0, MEM_RELEASE) }, 0);
    }

    #[test]
    fn bounded_builder_raw_writes_follow_page_owners_and_gpu_address_width() {
        use super::{BTreeMap, MEMORY_PAGES, Writes};
        let limit = u64::from(MEMORY_PAGES - 1) * PAGE;
        let pages = BTreeMap::from([(0, 6), (1, 4), (2, 9), (limit / PAGE - 1, 8)]);
        let parts = |ranges: [(u64, u64); 2]| {
            let mut output = Vec::new();
            Writes::Ranges(ranges).owned(&pages, |owner, at, len| output.push((owner, at, len)));
            output
        };
        // High tag bits have no effect in the shader. Neighboring pages
        // can have distinct owners, and a range can end on an unmapped page.
        assert_eq!(
            parts([((0xab << 40) | (2 * PAGE - 4), 24), (3 * PAGE - 4, 24)]),
            [(4, 2 * PAGE - 4, 4), (9, 2 * PAGE, 20), (9, 3 * PAGE - 4, 4)]
        );
        // The clamped directory entry is null; wrapping the shader's
        // 40-bit address can make the trailing dwords accessible again.
        assert_eq!(
            parts([(limit - 4, 24), ((1 << 40) - 4, 24)]),
            [(8, limit - 4, 4), (6, 0, 20)]
        );
    }

    #[test]
    fn raw_builder_maps_headers_named_by_the_view_heap_without_a_shader_table() {
        use crate::descriptor::{create_srv, host::HostHeap};
        use std::ptr::null;
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };

        let Some(heap) = HostHeap::new(false, 4) else {
            assert!(std::env::var_os("VICTORY_D3D12_WARP").is_none(), "requested WARP view heap unavailable");
            return;
        };
        let mut memory = Memory::new().unwrap();
        // Reserve three 4 MiB cells in guest address space. Only the first
        // page of each is a title buffer; the middle one has no AS descriptor.
        let start = (0x1300..0x1400).map(|i| i * 4 * CELL).find(|&at| unsafe {
            !VirtualAlloc(at as _, (3 * CELL) as usize, MEM_RESERVE, PAGE_READWRITE).is_null()
        }).unwrap();
        for at in [start, start + CELL, start + 2 * CELL] {
            assert_eq!(unsafe { VirtualAlloc(at as _, PAGE as usize, MEM_COMMIT, PAGE_READWRITE) } as u64, at);
            crate::shadow::register(at as usize, at, PAGE);
        }
        let directory_slot = |m: &Memory, at| unsafe {
            m.directory.1.add((at / PAGE * 2) as usize).read()
        };
        let mut slot = [0u32; 8];
        let handle = slot.as_mut_ptr() as u64;
        let install = |address: u64| {
            let desc = [0, 11, 5768, 0, address as u32, (address >> 32) as u32, 0, 0, 0, 0];
            create_srv(null(), null(), &desc, handle);
            heap.mirror(1, handle, false).unwrap();
        };
        // No table, root address, constants, or previous GPU fault supply
        // these addresses. The two buffers are several MiB apart.
        install(start + 256);
        memory.prepare_addresses(&heap, &[], &[]);
        let first = directory_slot(&memory, start);
        assert_ne!(first, 0);
        assert_eq!(memory.copies[(first - 1) as usize].span(), (start, PAGE));
        assert_eq!(directory_slot(&memory, start + CELL), 0);
        assert_eq!(directory_slot(&memory, start + 2 * CELL), 0);
        install(start + 2 * CELL + 256);
        memory.prepare_addresses(&heap, &[], &[]);
        let second = directory_slot(&memory, start + 2 * CELL);
        assert_ne!(second, 0);
        assert_eq!(memory.copies[(second - 1) as usize].span(), (start + 2 * CELL, PAGE));
        assert_eq!(directory_slot(&memory, start + CELL), 0);
        assert_eq!(memory.copies.len(), 2);
        assert!(heap.raw_page_views(&memory.copies).is_some());
        // A shifted scalar points to the uncommitted last page before the
        // middle cell. Its neighbor is valid but the scalar gives no reason
        // to allocate that neighbor. A GPU fault there is definite.
        let scalar = ((start + CELL - PAGE) >> 8) as u32;
        memory.prepare_addresses(&heap, &[], &[scalar]);
        assert_eq!(directory_slot(&memory, start + CELL), 0);
        assert_eq!(memory.copies.len(), 2);
        unsafe { memory.faults.1.add(1).write(((start + CELL) / PAGE) as u32) };
        memory.take_faults();
        assert_ne!(directory_slot(&memory, start + CELL), 0);
        assert_eq!((memory.from_roots, memory.from_constants, memory.from_faults), (2, 0, 1));
        drop(memory);
        // Both encodings of a real address still seed their current and
        // following cells before any shader can fault.
        let empty_heap = HostHeap::new(false, 4).unwrap();
        let mut memory = Memory::new().unwrap();
        memory.prepare_addresses(&empty_heap, &[], &[(start >> 8) as u32]);
        assert_ne!(directory_slot(&memory, start), 0);
        assert_ne!(directory_slot(&memory, start + CELL), 0);
        assert_eq!(directory_slot(&memory, start + 2 * CELL), 0);
        let middle = start + CELL;
        memory.prepare_addresses(&empty_heap, &[], &[middle as u32, (middle >> 32) as u32]);
        assert_ne!(directory_slot(&memory, start + 2 * CELL), 0);
        assert_eq!(memory.from_constants, 3);
        drop(memory);
        for at in [start, start + CELL, start + 2 * CELL] {
            crate::shadow::unregister(at as usize, at);
        }
        assert_ne!(unsafe { VirtualFree(start as _, 0, MEM_RELEASE) }, 0);
    }

    #[test]
    fn directory_pages_index_the_same_bytes_as_bound_buffer_views() {
        let heap = 0x1_0000_0000;
        let start = heap + 0x1230;
        let len = 3 * PAGE - 0x1230;
        // The first full guest page is at resource byte 0x10000, even
        // though it lies just 0xedd0 bytes into the guest allocation.
        assert_eq!(page_byte(heap + PAGE, start, len, 0x1230), Some(0x10000));
        assert_eq!(page_byte(heap + 2 * PAGE, start, len, 0x1230), Some(0x20000));
        assert_eq!(page_byte(heap + PAGE, start, len, 0), Some(0xedd0));
        assert_eq!(page_byte(heap, start, len, 0x1230), None);
        assert_eq!(page_byte(heap + 3 * PAGE, start, len, 0x1230), None);
        assert_eq!(page_byte(heap + PAGE, start, len, u32::MAX.into()), None);
    }
}
