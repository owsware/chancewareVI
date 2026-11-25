//! Opt-in scheduling hints for native shaders whose indices come straight from
//! the material record. Read the same GPU buffer as the shader, in command
//! order. The ordinary heap pass still owns validation, uploads and retries.

use std::sync::{
    Arc, Mutex, OnceLock,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use crate::descriptor::{
    Heap, View,
    host::{HostHeap, buffer_copy},
};
use crate::host::HostList;
use windows::Win32::Graphics::Direct3D12::D3D12_RANGE;

const LIMIT: usize = 8;
const MAX_AGE: u64 = 4;
const TABLE_DWORDS: u64 = 688;
const SLOT_BYTES: u64 = 32;

pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("VICTORY_MATERIAL_TEXTURE_DEMAND").is_some_and(|v| v == "1")
    })
}

pub(crate) fn fields(hash: &str) -> Option<&'static [u32]> {
    match hash {
        "d9fcb8a4425fc59b" | "43fc2af1f70920f1" => Some(&[0, 4, 8]),
        "271a87c54ac648a5" => Some(&[0, 4, 12, 16]),
        "d2e683031b764939" => Some(&[0, 4, 8, 16]),
        "43e24028d4a1faef" | "6590393ed93ff071" | "ee78da6862364ddd" | "c10e7d95d8266d2a"
        | "8a5351822b1a4929" | "09eb0a73df63b87d" | "899ef7a2932ae6c1" | "16053957ff2dff5c"
        | "7c18c77e40e164e4" | "47ec43235084505f" | "247a288b2ba23986" | "6d609ecfb830511f" => {
            Some(&[0, 4, 12, 16, 20])
        }
        "573d1857e903fab5" | "8e9b3db8a3eeba51" | "b8df138593588098" | "b5640b1075483a54"
        | "89afedb5b7720e76" | "085d374c5a795549" | "efc089fe381c43bf" | "0fbc5c91bdab02d2"
        | "072f26dae8824688" | "794198ca2f7c9b67" | "cedd7b38d3be97a1" | "8b901a7fb38c169b" => {
            Some(&[0, 4, 12, 16, 20, 24])
        }
        "3ec62ecabb36f1d2" | "3711fdad456538a1" | "de41d84ca748cc29" | "659fd97066bea4fc"
        | "3f93c9225b740fd9" | "26319e68357cef57" => {
            Some(&[0, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44])
        }
        _ => None,
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Mark {
    pub(crate) write: u64,
    pub(crate) descriptor: u64,
}

#[derive(Default)]
pub(crate) struct Mailbox(Arc<Mutex<Vec<Observation>>>);

struct Observation {
    source_slot: u32,
    view: View,
    shadow: std::sync::Weak<crate::shadow::Shadow>,
    frame: u64,
    stamp: Mark,
    slots: Vec<u32>,
}

type RecordKey = (usize, u32, &'static [u32]);

/// Collect a bounded lookahead from this frame's draws for the next. Sorting by
/// material offset gives each recurring record a turn even if draw order never
/// changes. The title may rotate root table slots that still bind the same material
/// buffer. The actual bound table, record and buffer are verified when it is read.
#[derive(Default)]
struct Sampler {
    frame: Option<u64>,
    cursor: Option<RecordKey>,
    active: Vec<RecordKey>,
    next: Vec<RecordKey>,
    wrap: Vec<RecordKey>,
}

impl Sampler {
    fn keep_smallest(batch: &mut Vec<RecordKey>, key: RecordKey) {
        if let Err(at) = batch.binary_search(&key)
            && at < LIMIT
        {
            if batch.len() == LIMIT {
                batch.pop();
            }
            batch.insert(at, key);
        }
    }

    fn select(&mut self, frame: u64, key: RecordKey) -> bool {
        if self.frame.is_some_and(|old| frame < old) {
            return false;
        }
        if self.frame != Some(frame) {
            self.active.clear();
            self.active.append(&mut self.next);
            let left = LIMIT - self.active.len();
            self.active.extend(self.wrap.drain(..).take(left));
            self.cursor = self.active.last().copied();
            self.frame = Some(frame);
        }
        if self.cursor.is_none_or(|last| key > last) {
            Self::keep_smallest(&mut self.next, key);
        } else {
            Self::keep_smallest(&mut self.wrap, key);
        }
        if let Some(at) = self.active.iter().position(|&selected| selected == key) {
            self.active.swap_remove(at);
            true
        } else {
            false
        }
    }
}

pub(crate) fn wants(heap: &Heap, offset: u32, fields: &'static [u32]) -> bool {
    static SAMPLER: OnceLock<Mutex<Sampler>> = OnceLock::new();
    let mut sampler = SAMPLER
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    sampler.select(
        crate::frame::current(),
        (heap as *const Heap as usize, offset, fields),
    )
}

static RECORDED: AtomicUsize = AtomicUsize::new(0);
static OBSERVED: AtomicUsize = AtomicUsize::new(0);
static PROMOTED_PASSES: AtomicUsize = AtomicUsize::new(0);
static OUTSTANDING: AtomicUsize = AtomicUsize::new(0);

struct Ticket(AtomicBool);
impl Ticket {
    fn take() -> Option<Self> {
        OUTSTANDING
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                (n < LIMIT).then_some(n + 1)
            })
            .ok()?;
        Some(Self(AtomicBool::new(true)))
    }
    fn finish(&self) -> bool {
        if self.0.swap(false, Ordering::Relaxed) {
            OUTSTANDING.fetch_sub(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }
}
impl Drop for Ticket {
    fn drop(&mut self) {
        self.finish();
    }
}

fn byte_range(record: u32, fields: &[u32], len: u64) -> Option<(u64, usize)> {
    let bytes = fields.last()?.checked_add(4)?;
    let end = record.checked_add(bytes)?;
    (record.is_multiple_of(4) && u64::from(end) <= len)
        .then_some((u64::from(record), bytes as usize))
}

fn indices(words: &[u32], fields: &[u32], count: u32) -> Vec<u32> {
    let mut slots = Vec::new();
    for field in fields {
        if let Some(&word) = words.get((field / 4) as usize) {
            // The guest shifts the packed index left five bits in a 32-bit register.
            let slot = word.wrapping_shl(5) >> 5;
            if slot < count && !slots.contains(&slot) {
                slots.push(slot);
            }
        }
    }
    slots
}

pub(crate) fn record(
    list: &HostList,
    heap: &Heap,
    table: u64,
    record: u32,
    fields: &'static [u32],
) {
    let Some(handle) = table.checked_add(TABLE_DWORDS * 4) else {
        return;
    };
    let Some(slot) = heap.index(handle) else {
        return;
    };
    if heap.handle(slot) != Some(handle) || !table.is_multiple_of(SLOT_BYTES) {
        return;
    }
    let frame = crate::frame::current();
    let Some(ticket) = Ticket::take() else { return };
    let Some(host) = heap.host() else { return };
    let stamp = host.material_write_stamp();
    let Some(view @ View::Placed(false, _, _)) = crate::descriptor::checked_view(handle) else {
        return;
    };
    let Some((source, address, len)) = buffer_copy(&view) else {
        return;
    };
    let Some((at, bytes)) = byte_range(record, fields, len) else {
        return;
    };
    let Some(at) = address.checked_add(at) else {
        return;
    };
    if source.offset_in(at).is_none()
        || source.left(at) < bytes as u64
        || host.shows(slot, &source.buffer) != Some(true)
    {
        return;
    }
    let Some(gpu) = crate::host::host() else {
        return;
    };
    let Some(back) = gpu.read_back_buffer(bytes as u64) else {
        return;
    };
    let shadow = Arc::downgrade(&source);
    let mailbox = Arc::downgrade(&host.material_demand.0);
    let count = host.count();
    // SAFETY: the verified range is in the title buffer in use as an SRV.
    // Like the title's indirect-record reads, its buffer promotes to COPY_SOURCE.
    // Both resources survive command execution and the completion callback.
    unsafe {
        list.list
            .CopyBufferRegion(&back, 0, &source.buffer, source.offset(at), bytes as u64);
    }
    list.keep(&back);
    list.keep(&source.buffer);
    RECORDED.fetch_add(1, Ordering::Relaxed);
    list.defer_after(move || {
        if !ticket.finish() || frame.abs_diff(crate::frame::current()) > MAX_AGE {
            return;
        }
        let Some(mailbox) = mailbox.upgrade() else {
            return;
        };
        let mut observations = mailbox.lock().unwrap_or_else(|e| e.into_inner());
        if observations.len() == LIMIT {
            return;
        }
        let mut data = std::ptr::null_mut();
        let read = D3D12_RANGE {
            Begin: 0,
            End: bytes,
        };
        // SAFETY: execution is complete; bytes is a multiple of four and is the full allocation.
        unsafe {
            if back.Map(0, Some(&read), Some(&mut data)).is_err() {
                return;
            }
            let slots = if data.is_null() {
                Vec::new()
            } else {
                let words = std::slice::from_raw_parts(data.cast::<u32>(), bytes / 4);
                indices(words, fields, count)
            };
            back.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
            if !slots.is_empty() {
                let n = OBSERVED.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_power_of_two() {
                    tracing::info!(
                        frame,
                        observed = n,
                        recorded = RECORDED.load(Ordering::Relaxed),
                        promoted_passes = PROMOTED_PASSES.load(Ordering::Relaxed),
                        "GPU material records read for texture scheduling"
                    );
                }
                observations.push(Observation {
                    source_slot: slot,
                    view: view.clone(),
                    shadow: shadow.clone(),
                    stamp,
                    frame,
                    slots,
                });
            }
        }
    });
}

impl Mailbox {
    pub(crate) fn take(&self, heap: &Heap, host: &HostHeap, frame: u64) -> Vec<(u32, u64, bool)> {
        let observations = std::mem::take(&mut *self.0.lock().unwrap_or_else(|e| e.into_inner()));
        let mut priority = Vec::new();
        for obs in observations {
            if frame.abs_diff(obs.frame) > MAX_AGE
                || heap
                    .handle(obs.source_slot)
                    .and_then(crate::descriptor::checked_view)
                    .as_ref()
                    != Some(&obs.view)
                || !obs
                    .shadow
                    .upgrade()
                    .is_some_and(|s| host.shows(obs.source_slot, &s.buffer) == Some(true))
            {
                continue;
            }
            for slot in obs.slots {
                if let Some(item) = host.demand_material_texture(
                    heap,
                    slot,
                    obs.stamp,
                    frame,
                ) && !priority.contains(&item)
                {
                    priority.push(item);
                }
            }
        }
        priority
    }
}

// Reorder only existing due entries. Keep at least three normal slots
// between each promoted write and rotate who goes first across frames.
pub(crate) fn promote(due: &mut [(u32, u64, bool)], wanted: &[(u32, u64, bool)], frame: u64) -> usize {
    let mut moved = 0;
    let mut at = (frame % 4) as usize;
    for item in wanted {
        if at >= due.len() {
            break;
        }
        if let Some(offset) = due[at..].iter().position(|v| v == item) {
            due[at..=at + offset].rotate_right(1);
            moved += usize::from(offset > 0);
            at += 4;
        }
    }
    if moved > 0 {
        PROMOTED_PASSES.fetch_add(1, Ordering::Relaxed);
    }
    moved
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn late_records_get_turns_with_repeated_draws_and_bounded_lookahead() {
        let count = 83u32;
        let mut sampler = Sampler::default();
        let mut seen = vec![false; count as usize];
        for frame in 0..=(u64::from(count).div_ceil(LIMIT as u64)) {
            let mut accepted = 0;
            for ordinal in (0..count).rev() {
                let key = (1, ordinal * 64, &[0u32][..]);
                if sampler.select(frame, key) {
                    seen[ordinal as usize] = true;
                    accepted += 1;
                }
                assert!(
                    !sampler.select(frame, key),
                    "same record twice in one frame"
                );
                assert!(
                    sampler.active.len() <= LIMIT
                        && sampler.next.len() <= LIMIT
                        && sampler.wrap.len() <= LIMIT
                );
            }
            assert!(accepted <= LIMIT);
        }
        assert!(
            seen.iter().all(|&sampled| sampled),
            "including the last submitted record"
        );
    }

    #[test]
    fn sampled_fields_use_guest_shift_and_only_verified_words() {
        let fields = fields("43e24028d4a1faef").unwrap();
        let w = [0xf8000000, 17, 2, 0xf8000011, 99, 5];
        assert_eq!(indices(&w, fields, 20), [0, 17, 5]);
        assert_eq!(byte_range(12, fields, 36), Some((12, 24)));
        assert_eq!(byte_range(12, fields, 35), None);
        assert_eq!(byte_range(u32::MAX - 3, fields, u64::MAX), None);
        assert_eq!(byte_range(3, fields, 36), None);
    }

    #[test]
    fn only_current_existing_due_writes_move_ahead() {
        let mut due: Vec<_> = (0..10u32).map(|n| (n, u64::from(n) + 1, false)).collect();
        promote(&mut due, &[(40, 41, false), (8, 8, false), (8, 9, false), (9, 10, false)], 0);
        assert_eq!(
            due.iter().map(|v| v.0).collect::<Vec<_>>(),
            [8, 0, 1, 2, 9, 3, 4, 5, 6, 7]
        );
    }
}
