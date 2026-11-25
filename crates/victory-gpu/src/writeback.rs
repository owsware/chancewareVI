//! Ownership of guest bytes while GPU buffer writes are in flight. A fence may
//! complete after the title has unmapped those bytes and reused their address.
use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

// Fence completion on the submission thread precedes host-buffer merging.
// Ownership stays live until the publication itself has finished.
static STARTED: AtomicU64 = AtomicU64::new(0);
static PUBLISHED: AtomicU64 = AtomicU64::new(0);
static LAST_CHANGE: AtomicU64 = AtomicU64::new(0);

pub(crate) fn starting(number: u64) {
    STARTED.store(number, Ordering::Release);
}

pub(crate) fn published(number: u64) {
    PUBLISHED.fetch_max(number, Ordering::Release);
}

pub(crate) fn changed_since(number: u64) -> bool {
    LAST_CHANGE.load(Ordering::Acquire) >= number
}

/// Disjoint guest intervals and the most recent submission already started when
/// their physical backing changed. Finished submissions need no history.
#[derive(Default)]
struct Changes {
    intervals: BTreeMap<usize, (usize, u64)>,
    retired: u64,
}

impl Changes {
    fn split(&mut self, at: usize) {
        if let Some((&from, &(to, stamp))) = self.intervals.range(..at).next_back()
            && at < to
        {
            self.intervals.insert(from, (at, stamp));
            self.intervals.insert(at, (to, stamp));
        }
    }

    fn record(&mut self, range: Range<usize>, started: u64, finished: u64) {
        if finished > self.retired {
            self.intervals.retain(|_, (_, stamp)| *stamp > finished);
            self.retired = finished;
        }
        if started <= finished || range.is_empty() {
            return;
        }
        self.split(range.start);
        self.split(range.end);
        let replaced: Vec<_> = self
            .intervals
            .range(range.clone())
            .map(|(&at, _)| at)
            .collect();
        for at in replaced {
            self.intervals.remove(&at);
        }
        self.intervals.insert(range.start, (range.end, started));
    }

    fn intersects_pending(
        &self,
        from: usize,
        to: usize,
        busy: &[(usize, usize)],
        finished: u64,
    ) -> bool {
        if from >= to || busy.is_empty() {
            return false;
        }
        let start = self
            .intervals
            .range(..=from)
            .next_back()
            .map_or(from, |(&at, _)| at);
        self.intervals.range(start..to).any(|(&a, &(b, since))| {
            since > finished && crate::host::touches(busy, a.max(from), b.min(to))
        })
    }
}

static CHANGED: Mutex<Changes> = Mutex::new(Changes {
    intervals: BTreeMap::new(),
    retired: 0,
});

pub(crate) fn remapped(range: Range<usize>) {
    let mut changes = CHANGED.lock().unwrap_or_else(|e| e.into_inner());
    let started = STARTED.load(Ordering::Acquire);
    let finished = PUBLISHED.load(Ordering::Acquire);
    changes.record(range, started, finished);
    if started > finished {
        LAST_CHANGE.store(started, Ordering::Release);
    }
}

/// A newer submission must not reuse an older flight's shadow snapshot
/// where the title has reassigned the underlying physical memory.
pub(crate) fn needs_new_snapshot(from: usize, to: usize, busy: &[(usize, usize)]) -> bool {
    let finished = PUBLISHED.load(Ordering::Acquire);
    if from >= to || LAST_CHANGE.load(Ordering::Acquire) <= finished {
        return false;
    }
    let changes = CHANGED.lock().unwrap_or_else(|e| e.into_inner());
    changes.intersects_pending(from, to, busy, finished)
}

/// A landing may collect several fences. After a remap only new submissions
/// which actually declared this shadow and range can publish there.
type LaterWriters = BTreeMap<u64, Vec<(u64, Range<usize>)>>;

#[derive(Clone)]
pub(crate) struct Owners {
    first: u64,
    later: Option<Arc<LaterWriters>>,
}

impl Owners {
    pub(crate) fn single(first: u64) -> Self {
        Self { first, later: None }
    }

    pub(crate) fn declare(&mut self, number: u64, shadow: u64, ranges: &[(usize, usize)]) {
        if number > self.first {
            let writers = Arc::make_mut(self.later.get_or_insert_with(Arc::default))
                .entry(shadow)
                .or_default();
            writers.extend(ranges.iter().map(|&(from, to)| (number, from..to)));
        }
    }

    /// None means the requested span is still mapped as it was for all the
    /// submissions. Otherwise return just the surviving parts in order.
    fn current(
        &self,
        changes: &Changes,
        shadow: u64,
        range: Range<usize>,
    ) -> Option<Vec<Range<usize>>> {
        let mut cursor = range.start;
        let mut parts = Vec::new();
        let search = changes
            .intervals
            .range(..=range.start)
            .next_back()
            .map_or(range.start, |(&at, _)| at);
        for (&base, &(end, since)) in changes.intervals.range(search..range.end) {
            if since < self.first || end <= range.start {
                continue;
            }
            let from = base.max(range.start);
            let to = end.min(range.end);
            if cursor < from {
                parts.push(cursor..from);
            }
            if let Some(writers) = self.later.as_ref().and_then(|m| m.get(&shadow)) {
                parts.extend(writers.iter().filter_map(|(number, wrote)| {
                    let kept = from.max(wrote.start)..to.min(wrote.end);
                    (*number > since && !kept.is_empty()).then_some(kept)
                }));
            }
            cursor = to;
        }
        if cursor == range.start {
            return None;
        }
        if cursor < range.end {
            parts.push(cursor..range.end);
        }
        // Several newer submissions may have declared the same bytes.
        parts.sort_unstable_by_key(|r| r.start);
        let mut joined: Vec<Range<usize>> = Vec::new();
        for part in parts {
            if let Some(last) = joined.last_mut()
                && part.start <= last.end
            {
                last.end = last.end.max(part.end);
            } else {
                joined.push(part);
            }
        }
        Some(joined)
    }

    pub(crate) fn publish(&self, shadow: u64, address: usize, data: &[u8]) -> usize {
        self.publish_with(&CHANGED, shadow, address, data)
    }

    fn publish_with(
        &self,
        changes: &Mutex<Changes>,
        shadow: u64,
        address: usize,
        data: &[u8],
    ) -> usize {
        if data.is_empty() {
            return 0;
        }
        let Some(end) = address.checked_add(data.len()) else {
            return 0;
        };
        let range = address..end;
        // A backing change cannot race the check and the actual CPU write.
        let _mapped = crate::watch::title_memory_reading_range(range.clone());
        let parts = self.current(
            &changes.lock().unwrap_or_else(|e| e.into_inner()),
            shadow,
            range.clone(),
        );
        let write = |span: Range<usize>| {
            let n = span.len();
            crate::watch::writing(span.start, n);
            crate::host::write_mapped(
                span.start as *mut u8,
                data[span.start - address..].as_ptr(),
                n,
            );
            n
        };
        let Some(parts) = parts else {
            return write(range);
        };
        let bytes = parts.into_iter().map(write).sum();
        #[cfg(feature = "diagnostics")]
        if bytes < data.len() {
            static SPANS: AtomicU64 = AtomicU64::new(0);
            static BYTES: AtomicU64 = AtomicU64::new(0);
            let skipped = (data.len() - bytes) as u64;
            let total = BYTES.fetch_add(skipped, Ordering::Relaxed) + skipped;
            let count = SPANS.fetch_add(1, Ordering::Relaxed) + 1;
            if count <= 16 || count.is_power_of_two() {
                tracing::info!(
                    "stale shader buffer writeback {count}: first submission {}, shadow {shadow}, guest {address:#x}..{end:#x}, {skipped} of {} merged bytes withheld after physical remap ({total} total)",
                    self.first,
                    data.len()
                );
            }
        }
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_late_bvh_write_cannot_replace_the_cpu_object_in_a_reassigned_page() {
        const PAGE: usize = 65536;
        let mut guest = vec![0x25u8; 3 * PAGE];
        let start = guest.as_mut_ptr() as usize;
        let mut gpu = vec![0x80u8; guest.len()];
        let object = PAGE + 0x50;
        let saved = 0x0000_0006_0123_4500u64.to_le_bytes();
        guest[object..object + 8].copy_from_slice(&saved);
        gpu[object..object + 8].copy_from_slice(&0xff80_0000_ff80_0000u64.to_le_bytes());
        // Submission 12 was already in flight at the ownership change.
        let changes = Mutex::new(Changes::default());
        changes
            .lock()
            .unwrap()
            .record(start + PAGE..start + 2 * PAGE, 12, 11);
        let old = Owners::single(12);
        assert_eq!(old.publish_with(&changes, 9, start, &gpu), 2 * PAGE);
        assert_eq!(guest[..PAGE], gpu[..PAGE]);
        assert_eq!(guest[2 * PAGE..], gpu[2 * PAGE..]);
        assert_eq!(guest[object..object + 8], saved);

        // Another flight can write the replacement, but only via its own
        // shadow and only inside the bytes it submitted for writing.
        let mut together = old;
        together.declare(13, 8, &[(start + PAGE, start + 2 * PAGE)]);
        together.declare(13, 9, &[(start + PAGE + 8, start + PAGE + 12)]);
        assert_eq!(
            together.publish_with(&changes, 9, start + PAGE, &gpu[PAGE..2 * PAGE]),
            4
        );
        assert_eq!(guest[PAGE + 8..PAGE + 12], [0x80; 4]);
        assert_eq!(guest[object..object + 8], saved);
        assert_eq!(
            Owners::single(13).publish_with(&changes, 9, start + object, &[7; 8]),
            8
        );
        assert_eq!(guest[object..object + 8], [7; 8]);
    }

    #[test]
    fn overlapping_mapping_changes_preserve_the_latest_owner_on_each_span() {
        let mut changes = Changes::default();
        changes.record(100..200, 10, 4);
        changes.record(150..250, 11, 9);
        let mut owners = Owners::single(9);
        owners.declare(11, 1, &[(125, 210)]);
        assert_eq!(
            owners.current(&changes, 1, 90..260),
            Some(vec![90..100, 125..150, 250..260])
        );
        assert_eq!(Owners::single(12).current(&changes, 2, 90..260), None);
        changes.record(0..1, 11, 11);
        assert!(changes.intervals.is_empty());
    }

    #[test]
    fn fresh_snapshots_only_wait_on_unpublished_writes_to_reassigned_bytes() {
        let mut changes = Changes::default();
        changes.record(100..150, 10, 9);
        changes.record(200..250, 12, 9);
        let busy = [(40, 110), (220, 260)];
        assert!(changes.intersects_pending(105, 130, &busy, 9));
        assert!(!changes.intersects_pending(110, 220, &busy, 9));
        assert!(!changes.intersects_pending(80, 100, &busy, 9));
        assert!(!changes.intersects_pending(80, 150, &busy, 10));
        assert!(changes.intersects_pending(230, 260, &busy, 10));
        assert!(!changes.intersects_pending(230, 260, &busy, 12));
        assert!(!changes.intersects_pending(250, 250, &busy, 9));
    }
}
