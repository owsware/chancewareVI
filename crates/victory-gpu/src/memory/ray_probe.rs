//! Fenced snapshots of the selected traversal shader's counters. All host
//! command lists submit to the same D3D12 queue. Capture only the suffix
//! after the fault directory, leaving normal fault collection in place.

use victory_shader::emit::{MEMORY_FAULTS, RAY_PROBE_DWORDS, RAY_PROBE_HASH, ray_probe_enabled};
use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host, transition};
use crate::pipeline::host::HostPipeline;

pub(crate) fn after_dispatch(list: &HostList, pipeline: &HostPipeline) {
    if !ray_probe_enabled() || pipeline.stages[0].shader.hash != RAY_PROBE_HASH {
        return;
    }
    let frame = crate::queue::presented() + 1;
    if !pipeline.stages[0]
        .translated()
        .host
        .root_uavs
        .contains(&victory_shader::emit::SLOT_FAULTS)
    {
        tracing::warn!(
            shader = RAY_PROBE_HASH,
            frame,
            "ray intersection fault UAV unavailable"
        );
        return;
    }
    let guard = super::MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    let faults = guard.as_ref().map(|memory| memory.faults.0.clone());
    drop(guard);
    let Some(readback) = faults.and_then(|faults| copy_counters(list, &faults)) else {
        tracing::warn!(
            shader = RAY_PROBE_HASH,
            frame,
            "ray intersection readback unavailable"
        );
        return;
    };
    list.defer_after(move || {
        let Some(words) = read_counters(&readback) else {
            tracing::warn!(shader = RAY_PROBE_HASH, frame, "ray intersection readback unavailable");
            return;
        };
        let [visits, candidates, node_lo, node_hi, distance, extent, bary_u, bary_v] = words;
        tracing::info!(
            target: "victory_gpu::trace",
            shader = RAY_PROBE_HASH, frame,
            active_node_visits = visits, finite_triangle_candidates = candidates,
            sample_available = candidates != 0,
            first_node_lo = format_args!("{node_lo:#010x}"), first_node_hi = format_args!("{node_hi:#010x}"),
            first_distance_bits = format_args!("{distance:#010x}"), first_extent_bits = format_args!("{extent:#010x}"),
            first_bary_u_numerator_bits = format_args!("{bary_u:#010x}"),
            first_bary_v_numerator_bits = format_args!("{bary_v:#010x}"),
            "cumulative ray intersection counters after GPU fence"
        );
    });
}

fn copy_counters(list: &HostList, faults: &ID3D12Resource) -> Option<ID3D12Resource> {
    let bytes = u64::from(RAY_PROBE_DWORDS) * 4;
    let readback = host()?.read_back_buffer(bytes)?;
    // The preceding dispatch bound faults as a root UAV. Order its writes
    // before the copy, then restore common for subsequent shader accesses.
    // The CPU reads the private snapshot only after this list's fence.
    let uav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    let copy = D3D12_RESOURCE_STATE_COPY_SOURCE;
    unsafe {
        list.list.ResourceBarrier(&[transition(faults, uav, copy)]);
        list.list
            .CopyBufferRegion(&readback, 0, faults, u64::from(MEMORY_FAULTS) * 4, bytes);
        list.list
            .ResourceBarrier(&[transition(faults, copy, D3D12_RESOURCE_STATE_COMMON)]);
    }
    list.keep(faults);
    list.keep(&readback);
    Some(readback)
}

fn read_counters(readback: &ID3D12Resource) -> Option<[u32; RAY_PROBE_DWORDS as usize]> {
    let mut data = std::ptr::null_mut();
    // SAFETY: the readback holds the complete private snapshot. Its list's
    // fence has signaled before this function is called.
    unsafe {
        let range = D3D12_RANGE {
            Begin: 0,
            End: RAY_PROBE_DWORDS as usize * 4,
        };
        readback.Map(0, Some(&range), Some(&mut data)).ok()?;
        let data = std::ptr::NonNull::new(data)?;
        let words = data.cast::<[u32; RAY_PROBE_DWORDS as usize]>().read();
        readback.Unmap(0, Some(&D3D12_RANGE::default()));
        Some(words)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fenced_snapshot_contains_only_the_eight_probe_words() {
        if std::env::var("VICTORY_D3D12_WARP").as_deref() != Ok("1") {
            return;
        }
        let host = host().expect("WARP device");
        assert!(host.is_software());
        let expected = [123, 1, 0xa1234567, 0x1f, 0x41800000, 0x42000000, 17, 23];
        let (faults, mapped, _) =
            super::super::Memory::table(u64::from(MEMORY_FAULTS + RAY_PROBE_DWORDS), "probe test")
                .unwrap();
        let list = host.new_list().unwrap();
        // SAFETY: private mapped buffer, filled before submission. The
        // sentinel immediately preceding the suffix detects offset errors.
        unsafe {
            mapped.add(MEMORY_FAULTS as usize - 1).write(0xfeed1234);
            mapped
                .add(MEMORY_FAULTS as usize)
                .cast::<[u32; 8]>()
                .write(expected);
            list.list.ResourceBarrier(&[transition(
                &faults,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
        }
        let readback = copy_counters(&list, &faults).unwrap();
        let observed = std::sync::Arc::new(std::sync::OnceLock::new());
        let output = observed.clone();
        list.defer_after(move || {
            output.set(read_counters(&readback).unwrap()).unwrap();
        });
        list.close();
        assert!(observed.get().is_none());
        host.execute(&[&list]);
        assert_eq!(observed.get(), Some(&expected));
        assert_eq!(
            unsafe { mapped.add(MEMORY_FAULTS as usize - 1).read() },
            0xfeed1234
        );
    }
}
