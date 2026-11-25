use super::*;
use windows::core::s;
use windows_sys::Win32::System::Memory::*;

// Exercise the production page directory and heap run. Inspect the buffers
// directly on the GPU before any shadow refresh/copy-back can hide the owner.
#[test]
fn a_raw_gpu_write_follows_the_recorded_owner_of_a_reused_page() {
    let Some(host) = host() else {
        assert!(std::env::var_os("VICTORY_D3D12_WARP").is_none());
        return;
    };
    // A page in the address range representable by the guest directory.
    let page = (0x1500..0x1600)
        .map(|i| i * 4 * CELL)
        .find(|&at| unsafe {
            !VirtualAlloc(
                at as _,
                PAGE as usize,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
            .is_null()
        })
        .expect("reserve a guest page");
    let field = page + 0x180;
    let old_owner = 1;
    let new_owner = 2;
    let cpu_pointer = 0x5123_abcd_u32;
    unsafe { (field as *mut u32).write(cpu_pointer) };
    let mut memory = Memory::new().unwrap();
    let heap = HostHeap::new(false, 4).unwrap();
    let source = r#"
        ByteAddressBuffer directory : register(t0);
        RWByteAddressBuffer pages[2048] : register(u0);
        cbuffer Args : register(b0) { uint page; uint offset; uint marker; };
        [RootSignature("SRV(t0), DescriptorTable(UAV(u0,numDescriptors=2048)), RootConstants(num32BitConstants=3,b0)")]
        [numthreads(1,1,1)] void main() {
            uint2 entry = directory.Load2(page * 8);
            if (entry.x) pages[entry.x].Store(entry.y + offset, marker);
        }"#;
    let code = crate::host::compile(source, s!("main"), s!("cs_5_1")).unwrap();
    let root: ID3D12RootSignature = unsafe { host.device.CreateRootSignature(0, &code) }.unwrap();
    let pso: ID3D12PipelineState = unsafe {
        host.device
            .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: code.as_ptr().cast(),
                    BytecodeLength: code.len(),
                },
                ..Default::default()
            })
    }
    .unwrap();
    let record = |memory: &mut Memory, marker: u32| {
        memory.reconcile().unwrap();
        memory.map(field);
        let (base, count) = heap.raw_page_views(&memory.copies).unwrap();
        assert_eq!(count as usize, memory.copies.len());
        let list = host.new_list().unwrap();
        list.keep(&memory.directory.0);
        let args = [(page / PAGE) as u32, (field - page) as u32, marker];
        unsafe {
            list.list.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
            list.list.SetComputeRootSignature(&root);
            list.list.SetPipelineState(&pso);
            list.list
                .SetComputeRootShaderResourceView(0, memory.directory.2);
            list.list
                .SetComputeRootDescriptorTable(1, heap.gpu_handle(base));
            list.list
                .SetComputeRoot32BitConstants(2, 3, args.as_ptr().cast(), 0);
            list.list.Dispatch(1, 1, 1);
        }
        list.close();
        list
    };
    crate::shadow::register(old_owner, page, PAGE);
    let retired = crate::shadow::find(field, 1).unwrap();
    let first = record(&mut memory, cpu_pointer);
    // Fill the first reservation cheaply with views of the same backing.
    // Rebinding the next owner must grow the run while the first list lives.
    memory.copies.resize(1023, retired.clone());
    memory.ranges.resize_with(1023, Vec::new);
    crate::shadow::unregister(old_owner, page);
    crate::shadow::register(new_owner, page, PAGE);
    let current = crate::shadow::find(field, 1).unwrap();
    assert_ne!(retired.id(), current.id());
    let bvh_word = 0x1220_1260;
    let second = record(&mut memory, bvh_word);
    memory.update_ranges();
    assert!(memory.ranges[0].is_empty());
    assert_eq!(memory.ranges[1023], [(page, PAGE)]);
    // Submit only after both generations have recorded their raw writes.
    host.execute(&[&first]);
    host.execute(&[&second]);
    let readback = host.read_back_buffer(24).unwrap();
    let copy = host.new_list().unwrap();
    for (i, backing) in [&retired, &current].into_iter().enumerate() {
        unsafe {
            copy.list.CopyBufferRegion(
                &readback,
                12 * i as u64,
                &backing.buffer,
                backing.offset(field) - 4,
                12,
            )
        };
    }
    copy.close();
    host.execute(&[&copy]);
    let mut bytes = std::ptr::null_mut();
    let words = unsafe {
        readback
            .Map(
                0,
                Some(&D3D12_RANGE { Begin: 0, End: 24 }),
                Some(&mut bytes),
            )
            .unwrap();
        let words = bytes.cast::<[u32; 6]>().read();
        readback.Unmap(0, Some(&D3D12_RANGE::default()));
        words
    };
    let cpu_word = unsafe { (field as *const u32).read() };
    crate::shadow::unregister(new_owner, page);
    assert_ne!(unsafe { VirtualFree(page as _, 0, MEM_RELEASE) }, 0);
    assert_eq!(
        cpu_word, cpu_pointer,
        "no CPU shadow publication was scheduled"
    );
    assert_eq!(
        words,
        [0, cpu_pointer, 0, 0, bvh_word, 0],
        "GPU readback of retired and current backing: {words:08x?}"
    );
}
