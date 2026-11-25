//! Scalar descriptor reads that take a guest address from a computed resource-heap slot.
use super::{Attr, Bin, Fields, Lower, Operand, Reg, RootType, Site, V, Why, pred};

impl Lower<'_> {
    /// Some ray shaders take a four-word descriptor from a computed 32-byte
    /// heap slot and use only its low address and masked high address bits.
    /// A numeric use of any format, stride or record-count bits is a different ABI.
    pub(super) fn picked_address_words(&self, site: Site, code: u8) -> bool {
        let Some(a) = self.scalar.get(&site) else {
            return false;
        };
        let Some(dynamic) = a.dynamic else {
            return false;
        };
        let Some(block) = self.def_block.get(&dynamic) else {
            return false;
        };
        let aligned = self.x.p.blocks[*block]
            .insts
            .iter()
            .find(|i| i.writes.iter().any(|w| w.1 == dynamic))
            .is_some_and(|i| {
                i.name() == "s_lshl_b32"
                    && matches!(&i.fields, Fields::Sop {src, ..}
                if matches!(src.as_slice(), [_, Operand::Const(133) | Operand::Literal(5)]))
            });
        if !aligned
            || !a.offset.is_multiple_of(32)
            || !self
                .x
                .native
                .layout
                .get(a.param)
                .is_some_and(|p| p.root == RootType::ResourceHeap)
        {
            return false;
        }
        let loaded = &self.x.p.blocks[site.0].insts[site.1];
        let (Some(lo), Some(hi)) = (loaded.write(Reg::S(code)), loaded.write(Reg::S(code + 1)))
        else {
            return false;
        };
        let other = [
            loaded.write(Reg::S(code + 2)),
            loaded.write(Reg::S(code + 3)),
        ];
        let (mut masked, mut aligned_pointer) = (false, false);
        for block in &self.x.p.blocks {
            if block
                .phis
                .iter()
                .any(|p| p.args.iter().any(|v| *v == hi || other.contains(&Some(*v))))
            {
                return false;
            }
            for i in &block.insts {
                if i.reads.iter().any(|r| r.1 == lo)
                    && i.name() == "s_and_b32"
                    && matches!(&i.fields, Fields::Sop {src, ..} if src.contains(&Operand::Literal(0xffff_ff00)))
                {
                    aligned_pointer = true;
                }
                if i.reads.iter().any(|r| other.contains(&Some(r.1))) {
                    return false;
                }
                if !i.reads.iter().any(|r| r.1 == hi) {
                    continue;
                }
                if i.name() != "s_and_b32"
                    || !matches!(&i.fields, Fields::Sop {src, ..}
                    if src.iter().any(|o| matches!(o, Operand::Literal(v) if *v != 0 && *v <= 0xffff)))
                {
                    return false;
                }
                masked = true;
            }
        }
        masked && aligned_pointer
    }

    pub(super) fn picked_address(&mut self, site: Site) -> Result<(V, V), Why> {
        let slot = self.heap_index(site, 0, false)?;
        let c = self.cur;
        let handle = self.s.root_srv_handle(crate::emit::SLOT_VIEW_ADDRESSES);
        let undef = self.s.m.undef(self.s.t.i32);
        let zero = self.s.m.i32(0);
        let one = self.s.m.i32(1);
        let two = self.s.m.i32(2);
        let three = self.s.m.i32(3);
        let align = self.s.m.i32(4);
        let count_mask = self.s.m.i8(1);
        let address_mask = self.s.m.i8(3);
        let count = self.s.op(
            c,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            139,
            &[handle, zero, undef, count_mask, align],
        );
        let count = self.s.m.extract(c, count, 0);
        let count = self.as_int(count);
        let inside = self.s.m.cmp(c, pred::IULT, slot, count);
        let record = self.s.m.bin(c, Bin::Add, slot, two);
        let record = self.s.m.select(c, inside, record, one);
        let byte = self.s.m.bin(c, Bin::Shl, record, three);
        let words = self.s.op(
            c,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            139,
            &[handle, byte, undef, address_mask, align],
        );
        let lo = self.s.m.extract(c, words, 0);
        let hi = self.s.m.extract(c, words, 1);
        Ok((self.as_int(lo), self.as_int(hi)))
    }
}

#[cfg(test)]
mod tests {
    use super::super::Lowered;
    use super::super::tests::{S_ENDPGM, lower_compute, lower_with, root};
    use crate::native::{HwStage, ResourceType, RootType, UsedResource};

    #[test]
    fn dynamic_address_only_descriptor_reads_use_the_guest_address_table() {
        let words = [
            0xBE84_0300,
            0xBE85_0380, // s[4:5] = resource heap
            0x8F02_8501, // s2 = root index s1 << 5
            0xF408_0202,
            0x0400_0000, // s_load_dwordx4 s[8:11], s[4:5], s2
            0x870C_FF08,
            0xFFFF_FF00, // low address bits
            0x870D_FF09,
            0x0000_FFFF, // high address bits
            0x7E02_020C,
            0x7E04_020D, // v1 = s12, v2 = s13
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            0xD834_0004,
            0x0000_0200, // ds_write_b32 v0, v2 offset:4
            S_ENDPGM,
        ];
        let compile = |code: &[u32]| {
            let layout = vec![
                root(RootType::ResourceHeap, 0, 0),
                root(RootType::Constants, 1, 1),
            ];
            lower_with(code, layout, vec![])
        };
        let requires_addresses =
            |shader: Lowered| shader.root_srvs.contains(&crate::emit::SLOT_VIEW_ADDRESSES);
        assert!(requires_addresses(compile(&words).unwrap()));
        let mut uses_stride = words;
        uses_stride[8] = 0xFFFF_0000; // the upper half of word 1 is descriptor metadata
        assert!(!compile(&uses_stride).is_ok_and(requires_addresses));
        let mut uses_record_count = words;
        uses_record_count[9] = 0x7E02_020A; // v1 = s10, word 2
        assert!(!compile(&uses_record_count).is_ok_and(requires_addresses));
    }

    #[test]
    #[cfg(windows)]
    fn warp_executes_guest_heap_address_selection_and_bounds_check() {
        use crate::emit::host::{Dispatch, Host, Root, View};
        let words = [
            0xBE84_0300,
            0xBE85_0380, // s[4:5] = resource heap
            0xBE86_0302,
            0xBE87_0380, // s[6:7] = output table
            0x8F02_8501, // s2 = root index s1 << 5
            0xF408_0202,
            0x0400_0000, // s[8:11] = heap[index]
            0x870C_FF08,
            0xFFFF_FF00, // s12 = low address aligned to 256
            0x870D_FF09,
            0x0000_FFFF, // s13 = high 16 address bits
            0x7E02_020C,
            0x7E04_020D, // v1 = s12, v2 = s13
            0xF408_0403,
            0xFA00_0000, // s[16:19] = output UAV
            0xE070_1000,
            0x8004_0100, // buffer_store_dword v1, v0, s[16:19], 0 offen
            0xE070_1004,
            0x8004_0200, // buffer_store_dword v2, v0, s[16:19], 0 offen offset:4
            S_ENDPGM,
        ];
        let layout = vec![
            root(RootType::ResourceHeap, 0, 0),
            root(RootType::Constants, 1, 1),
            root(RootType::Table, 2, 2),
        ];
        let output = UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 2,
            slot: 2,
        };
        let lowered = lower_compute(&words, layout, vec![output], false, false).unwrap();
        let shader = crate::emit::HostShader {
            stage: HwStage::Cs,
            dxil: lowered.dxil,
            user_data: lowered.user_data,
            samplers: lowered.samplers,
            static_samplers: Vec::new(),
            root_cbvs: lowered.root_cbvs,
            root_srvs: lowered.root_srvs,
            root_uavs: lowered.root_uavs,
            written_root_uavs: lowered.written_root_uavs,
            writes_guest_memory: lowered.writes_guest_memory,
            root_indexed_views: Vec::new(),
        };
        assert_eq!(shader.root_srvs, [crate::emit::SLOT_VIEW_ADDRESSES]);
        let _one = crate::emit::valu_tests::ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let Ok(host) = Host::new(true) else { return };
        let (sig, pso) = host.compute_pipeline(&shader).unwrap();
        for (index, expected) in [
            (0, [0x1122_3300, 13]),
            (1, [0x5566_7700, 14]),
            (2, [0, 0]),
            (99, [0, 0]),
        ] {
            let mut data = vec![0; shader.user_data_dwords().max(1) as usize];
            for param in &shader.user_data {
                if param.root == RootType::Constants {
                    data[param.index as usize] = index;
                }
            }
            let addresses = vec![2, 0, 0, 0, 0x1122_3300, 13, 0x5566_7700, 14];
            let outputs = host
                .dispatch(
                    &sig,
                    &pso,
                    &Dispatch {
                        roots: vec![
                            (0, Root::Constants(data)),
                            (
                                1,
                                Root::View {
                                    srv: true,
                                    data: addresses,
                                },
                            ),
                        ],
                        views: &[View::RawUav(2)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap();
            assert_eq!(outputs[0], expected, "heap index {index}");
        }
    }
}
