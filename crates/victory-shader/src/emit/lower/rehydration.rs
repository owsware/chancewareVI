//! The 64-thread Box32/Box16 rehydration kernels selected by the container digest.
//!
//! A short dispatch still needs both physical Wave32 halves at the three entry
//! collectives. Mask the packed-header load and the original EXEC count, so
//! inactive lanes contribute zero. Retire them only after the last exchange.
//! The rest of the kernel uses private LDS rows: lane `t` addresses row `24*t`,
//! with 24 dwords (Box32) or halfwords (Box16). The child loops address six
//! elements per child of a four-child node; the packed override loop addresses
//! only set bits of the low 24 header bits. No later operation needs the other
//! half or a group barrier.

use super::{Bin, EXEC, Fields, Inst, Lower, Operand, Parts, Reg, StageIo, Stop, Term, V, Val, pred, unsupported};

#[derive(Clone, Copy)]
pub(super) struct Split {
    count: usize,
    count_reg: u8,
    header: usize,
    header_value: crate::ir::Value,
    body: usize,
    tail: Option<(V, V)>,
}

impl Split {
    /// Call only after the digest, console wave size and host cap checks.
    pub(super) fn identify(x: &Parts<'_>) -> Option<Self> {
        if !matches!(x.native.io, StageIo::Cs(cs) if cs.threads == [64, 1, 1])
            || !x.st.wide.is_empty()
            || super::super::group_sync() == 0
            || x.p.blocks.iter().flat_map(|b| &b.insts).any(|i| i.name() == "s_barrier")
        {
            return None;
        }
        let entry = &x.p.blocks.first()?.insts;
        let at = |offset, name| -> Option<(usize, &Inst)> {
            entry.iter().enumerate().find(|(_, i)| i.machine.offset == offset && i.name() == name)
        };
        [(0x78, 0x7c, 0x1dc), (0x7c, 0x80, 0x1e0)].into_iter().find_map(|(count, header, body)| {
            let (count, c) = at(count, "s_bcnt1_i32_b64")?;
            let (header, h) = at(header, "buffer_load_dword")?;
            let (body, _) = at(body, "buffer_load_dwordx4")?;
            let Fields::Sop { dst: Some(Operand::Scalar { code: count_reg, .. }), src, .. } = &c.fields else {
                return None;
            };
            if !matches!(src.first(), Some(Operand::Scalar { code: EXEC, .. }))
                || x.st.guards[0][header].is_some()
            {
                return None;
            }
            Some(Self { count, count_reg: *count_reg, header, header_value: h.write(Reg::V(12))?, body, tail: None })
        })
    }
}

impl Lower<'_> {
    /// Leave all physical lanes running through the entry collectives. The
    /// host dispatch's X group index is bounded; `first = group * 64` and
    /// `first + lane` cannot wrap.
    pub(super) fn rehydration_prologue(&mut self, limit: V, first: V, active: V) -> bool {
        let Some(split) = &mut self.rehydration else { return false };
        let e = self.s.entry;
        let (zero, width) = (self.s.m.i32(0), self.s.m.i32(64));
        let left = self.s.m.bin(e, Bin::Sub, limit, first);
        let short = self.s.m.cmp(e, pred::IULT, left, width);
        let bounded = self.s.m.select(e, short, left, width);
        let in_range = self.s.m.cmp(e, pred::IULT, first, limit);
        let bounded = self.s.m.select(e, in_range, bounded, zero);
        let unlimited = self.s.m.cmp(e, pred::EQ, limit, zero);
        let count = self.s.m.select(e, unlimited, width, bounded);
        split.tail = Some((active, count));
        true
    }

    pub(super) fn rehydration_inst(&mut self, b: usize, i: usize) -> Result<bool, Stop> {
        let Some(split) = self.rehydration.filter(|_| b == 0) else { return Ok(false) };
        let (active, count) = split.tail.expect("rehydration entry predicate");
        let inst = &self.x.p.blocks[b].insts[i];
        if i == split.count {
            self.set(inst.write(Reg::S(split.count_reg)), Val::I32(count));
            let zero = self.s.m.i32(0);
            let nonzero = self.s.m.cmp(self.cur, pred::NE, count, zero);
            self.set(inst.write(Reg::Scc), Val::Bit(nonzero));
            return Ok(true);
        }
        if i == split.header {
            let fail = |why| unsupported(inst.machine.offset, why);
            let guard = self.begin_guarded(Some(active));
            self.buffer(inst, b, i).map_err(fail)?;
            let header = self.i32_of(split.header_value).map_err(fail)?;
            let zero = self.s.m.i32(0);
            let masked = self.end_guarded(guard, &[(header, zero)])[0];
            self.set(Some(split.header_value), Val::I32(masked));
            return Ok(true);
        }
        if i == split.body {
            // Even the rest of IR block zero no longer permits exchanges.
            self.exchange.as_mut().expect("rehydration exchange").2[0] = false;
            let body = self.s.m.new_block();
            let done = self.s.m.new_block();
            self.s.m.terminate(self.cur, Term::CondBr(active, body, done));
            self.s.m.terminate(done, Term::Ret);
            self.cur = body;
        }
        Ok(false)
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;
    use crate::emit::host::{Dispatch, Host, Root, View};
    use crate::emit::{HostShader, validator, wave};
    use crate::native::{ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, UsedResource};
    use crate::{io, ir, resource, structure};

    /// A synthetic 64-lane kernel with the same four entry results as Box32,
    /// but it only writes those results to consecutive output rows.
    fn program() -> NativeProgram {
        let mut words = vec![
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202, 0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
            0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        ];
        words.resize(0x78 / 4, 0xBF80_0000); // register the split entry layout
        words.extend([
            0xBE94_107E, // s_bcnt1_i32_b64 s20, exec
            0xE030_1000, 0x8002_0C01, // buffer_load_dword v12, v1, s[8:11], 0 offen
            0x2C0E_1898, // v_lshrrev_b32 v7, 24, v12
            0x3610_18FF, 0x00FF_FFFF, // v_and_b32 v8, 0xffffff, v12
            0xD764_0008, 0x0001_0108, // v_bcnt_u32_b32 v8, v8, 0
        ]);
        words.extend(wave::WAVE_OR_V7_TO_S7.iter().map(|&w| {
            if w >> 25 == 28 { w & 0x01FF_FFFF | 37 << 25 } else { w }
        }));
        for (src, dest) in [(7, 4), (8, 5)] {
            words.extend([
                0xBE84_25C1, // s_or_saveexec_b64 s[4:5], -1
                0xD501_0001, 0x0012_0080 | src << 9, // v_cndmask_b32 v1, 0, vSrc, s[4:5]
                0x7E0C_0301, // v_mov_b32 v6, v1
            ]);
            for shift in [1, 2, 4, 8] {
                words.extend([
                    0x4A0C_0CFA, // v_add_nc_u32 v6, v6, v6 row_shr:shift
                    (if shift == 1 { 0xFF09_1106 } else { 0xFF01_1006 }) | shift << 8,
                ]);
            }
            words.extend([
                0xBEFE_03FF, 0xFFFF_0000, // s_mov_b32 exec_lo, 0xffff0000
                0xBEFF_03FF, 0xFFFF_0000, // s_mov_b32 exec_hi, 0xffff0000
                0xD778_4802, 0x0305_8306, // v_permlanex16_b32 v2, v6, -1, -1 fi:1
                0x4A0C_0D02, // v_add_nc_u32 v6, v2, v6
                0xD760_0006, 0x0001_3F06, // v_readlane_b32 s6, v6, 31
                0xBEFE_0380, 0xBEFF_03C1, // exec = 0xffffffff00000000
                0x4A0C_0C06, // v_add_nc_u32 v6, s6, v6
                0xBEFE_03C1, 0xBEFF_03C1, // exec = -1
                0x4C0C_0306, // v_sub_nc_u32 v6, v6, v1
                0xBEFE_0404, // s_mov_b64 exec, s[4:5]
                0x7E00_0306 | dest << 17, // v_mov_b32 vDest, v6
            ]);
        }
        assert!(words.len() <= 0x1dc / 4);
        words.resize(0x1dc / 4, 0xBF80_0000);
        words.extend([
            0xE038_0000, 0x8002_0C00, // buffer_load_dwordx4 v[12:15], off, s[8:11], 0
            0x3402_0084, // v_lshlrev_b32 v1, 4, v0
            0x7E04_0214, // v_mov_b32 v2, s20
            0x7E06_0207, // v_mov_b32 v3, s7
            0xE078_1100, 0x8002_0201, // buffer_store_dwordx4 v[2:5], v1, s[8:11], 0 offen offset:256
            0xBF81_0000, // s_endpgm
        ]);
        NativeProgram {
            stage: HwStage::Cs,
            wave64: true,
            vgprs: 16,
            layout: vec![RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            }],
            used: vec![UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 4,
                slot: 1,
            }],
            io: StageIo::Cs(ComputeIo { rsrc2: 5 << 1, threads: [64, 1, 1] }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        }
    }

    #[test]
    fn partial_wave64_rehydration_preserves_count_total_and_both_prefixes() {
        let native = program();
        let p = ir::build(&native.code, native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let was = super::super::TEST_HOST_WAVE.replace(Some((32, 32)));
        let lowered = super::super::lower_with_thread_limit(&Parts {
            native: &native, p: &p, st: &st, io: &io, res: &res,
            inputs: Vec::new(), outputs: Vec::new(), prims: Vec::new(), mesh_stride: 0,
        }, true);
        super::super::TEST_HOST_WAVE.set(was);
        let lowered = lowered.unwrap();
        assert_eq!(lowered.wave_size, Some(32));
        let mut shader = HostShader {
            stage: HwStage::Cs,
            dxil: lowered.dxil,
            user_data: lowered.user_data,
            samplers: Vec::new(),
            static_samplers: Vec::new(),
            root_cbvs: lowered.root_cbvs,
            root_srvs: lowered.root_srvs,
            root_uavs: lowered.root_uavs,
            written_root_uavs: lowered.written_root_uavs,
            writes_guest_memory: lowered.writes_guest_memory,
            root_indexed_views: res.root_indexed_views,
        };
        let limit = shader.user_data.iter()
            .find(|u| u.root == RootType::Other(crate::emit::USER_DISPATCH_THREADS_X))
            .unwrap().index as usize;
        if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
            v.validate(&mut shader.dxil).unwrap();
        }
        let _one = crate::emit::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(host) = Host::new(false) else { return };
        let (root, pso) = host.compute_pipeline(&shader).unwrap();
        let mut data = vec![u32::MAX; 64 + 64 * 4];
        for (i, x) in data[..64].iter_mut().enumerate() {
            *x = ((i % 4 + 1) as u32) << 24 | (1 << (i % 7 + 1)) - 1;
        }
        for size in [0, 1, 31, 32, 33, 47, 63, 64] {
            let active = if size == 0 { 64 } else { size as usize };
            let total = data[..active].iter().map(|x| x >> 24).sum::<u32>();
            let mut constants = vec![0; shader.user_data_dwords() as usize];
            constants[limit] = size;
            let values = host.dispatch(&root, &pso, &Dispatch {
                roots: vec![(0, Root::Constants(constants))],
                views: &[View::RawUavData(&data)],
                samplers: &[],
                groups: [1, 1, 1],
            }).unwrap();
            assert_eq!(values[0].len(), data.len());
            let (mut children, mut bits) = (0, 0);
            for (i, row) in values[0][64..].chunks_exact(4).enumerate() {
                let expected = if i < active {
                    let row = [active as u32, total, children, bits];
                    children += data[i] >> 24;
                    bits += (data[i] & 0x00FF_FFFF).count_ones();
                    row
                } else {
                    [u32::MAX; 4]
                };
                assert_eq!(row, expected, "limit {size}, lane {i}");
            }
        }
    }
}
