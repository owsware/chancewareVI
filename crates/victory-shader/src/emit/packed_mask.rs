//! A scalar bitwise instruction can pack a number drawn from a join of masks
//! and numbers. Preserve the numeric operation when its inputs have exact words.
use super::{Bin, Bool, Fields, LaneExit, Lower, MaskData, Operand, Reg, V, Val, Value};

impl Lower<'_> {
    pub(super) fn packed_mask_word(&mut self, value: Value) -> Option<V> {
        if self.mask_data != MaskData::Console {
            return None;
        }
        // A failed proof must not make an unrelated join live.
        let mut work = Vec::new();
        let result = self.scalar_bitwise_word(value, &mut 64, &mut work)?;
        // Once numeric phis are being filled, earlier phis may already have
        // substituted unused inputs. New dependencies can no longer be retained.
        if self.data_phis.is_empty() && work.iter().any(|v| !self.packed_phis.contains(v)) {
            return None;
        }
        while let Some(input) = work.pop() {
            if let Some(&(b, i)) = self.phi_at.get(&input)
                && self.packed_phis.insert(input)
            {
                self.read_phis.insert(input);
                let block = &self.x.p.blocks[b];
                for (&arg, &from) in block.phis[i].args.iter().zip(&block.preds) {
                    if !self.impossible_forward_edge(from, b) {
                        work.push(arg);
                    }
                }
            }
        }
        self.vals.insert(value, Val::I32(result));
        Some(result)
    }

    fn scalar_bitwise_word(
        &mut self,
        value: Value,
        budget: &mut u32,
        phis: &mut Vec<Value>,
    ) -> Option<V> {
        *budget = budget.checked_sub(1)?;
        if let Some(&Val::I32(word)) = self.vals.get(&value) {
            if self.phi_at.contains_key(&value) {
                phis.push(value);
            }
            return Some(word);
        }
        let &block = self.def_block.get(&value)?;
        let inst = self.x.p.blocks[block]
            .insts
            .iter()
            .find(|i| i.writes.iter().any(|w| w.1 == value))?;
        let Fields::Sop {
            dst: Some(Operand::Scalar { code: dest, .. }),
            src,
            ..
        } = &inst.fields
        else {
            return None;
        };
        if inst.write(Reg::S(*dest)) != Some(value) {
            return None;
        }
        let op = match inst.name() {
            "s_and_b32" => Bin::And,
            "s_or_b32" => Bin::Or,
            "s_xor_b32" => Bin::Xor,
            _ => return None,
        };
        let [a, b] = src.as_slice() else { return None };
        let mut arg = |this: &mut Self, op: &Operand| match *op {
            Operand::Literal(k) => Some(this.s.m.i32(k)),
            Operand::Const(k) => crate::resource::inline(k).map(|v| this.s.m.i32(v)),
            Operand::Scalar { code, .. } => {
                this.scalar_bitwise_word(inst.read(Reg::S(code))?, budget, phis)
            }
            _ => None,
        };
        let at = if block == self.cur_ir {
            self.cur
        } else {
            self.tail[block]?
        };
        let x = arg(self, a)?;
        let y = arg(self, b)?;
        Some(self.s.m.bin(at, op, x, y))
    }

    /// A scalar word at a join need not come from an edge no lane can take.
    /// Only ordinary forward branches qualify: a loop limit can force an exit.
    pub(super) fn impossible_forward_edge(&self, from: usize, to: usize) -> bool {
        let LaneExit::Branch {
            when,
            taken,
            not_taken,
        } = self.x.st.exits[from]
        else {
            return false;
        };
        if taken == not_taken
            || self.x.st.wide.contains(&from)
            || self.rpo[taken] <= self.rpo[from]
            || self.rpo[not_taken] <= self.rpo[from]
        {
            return false;
        }
        (when == Bool::FALSE && to == taken) || (when == Bool::TRUE && to == not_taken)
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::super::tests::{S_ENDPGM, lower_compute, root};
    use crate::emit::host::{Dispatch, Host, Root, View};
    use crate::emit::{HostShader, validator::Validator};
    use crate::native::{HwStage, ResourceType, RootType, UsedResource};

    #[test]
    fn a_packed_scalar_id_from_a_reused_mask_register_retains_all_bits() {
        let words = [
            0xBE84_0300,
            0xBE85_0380, // s[4:5] = output table
            0xF408_0402,
            0xFA00_0000, // s[16:19] = output UAV
            0xBE8A_037E, // s10 = initial exec
            0xBE8C_3C0A, // s_and_saveexec_b32 s12, s10
            0xBF88_0001, // s_cbranch_execz merge (never taken)
            0xBE8A_0301, // s10 = packed scalar input
            0xBEFE_030C, // merge: restore exec
            0xD4C2_0006,
            0x0002_0080, // v_cmp_eq_u32 s6, 0, v0
            0xBE8E_3C06, // s_and_saveexec_b32 s14, s6
            0xBF88_0007, // only lane zero uses the packed word
            0x870B_FF0A,
            0x001F_FFFF, // s11 = lower 21 bits
            0x880D_FF0B,
            0xDAE0_0000, // s13 = upper field | s11
            0x7E02_020D, // v1 = s13
            0xE070_1000,
            0x8004_0100, // buffer_store_dword v1, v0, s[16:19]
            0xBEFE_030E, // restore exec
            S_ENDPGM,
        ];
        let layout = vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
        let output = UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        };
        let lowered = lower_compute(&words, layout, vec![output], false, false).unwrap();
        let mut shader = HostShader {
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
        if let Ok(mut validator) = Validator::load(Validator::default_path().as_deref()) {
            validator.validate(&mut shader.dxil).unwrap();
        }
        let _one = crate::emit::valu_tests::ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let Ok(host) = Host::new(false) else { return };
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_FEATURE_D3D12_OPTIONS1, D3D12_FEATURE_DATA_D3D12_OPTIONS1,
        };
        let mut caps = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
        // SAFETY: feature query with the corresponding initialized structure.
        unsafe {
            host.device
                .CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS1,
                    (&mut caps as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS1).cast(),
                    size_of_val(&caps) as u32,
                )
                .unwrap()
        };
        if !(caps.WaveLaneCountMin..=caps.WaveLaneCountMax).contains(&32) {
            eprintln!("packed ID execution needs a wave32 adapter");
            return;
        }
        let (root, pso) = host.compute_pipeline(&shader).unwrap();
        for value in [0, 0x0012_3456, 0xF123_FFFF] {
            let mut data = vec![0; shader.user_data_dwords().max(1) as usize];
            for param in &shader.user_data {
                if param.root == RootType::Constants {
                    data[param.index as usize] = value;
                }
            }
            let result = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(data))],
                        views: &[View::RawUav(1)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap();
            assert_eq!(result[0], [0xDAE0_0000 | (value & 0x001F_FFFF)]);
        }
    }
}
