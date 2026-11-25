use super::barycentric_tests::{element, pixel_with_inputs};
use super::bitcode::{Attr, Term};
use super::host::{Draw, Host};
use super::shader::Shader;
use windows::Win32::Graphics::Direct3D12::D3D12_SHADER_VISIBILITY_PIXEL;

/// Pass clip-space coordinates through to SV_Position.
fn vertex() -> Vec<u8> {
    let mut vs = Shader::new(false);
    let mut input = element("POSITION", 0, 0);
    input.usage = 15;
    vs.inputs.push(input);
    let mut output = element("SV_Position", 3, 0);
    output.usage = 15;
    output.interp = 4;
    vs.outputs.push(output);
    let zero = vs.m.i32(0);
    let undef = vs.m.undef(vs.t.i32);
    for col in 0..4 {
        let col = vs.m.i8(col);
        let value = vs.op(
            vs.entry,
            "loadInput.f32",
            Attr::ReadNone,
            vs.t.f32,
            4,
            &[zero, zero, col, undef],
        );
        vs.op(
            vs.entry,
            "storeOutput.f32",
            Attr::NoUnwind,
            vs.t.void,
            5,
            &[zero, zero, col, value],
        );
    }
    vs.m.terminate(vs.entry, Term::Ret);
    vs.finish()
}

fn quad_broadcast_shader(dpp: bool) -> super::HostShader {
    // In whole-quad mode, broadcast the x coordinate of lanes 0 and 1
    // and the y coordinate of lanes 2 and 3. Helpers have interpolated
    // coordinates even where their pixel belongs to the other triangle.
    let mut words = vec![0xBEFE_097E]; // s_wqm_b32 exec_lo, exec_lo
    for (lane, src) in [(0, 0), (1, 0), (2, 1), (3, 1)] {
        let selector = lane * 0x55;
        let dst = lane + 2;
        if dpp {
            // v_mov_b32 v[dst], v[src] quad_perm:[lane,lane,lane,lane]
            words.extend([
                0x7E00_02FA | (dst << 17),
                0xFF00_0000 | (selector << 8) | src,
            ]);
        } else {
            // ds_swizzle_b32 v[dst], v[src] offset:QUAD_PERM
            words.extend([
                0xD800_0000 | (0x35 << 18) | 0x8000 | selector,
                (dst << 24) | src,
            ]);
        }
    }
    words.extend([0xF800_180F, 0x0504_0302, 0xBF81_0000]); // export v2..v5
    pixel_with_inputs(&words, (1 << 8) | (1 << 9), &[element("SV_Position", 3, 0)]).unwrap()
}

#[test]
fn pixel_quad_broadcast_reads_helper_coordinates_at_shared_triangle_edge() {
    let vs = vertex();

    const SIZE: usize = 16;
    let a = [-1.0, -1.0, 0.5, 1.0];
    let b = [-1.0, 1.0, 0.5, 1.0];
    let c = [1.0, -1.0, 0.5, 1.0];
    let d = [1.0, 1.0, 0.5, 1.0];
    let mut failures = Vec::new();
    for (name, warp) in [("WARP", true), ("hardware", false)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {name} device; skipped");
            continue;
        };
        for (instruction, dpp) in [("ds_swizzle", false), ("DPP", true)] {
            let ps = quad_broadcast_shader(dpp);
            let root = host
                .root_signature_for(&[(&ps, D3D12_SHADER_VISIBILITY_PIXEL)])
                .unwrap();
            let pso = host.pipeline_with(&root, &vs, &ps.dxil).unwrap();
            // The harness draws one triangle per call into a cleared target. Add
            // the targets so every pixel must be owned by exactly one triangle.
            let halves = [[a, b, c], [b, c, d]].map(|triangle| {
                host.draw(
                    &root,
                    &pso,
                    &Draw {
                        roots: Vec::new(),
                        textures: &[],
                        samplers: &[],
                        vertices: triangle.as_flattened(),
                        stride: 16,
                        size: (SIZE as u32, SIZE as u32),
                        mesh_groups: 0,
                    },
                )
                .unwrap()
            });
            let mut mismatches = Vec::new();
            for y in 0..SIZE {
                for x in 0..SIZE {
                    let i = y * SIZE + x;
                    let pixel: [f32; 4] =
                        std::array::from_fn(|c| halves[0][i][c] + halves[1][i][c]);
                    let left = (x & !1) as f32 + 0.5;
                    let bottom = (y & !1) as f32 + 1.5;
                    let expected = [left, left + 1.0, bottom, bottom];
                    if pixel != expected {
                        mismatches.push((x, y, pixel, expected));
                    }
                }
            }
            if let Some(first) = mismatches.first() {
                let edge = mismatches
                    .iter()
                    .filter(|(x, y, _, _)| x / 2 == y / 2)
                    .count();
                failures.push(format!("{name} {instruction}: {} incorrect pixels, {edge} in quads spanning the diagonal; first (x, y, RGBA, expected) {first:?}", mismatches.len()));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
