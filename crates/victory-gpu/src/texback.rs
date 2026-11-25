//! Textures shaders write that the title's own code then reads on the CPU.
//! On the console the CPU and the GPU share memory, so what a compute pass
//! stores into a texture is in title memory for the CPU to read. Here it stays
//! in the texture's host copy. The title's ocean is one: its wave passes write
//! a 256x256 height map (R32_FLOAT) and a displacement map (R32G32_FLOAT), and
//! its CPU code samples both, which is what boats float on; it read memory
//! nothing had written (run rw1, `VICTORY_READ_WATCH=1`: about 30 places in
//! the title read those two kinds), and boats sank.
//!
//! For textures of the kinds in `VICTORY_TEXTURE_BACK` (comma-separated
//! `<width>x<height>f<format>`; `0` for none; by default the kinds the read
//! watch found the title reading), a list in which shaders write one copies it
//! into a read-back buffer of ours when it closes, and once the list has run
//! the texels are tiled into the texture's title memory in the console's
//! layout, as the console's GPU would have left them.
//!
//! shortcut: by kind, not by texture: textures of those kinds the title never
//! reads are copied back too (about a megabyte per such list). The copy shows
//! what the list's last write left; a title read in between sees the frame
//! before, as it would before the console's GPU has run.

use std::sync::OnceLock;

use windows::Win32::Graphics::Direct3D12::*;

use crate::object::This;

/// The kinds copied back: (width, height, format).
fn kinds() -> &'static [(u64, u32, u32)] {
    static KINDS: OnceLock<Vec<(u64, u32, u32)>> = OnceLock::new();
    KINDS.get_or_init(|| {
        const DEFAULT: &str = "256x256f41,256x256f16,64x64f16,128x128f36";
        let list = std::env::var("VICTORY_TEXTURE_BACK").unwrap_or_else(|_| DEFAULT.to_owned());
        let kinds: Vec<(u64, u32, u32)> = list
            .split(',')
            .filter_map(|k| {
                let (size, format) = k.trim().split_once('f')?;
                let (w, h) = size.split_once('x')?;
                Some((w.parse().ok()?, h.parse().ok()?, format.parse().ok()?))
            })
            .collect();
        if !kinds.is_empty() {
            tracing::info!(?kinds, "textures of these kinds that shaders write are copied back into title memory (VICTORY_TEXTURE_BACK)");
        }
        kinds
    })
}

/// Whether title texture `resource` is of a kind copied back.
pub(crate) fn wanted(resource: This) -> bool {
    let kinds = kinds();
    !kinds.is_empty()
        && crate::resource::texture_kind(resource).is_some_and(|k| kinds.contains(&k))
}

/// A texture's copy into a read-back buffer, recorded into a list, and where
/// its texels go once the list has run.
pub(crate) struct Back {
    // Own the guest memory through every execution of the closed command list.
    _resource: crate::resource::ResourceRef,
    mapping: crate::inplace::ReadbackMapping,
    buffer: ID3D12Resource,
    buffer_bytes: usize,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    mips: u32,
    layout: victory_formats::texture::Layout,
    /// The texture's title memory and its layout's bytes.
    at: usize,
    size: usize,
    /// Width, height, format (for the log).
    kind: Option<(u64, u32, u32)>,
}

fn subresource_count(desc: &D3D12_RESOURCE_DESC) -> u32 {
    let slices = if desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        1
    } else {
        u32::from(desc.DepthOrArraySize)
    };
    u32::from(desc.MipLevels) * slices
}

impl Back {
    pub(crate) fn destination(&self) -> (usize, usize) {
        (self.at, self.at + self.size)
    }
}

/// Expand a submission's intervals through currently mapped title aliases,
/// after releasing each list lock. CPU preparation may use any such view.
pub(crate) fn add_destination_aliases(ranges: &mut Vec<(usize, usize)>) {
    const PAGE: usize = 64 << 10;
    for index in 0..ranges.len() {
        let (at, end) = ranges[index];
        let mut page = at & !(PAGE - 1);
        while page < end {
            let lo = at.saturating_sub(page);
            let hi = (end - page).min(PAGE);
            for alias in crate::inplace::pages_of(page) {
                if alias != page {
                    ranges.push((alias + lo, alias + hi));
                }
            }
            page = page.saturating_add(PAGE);
        }
    }
}

/// Records into `list` the copy of `texture` (title resource `resource`'s
/// host copy, in the copy-source state) into a read-back buffer.
pub(crate) fn record(list: &ID3D12GraphicsCommandList, retained: crate::resource::ResourceRef, texture: &ID3D12Resource) -> Option<Back> {
    let host = crate::host::host()?;
    let resource = retained.as_ptr();
    let (layout, at, size) = crate::resource::title_texture(resource)?;
    at.checked_add(size)?;
    // SAFETY: a live texture.
    let desc = unsafe { texture.GetDesc() };
    let mips = u32::from(desc.MipLevels);
    let subresources = subresource_count(&desc);
    let n = subresources as usize;
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n];
    let mut rows = vec![0u32; n];
    let mut total = 0u64;
    // SAFETY: out arrays of `n` entries.
    unsafe {
        host.device.GetCopyableFootprints(
            &desc,
            0,
            subresources,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        )
    };
    let buffer_bytes = usize::try_from(total).ok()?;
    if buffer_bytes == 0 || buffer_bytes > isize::MAX as usize || at == 0 || size == 0 {
        return None;
    }
    let mapping = crate::inplace::ReadbackMapping::record(at, size)?;
    let buffer = host.read_back_buffer(total)?;
    for (i, footprint) in footprints.iter().enumerate() {
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(&buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *footprint },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i as u32 },
        };
        // SAFETY: an open list; the texture is in the copy-source state.
        unsafe { list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    COPIED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let kind = crate::resource::texture_kind(resource);
    Some(Back { _resource: retained, mapping, buffer, buffer_bytes, footprints, rows, mips, layout, at, size, kind })
}

/// Textures copied back since the start.
pub(crate) static COPIED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Once the list that recorded `back` has run: its texels, tiled into the
/// title's layout, into the texture's title memory.
pub(crate) fn land(back: &Back) {
    let Some(_mapping) = back.mapping.for_write() else { return };
    let mut tiled = vec![0u8; back.size];
    // Bytes of title memory the subresources do not cover stay as they are.
    crate::host::copy_mapped(tiled.as_mut_ptr(), back.at as *const u8, back.size, false);
    let mut data = std::ptr::null_mut();
    // SAFETY: our read-back buffer, written by the list that has run.
    unsafe {
        if back.buffer.Map(0, None, Some(&mut data)).is_err() || data.is_null() {
            return;
        }
    }
    // GetCopyableFootprints omits the final row's trailing padding.
    // SAFETY: Map succeeded and the buffer was created with exactly this width.
    let mapped = unsafe { std::slice::from_raw_parts(data.cast::<u8>(), back.buffer_bytes) };
    for (i, (footprint, &rows)) in back.footprints.iter().zip(&back.rows).enumerate() {
        let (mip, slice) = (i as u32 % back.mips, i as u32 / back.mips);
        let f = footprint.Footprint;
        let pitch = f.RowPitch as usize;
        // A slice ends where the next footprint begins, or at the actual buffer end.
        let end = back.footprints.get(i + 1).map_or(mapped.len(), |next| next.Offset as usize);
        let linear = &mapped[footprint.Offset as usize..end];
        if let Err(e) = back.layout.tile(&mut tiled, mip, slice, linear, pitch, pitch * rows as usize) {
            static SEEN: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());
            let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
            if !seen.contains(&back.at) {
                seen.push(back.at);
                tracing::warn!(
                    ?e,
                    mip,
                    slice,
                    kind = ?back.kind,
                    at = format_args!("{:#x}", back.at),
                    desc = ?back.layout.desc,
                    "a texture copied back could not be tiled (logged once per texture)"
                );
            }
            // SAFETY: mapped above.
            unsafe { back.buffer.Unmap(0, None) };
            return;
        }
    }
    // SAFETY: mapped above.
    unsafe { back.buffer.Unmap(0, None) };
    crate::host::copy_mapped(back.at as *mut u8, tiled.as_ptr(), back.size, true);
    #[cfg(feature = "diagnostics")]
    verify_first_landing(back, &tiled);
}

#[cfg(feature = "diagnostics")]
fn verify_first_landing(back: &Back, tiled: &[u8]) {
    static SEEN: std::sync::Mutex<Vec<(u64, u32, u32)>> = std::sync::Mutex::new(Vec::new());
    let Some(kind) = back.kind else { return };
    {
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        if seen.contains(&kind) {
            return;
        }
        seen.push(kind);
    }
    // Check the bytes in the title destination; copy_mapped also accepts sparse ranges.
    let mut guest = vec![0; tiled.len()];
    let complete = crate::host::read(back.at as *const u8, guest.as_mut_ptr(), guest.len());
    let verified = complete && guest == tiled;
    // Sample actual mip-zero texels, excluding alignment and mip padding.
    let desc = back.layout.desc;
    let (mut finite, mut nonzero, mut lo, mut hi) = (0usize, 0usize, f32::INFINITY, f32::NEG_INFINITY);
    if verified && matches!(kind.2, 16 | 41) {
        for gy in 0..5 {
            for gx in 0..5 {
                let (x, y) = ((desc.width - 1) * gx / 4, (desc.height - 1) * gy / 4);
                let Ok(offset) = back.layout.offset(0, x, y, 0) else { continue };
                let at = offset as usize;
                let Some(texel) = guest.get(at..at + desc.element_bytes as usize) else { continue };
                for bytes in texel.chunks_exact(4) {
                    let value = f32::from_le_bytes(bytes.try_into().unwrap());
                    if value.is_finite() {
                        finite += 1;
                        nonzero += usize::from(value != 0.0);
                        lo = lo.min(value);
                        hi = hi.max(value);
                    }
                }
            }
        }
    }
    let sample_range = (finite != 0).then_some((lo, hi));
    tracing::info!(?kind, at = format_args!("{:#x}", back.at), resource = ?back._resource.as_ptr(),
        bytes = tiled.len(), verified, finite, nonzero, ?sample_range, "first texture copy-back checked in title memory");
}
