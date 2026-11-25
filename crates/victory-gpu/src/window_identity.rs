use std::os::windows::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use windows::Win32::Foundation::{HWND, LPARAM, WPARAM};
use windows::Win32::Graphics::GdiPlus::{
    GdipCreateBitmapFromFile, GdipCreateHICONFromBitmap, GdipDisposeImage, GdiplusShutdown,
    GdiplusStartup, GdiplusStartupInput, Ok as GDIP_OK,
};
use windows::Win32::UI::Shell::ExtractIconExW;
use windows::Win32::UI::WindowsAndMessaging::{
    HICON, ICON_BIG, ICON_SMALL, PostMessageW, WM_SETICON,
};
use windows::core::PCWSTR;

struct WindowIdentity {
    caption: String,
    icon: Option<usize>,
}

static TITLE: OnceLock<WindowIdentity> = OnceLock::new();

/// Sets the presentation metadata before the title starts, once per process.
pub fn use_title_window(exe: PathBuf, display_name: Option<String>, logo: Option<PathBuf>) {
    TITLE.get_or_init(|| {
        // Artwork reads and decoding belong here, before the title starts.
        // Win32 does not own these handles; keep the icon until process exit.
        let icon = logo
            .as_deref()
            .and_then(png_icon)
            .or_else(|| exe_icon(&exe))
            .map(|h| h.0 as usize);
        WindowIdentity {
            caption: format_title(&exe, display_name.as_deref()),
            icon,
        }
    });
}

fn format_title(exe: &Path, display_name: Option<&str>) -> String {
    if let Some(name) = display_name {
        format!("{name} — Chanceware")
    } else if let Some(stem) = exe.file_stem() {
        let stem = stem.to_string_lossy();
        format!("{stem} — Chanceware")
    } else {
        "Chanceware".to_owned()
    }
}

pub(crate) fn title() -> &'static str {
    TITLE.get().map_or("Chanceware", |title| &title.caption)
}

pub(crate) fn apply_icon(hwnd: HWND) {
    if let Some(icon) = TITLE.get().and_then(|title| title.icon) {
        post_icon(hwnd, icon);
    }
}

fn post_icon(hwnd: HWND, icon: usize) {
    // Do not change the window class or call into the title's window procedure
    // synchronously. The first present can run while that thread is waiting
    // for the renderer and not processing messages.
    for size in [ICON_SMALL, ICON_BIG] {
        // SAFETY: the icon handle stays live through delivery and later use.
        let _ = unsafe {
            PostMessageW(
                Some(hwnd),
                WM_SETICON,
                WPARAM(size as usize),
                LPARAM(icon as isize),
            )
        };
    }
}

fn exe_icon(path: &Path) -> Option<HICON> {
    let path: Vec<u16> = path.as_os_str().encode_wide().chain([0]).collect();
    let mut icon = HICON::default();
    // SAFETY: requests one icon from a NUL-terminated path into one output slot.
    let count = unsafe { ExtractIconExW(PCWSTR(path.as_ptr()), 0, Some(&mut icon), None, 1) };
    (count == 1 && !icon.is_invalid()).then_some(icon)
}

fn png_icon(path: &Path) -> Option<HICON> {
    let path: Vec<u16> = path.as_os_str().encode_wide().chain([0]).collect();
    let input = GdiplusStartupInput {
        GdiplusVersion: 1,
        SuppressExternalCodecs: true.into(),
        ..Default::default()
    };
    let mut token = 0;
    let mut bitmap = std::ptr::null_mut();
    let mut icon = HICON::default();
    // SAFETY: path and output pointers remain valid; all GDI+ objects are
    // disposed before shutdown. The returned Win32 icon outlives the bitmap.
    unsafe {
        if GdiplusStartup(&mut token, &input, std::ptr::null_mut()) != GDIP_OK {
            return None;
        }
        let loaded = GdipCreateBitmapFromFile(PCWSTR(path.as_ptr()), &mut bitmap) == GDIP_OK;
        let made = loaded && GdipCreateHICONFromBitmap(bitmap, &mut icon) == GDIP_OK;
        if loaded {
            let _ = GdipDisposeImage(bitmap.cast());
        }
        GdiplusShutdown(token);
        (made && !icon.is_invalid()).then_some(icon)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caption_prefers_manifest_and_falls_back_to_the_executable() {
        let exe = Path::new(r"D:\Local Titles\Example Client.exe");
        assert_eq!(
            format_title(exe, Some("Shape & Sound")),
            "Shape & Sound — Chanceware"
        );
        assert_eq!(format_title(exe, None), "Example Client — Chanceware");
        assert_eq!(format_title(Path::new(""), None), "Chanceware");
    }

    #[test]
    fn posts_a_local_png_without_waiting_for_the_window_procedure() {
        use windows::Win32::UI::WindowsAndMessaging::{
            CreateWindowExW, DestroyIcon, DestroyWindow, DispatchMessageW, MSG, PM_REMOVE,
            PeekMessageW, SendMessageW, WINDOW_EX_STYLE, WINDOW_STYLE, WM_GETICON,
        };
        use windows::core::w;

        // Synthetic solid-color 32×32 PNG, not a title asset.
        const LOGO: &[u8] = &[
            0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48,
            0x44, 0x52, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x20, 0x08, 0x06, 0x00, 0x00,
            0x00, 0x73, 0x7a, 0x7a, 0xf4, 0x00, 0x00, 0x00, 0x2f, 0x49, 0x44, 0x41, 0x54, 0x78,
            0x9c, 0xed, 0xce, 0x21, 0x01, 0x00, 0x00, 0x08, 0x03, 0x30, 0x4a, 0x50, 0x1a, 0x49,
            0x61, 0x88, 0x71, 0x33, 0x31, 0xbf, 0xea, 0x9d, 0x4b, 0x2a, 0x01, 0x01, 0x01, 0x01,
            0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x81, 0x74, 0xe0, 0x01, 0xc6,
            0x36, 0xc0, 0x79, 0xf3, 0x45, 0x3e, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e,
            0x44, 0xae, 0x42, 0x60, 0x82,
        ];
        let dir = tempfile::tempdir().unwrap();
        let png = dir.path().join("local icon.png");
        std::fs::write(&png, LOGO).unwrap();
        let icon = png_icon(&png).expect("decode synthetic PNG");
        // SAFETY: a hidden top-level test window, created and destroyed on this thread.
        unsafe {
            let window = CreateWindowExW(
                WINDOW_EX_STYLE(0),
                w!("STATIC"),
                w!("Chanceware test"),
                WINDOW_STYLE(0),
                0,
                0,
                32,
                32,
                None,
                None,
                None,
                None,
            )
            .unwrap();
            let get = |size| {
                SendMessageW(window, WM_GETICON, Some(WPARAM(size as usize)), None).0
            };
            let old_small = get(ICON_SMALL);
            let old_large = get(ICON_BIG);
            post_icon(window, icon.0 as usize);
            // A caller must not dispatch icon messages before the owner pumps.
            let (queued_small, queued_large) = (get(ICON_SMALL), get(ICON_BIG));
            let mut msg = MSG::default();
            while PeekMessageW(&mut msg, Some(window), WM_SETICON, WM_SETICON, PM_REMOVE).as_bool() {
                DispatchMessageW(&msg);
            }
            let (small, large) = (get(ICON_SMALL), get(ICON_BIG));
            DestroyWindow(window).unwrap();
            DestroyIcon(icon).unwrap();
            assert_eq!((queued_small, queued_large), (old_small, old_large));
            assert_eq!((small, large), (icon.0 as isize, icon.0 as isize));
        }
    }
}
