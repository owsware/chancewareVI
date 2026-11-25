//! Process-wide state the implementations need.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

static MAIN_IMAGE_BASE: AtomicUsize = AtomicUsize::new(0);

/// Records where the title's main image is mapped. The title expects
/// `GetModuleHandle(NULL)` and friends to return this, not our own exe.
pub fn set_main_image(base: usize) {
    MAIN_IMAGE_BASE.store(base, Ordering::Release);
}

pub fn main_image() -> Option<usize> {
    match MAIN_IMAGE_BASE.load(Ordering::Acquire) {
        0 => None,
        base => Some(base),
    }
}

/// Original file of each title image we mapped, by base address.
static IMAGE_PATHS: Mutex<Vec<(usize, PathBuf)>> = Mutex::new(Vec::new());

/// Records the file a title image was mapped from: the title asks for its
/// own path (`GetModuleFileName`) to find its data, and must not get our
/// cache copy.
pub fn set_image_path(base: usize, path: &Path) {
    let path = std::path::absolute(path).unwrap_or_else(|_| path.to_owned());
    let mut paths = IMAGE_PATHS.lock().unwrap_or_else(|e| e.into_inner());
    paths.retain(|(b, _)| *b != base);
    paths.push((base, path));
}

/// Forgets a title image that was unmapped again.
pub fn forget_image(base: usize) {
    let mut paths = IMAGE_PATHS.lock().unwrap_or_else(|e| e.into_inner());
    paths.retain(|(b, _)| *b != base);
}

pub fn image_path(base: usize) -> Option<PathBuf> {
    let paths = IMAGE_PATHS.lock().unwrap_or_else(|e| e.into_inner());
    paths
        .iter()
        .find(|(b, _)| *b == base)
        .map(|(_, p)| p.clone())
}

/// The title image whose original file is called `file` (any case).
pub fn image_named(file: &str) -> Option<usize> {
    let paths = IMAGE_PATHS.lock().unwrap_or_else(|e| e.into_inner());
    paths
        .iter()
        .find(|(_, p)| p.file_name().is_some_and(|n| n.eq_ignore_ascii_case(file)))
        .map(|(b, _)| *b)
}

/// Maps a title DLL the title loads itself (`LoadLibrary`) and returns its
/// base; `Ok(None)` if `file` is not one of the title's DLLs.
pub type TitleDllLoader = Box<dyn Fn(&str) -> Result<Option<usize>, String> + Send + Sync>;

static TITLE_DLL_LOADER: std::sync::OnceLock<TitleDllLoader> = std::sync::OnceLock::new();

/// Set once, by whoever mapped the title (it knows the folders and the
/// resolver); later calls are ignored.
pub fn set_title_dll_loader(loader: TitleDllLoader) {
    let _ = TITLE_DLL_LOADER.set(loader);
}

pub fn load_title_dll(file: &str) -> Result<Option<usize>, String> {
    TITLE_DLL_LOADER.get().map_or(Ok(None), |load| load(file))
}

/// What the title's manifest (`MicrosoftGame.config` beside its executable)
/// says about it, and where its persistent local storage lives.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Title {
    pub id: u32,
    /// The store's product id (empty if the manifest has none).
    pub store_id: String,
    /// Persistent local storage: our folder for it and the title's quota.
    /// Game saves live beside it ([`Title::saves`]).
    pub storage: PathBuf,
    pub storage_bytes: u64,
}

#[derive(Default)]
pub struct ShellVisuals {
    pub display_name: Option<String>,
    logo: Option<String>,
}

impl ShellVisuals {
    /// An optional local PNG. Do not follow manifest paths outside the title folder.
    pub fn logo_in(&self, title_dir: &Path) -> Option<PathBuf> {
        let root = title_dir.canonicalize().ok()?;
        let local_png = |file: &str| {
            let path = Path::new(file);
            if !path
                .components()
                .all(|c| matches!(c, std::path::Component::Normal(_)))
                || !path.extension().is_some_and(|e| e.eq_ignore_ascii_case("png"))
            {
                return None;
            }
            let path = root.join(path).canonicalize().ok()?;
            (path.starts_with(&root) && path.is_file()).then_some(path)
        };
        self.logo
            .as_deref()
            .and_then(local_png)
            .or_else(|| local_png("xbs_logo_150.png"))
    }
}

impl Title {
    /// Reads console metadata and optional window presentation from the manifest.
    pub fn from_manifest(text: &str) -> (Self, ShellVisuals) {
        let Ok(doc) = roxmltree::Document::parse(text) else {
            return (Self::default(), ShellVisuals::default());
        };
        let game = doc.root_element();
        if !game.has_tag_name("Game") {
            return (Self::default(), ShellVisuals::default());
        }
        fn child<'a, 'input>(
            parent: roxmltree::Node<'a, 'input>,
            name: &str,
        ) -> Option<roxmltree::Node<'a, 'input>> {
            parent.children().find(|n| n.has_tag_name(name))
        }
        let get = |name| child(game, name).and_then(|n| n.text()).map(str::trim);
        let id = get("TitleId").and_then(|v| u32::from_str_radix(v, 16).ok());
        let storage_mb: Option<u64> = child(game, "PersistentLocalStorage")
            .and_then(|n| child(n, "SizeMB"))
            .and_then(|n| n.text()?.trim().parse().ok());
        let shell = child(game, "ShellVisuals");
        let display_name = shell
            .and_then(|n| n.attribute("DefaultDisplayName"))
            .map(str::trim)
            .filter(|name| {
                !name.is_empty()
                    && !name.starts_with("ms-resource:")
                    && !name.chars().any(char::is_control)
            })
            .map(str::to_owned);
        let logo = shell
            .and_then(|n| n.attribute("Square150x150Logo"))
            .map(str::to_owned);
        let title = Self {
            id: id.unwrap_or(0),
            store_id: get("StoreId").unwrap_or_default().to_owned(),
            storage: PathBuf::new(),
            storage_bytes: storage_mb.unwrap_or(0) << 20,
        };
        (title, ShellVisuals { display_name, logo })
    }

    /// Where game saves live: next to persistent local storage, under the
    /// data directory, never the cache.
    pub fn saves(&self) -> PathBuf {
        self.storage.with_file_name("saves")
    }
}

static TITLE: Mutex<Option<Title>> = Mutex::new(None);

pub fn set_title(title: Title) {
    *TITLE.lock().unwrap_or_else(|e| e.into_inner()) = Some(title);
}

/// The running title; the default (id 0, no storage) until set.
pub fn title() -> Title {
    TITLE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
        .unwrap_or_default()
}

/// Gives the process the console's millisecond timers. Titles pace
/// themselves with short waits (the title's audio mixer sleeps
/// about 4 ms per 5.3 ms block) and cannot tune the timer on the console;
/// a desktop process waits in 15.6 ms ticks unless it asks for finer ones,
/// and Windows 11 drops that request while the window is hidden unless the
/// process opts out. Process-wide, for the whole run.
#[cfg(windows)]
pub fn use_console_timers() {
    use windows_sys::Win32::Media::timeBeginPeriod;
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcess, PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION, PROCESS_POWER_THROTTLING_STATE,
        ProcessPowerThrottling, SetProcessInformation,
    };
    let keep = PROCESS_POWER_THROTTLING_STATE {
        Version: PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        ControlMask: PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION,
        StateMask: 0,
    };
    // SAFETY: plain process settings; `keep` outlives the call.
    unsafe {
        SetProcessInformation(
            GetCurrentProcess(),
            ProcessPowerThrottling,
            (&raw const keep).cast(),
            size_of_val(&keep) as u32,
        );
        timeBeginPeriod(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_title_id_store_id_and_storage_quota() {
        let text = "\u{feff}<?xml version=\"1.0\"?>\n<Game configVersion=\"1\">\
                    <!-- <TitleId>00000000</TitleId> -->\
                    <Identity Name=\"x\"/>\n  <TitleId>639AA8DD</TitleId>\n\
                    <![CDATA[ <TitleId>1</TitleId> ]]>                    <StoreId>9&amp;C</StoreId><Other><SizeMB>7</SizeMB></Other>\
                    <PersistentLocalStorage>\n  <SizeMB>1024</SizeMB> <!-- c -->\n\
                    </PersistentLocalStorage></Game>";
        let (t, visuals) = Title::from_manifest(text);
        assert_eq!((t.id, t.storage_bytes), (0x639A_A8DD, 1024 << 20));
        assert_eq!(t.store_id, "9&C");
        assert!(visuals.display_name.is_none());
        assert_eq!(Title::from_manifest("").0.storage_bytes, 0);
    }

    #[test]
    fn reads_literal_display_name_and_uses_only_local_png_paths() {
        let (_, visuals) = Title::from_manifest(
            "<Game><ShellVisuals DefaultDisplayName='  Shape &amp; Sound &#x2605;  ' \
             Square150x150Logo='art/window.png'/></Game>",
        );
        assert_eq!(visuals.display_name.as_deref(), Some("Shape & Sound ★"));
        let tmp = tempfile::tempdir().unwrap();
        let title = tmp.path().join("title");
        std::fs::create_dir_all(title.join("art")).unwrap();
        let logo = title.join("art/window.png");
        std::fs::write(&logo, b"").unwrap();
        assert_eq!(visuals.logo_in(&title), Some(logo.canonicalize().unwrap()));

        let (_, invalid) = Title::from_manifest(
            "<Game><ShellVisuals DefaultDisplayName='ms-resource:Title' \
             Square150x150Logo='../outside.png'/></Game>",
        );
        std::fs::write(tmp.path().join("outside.png"), b"").unwrap();
        assert!(invalid.display_name.is_none());
        assert_eq!(invalid.logo_in(&title), None);
        let fallback = title.join("xbs_logo_150.png");
        std::fs::write(&fallback, b"").unwrap();
        assert_eq!(invalid.logo_in(&title), Some(fallback.canonicalize().unwrap()));
    }
}
