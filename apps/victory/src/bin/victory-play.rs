#![cfg_attr(windows, windows_subsystem = "windows")]

#[cfg(not(windows))]
fn main() {
    eprintln!("Chanceware requires Windows.");
}

#[cfg(windows)]
#[path = "../play_launcher.rs"]
mod windows_launcher;

#[cfg(windows)]
fn main() {
    if let Err(error) = windows_launcher::run(false) {
        windows_launcher::show_error(&format!("{error:#}"));
        std::process::exit(1);
    }
}
