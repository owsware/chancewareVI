//! Diagnostic stand-in pad commands. Replace the local file with one line
//! `<sequence> <W|A|S|D|I|J|K|L> <hold_ms>`. Only increasing sequences run.

use std::io::Read;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use super::PadState;

struct Command {
    sequence: u64,
    key: u8,
    hold_ms: u64,
}

impl Command {
    fn parse(line: &str) -> Option<Self> {
        let mut words = line.split_ascii_whitespace();
        let sequence = words.next()?.parse().ok()?;
        let [key @ (b'W' | b'A' | b'S' | b'D' | b'I' | b'J' | b'K' | b'L')] =
            *words.next()?.as_bytes()
        else {
            return None;
        };
        let hold_ms = words.next()?.parse().ok()?;
        (words.next().is_none() && (1..=10_000).contains(&hold_ms)).then_some(Command {
            sequence,
            key,
            hold_ms,
        })
    }
}

pub(super) struct HeldMovement {
    command: Command,
    until: Instant,
    read: AtomicBool,
}

// Multiple readings from one hold share its acknowledgement.
impl PartialEq for HeldMovement {
    fn eq(&self, other: &Self) -> bool {
        self.command.sequence == other.command.sequence
    }
}

impl HeldMovement {
    pub(super) fn apply(&self, mut pad: PadState) -> Option<PadState> {
        let (axis, value) = match self.command.key {
            b'W' => (&mut pad.left_y, 1.0),
            b'S' => (&mut pad.left_y, -1.0),
            b'A' => (&mut pad.left_x, -1.0),
            b'D' => (&mut pad.left_x, 1.0),
            b'I' => (&mut pad.right_y, 1.0),
            b'K' => (&mut pad.right_y, -1.0),
            b'J' => (&mut pad.right_x, -1.0),
            b'L' => (&mut pad.right_x, 1.0),
            _ => unreachable!("parsed movement key"),
        };
        // A physical keyboard direction already driving this axis takes priority.
        if *axis != 0.0 {
            return None;
        }
        *axis = value;
        Some(pad)
    }

    pub(super) fn confirm_read(&self, state: &PadState) {
        if !self.read.load(Ordering::Relaxed) && !self.read.swap(true, Ordering::Relaxed) {
            tracing::info!(sequence = self.command.sequence, key = %char::from(self.command.key),
                left_x = state.left_x, left_y = state.left_y, right_x = state.right_x,
                right_y = state.right_y, "GetGamepadState returned movement");
        }
    }
}

#[derive(Default)]
pub(super) struct MovementFile {
    next_check: Option<Instant>,
    last_sequence: Option<u64>,
    active: Option<Arc<HeldMovement>>,
}

impl MovementFile {
    pub(super) fn poll(
        &mut self,
        path: &Path,
        now: Instant,
        available: bool,
    ) -> Option<Arc<HeldMovement>> {
        if self
            .active
            .as_ref()
            .is_some_and(|cue| now >= cue.until || !available)
        {
            self.end();
        }
        if self.next_check.is_none_or(|next| now >= next) {
            self.next_check = Some(now + Duration::from_millis(200));
            let mut line = String::new();
            if let Ok(file) = std::fs::File::open(path)
                && file.take(65).read_to_string(&mut line).is_ok()
                && line.len() <= 64
                && let Some(command) = Command::parse(&line)
                && self
                    .last_sequence
                    .is_none_or(|last| command.sequence > last)
            {
                self.last_sequence = Some(command.sequence);
                self.end();
                if available {
                    let until = now + Duration::from_millis(command.hold_ms);
                    tracing::info!(sequence = command.sequence, key = %char::from(command.key),
                        hold_ms = command.hold_ms, "stand-in movement started");
                    self.active = Some(Arc::new(HeldMovement {
                        command,
                        until,
                        read: AtomicBool::new(false),
                    }));
                }
            }
        }
        self.active.clone()
    }

    fn end(&mut self) {
        if let Some(cue) = self.active.take() {
            tracing::info!(sequence = cue.command.sequence, key = %char::from(cue.command.key),
                "stand-in movement ended");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn movements_accept_only_one_direction_and_a_bounded_hold() {
        for key in "WASDIJKL".chars() {
            let command = Command::parse(&format!("0 {key} 10000\r\n")).unwrap();
            let held = HeldMovement {
                command,
                until: Instant::now(),
                read: false.into(),
            };
            let expected = super::super::fallback(&[super::super::KeyState {
                virtual_key: key as u8,
                ..Default::default()
            }]);
            assert_eq!(held.apply(PadState::default()), Some(expected));
            assert!(held.apply(expected).is_none());
        }
        for input in [
            "1 W 0",
            "1 W -1",
            "1 W 10001",
            "1 Q 1",
            "1 WA 1",
            "1 W 1 extra",
            "x W 1",
        ] {
            assert!(Command::parse(input).is_none(), "{input}");
        }
    }

    #[test]
    fn duplicate_and_old_sequences_do_not_extend_or_restart_movement() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("command");
        let now = Instant::now();
        let mut movement = MovementFile::default();
        let mut poll =
            |ms, available| movement.poll(&path, now + Duration::from_millis(ms), available);
        std::fs::write(&path, "1 W 450").unwrap();
        let hold = poll(0, true).unwrap();
        std::fs::write(&path, "1 D 10000").unwrap();
        assert!(Arc::ptr_eq(&hold, &poll(400, true).unwrap()));
        assert!(poll(450, true).is_none());
        std::fs::write(&path, "0 D 10000").unwrap();
        assert!(poll(600, true).is_none());
        std::fs::write(&path, "2 J 1").unwrap();
        assert!(poll(800, true).is_some());
        assert!(poll(801, true).is_none());
        std::fs::write(&path, "3 I 10000").unwrap();
        assert!(poll(1000, true).is_some());
        assert!(poll(1001, false).is_none());
        std::fs::write(&path, "4 A 5000").unwrap();
        assert!(poll(1200, false).is_none());
        assert!(poll(1400, true).is_none());
    }

    #[test]
    fn get_gamepad_state_acknowledges_the_reading_it_returned() {
        use super::super::{READING, Reading, Value, gamepad_state};

        let cue = Arc::new(HeldMovement {
            command: Command::parse("17 L 3000").unwrap(),
            until: Instant::now() + Duration::from_secs(3),
            read: false.into(),
        });
        let moved = cue.apply(PadState::default()).unwrap();
        let mut reading = Reading {
            vtable: &READING,
            refs: 1.into(),
            device: std::ptr::null(),
            sequence: 3,
            timestamp: 1,
            value: Value::Pad(moved),
        };
        let mut pad = PadState::default();
        assert!(gamepad_state(&reading, &mut pad));
        assert!(!cue.read.load(Ordering::Relaxed));
        reading.value = Value::Movement(moved, Arc::clone(&cue));
        assert!(!gamepad_state(&reading, std::ptr::null_mut()));
        assert!(!cue.read.load(Ordering::Relaxed));
        assert!(gamepad_state(&reading, &mut pad));
        assert_eq!(pad.right_x, 1.0);
        assert!(cue.read.load(Ordering::Relaxed));
    }
}
