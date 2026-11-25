//! Wired DualSense HID input through the Windows HID class driver.

use std::time::{Duration, Instant};

use hidapi::{BusType, HidApi, HidDevice, HidResult};

use super::{
    A, B, DOWN, LEFT, LEFT_SHOULDER, LEFT_STICK, MENU, PadState, RIGHT, RIGHT_SHOULDER,
    RIGHT_STICK, UP, VIEW, X, Y,
};

const SONY: u16 = 0x054c;
const DUALSENSE: u16 = 0x0ce6;
const REPORT_LEN: usize = 64;
const RECHECK: Duration = Duration::from_secs(1);

#[derive(Default)]
pub(super) struct DualSense {
    device: Option<HidDevice>,
    state: Option<PadState>,
    retry_at: Option<Instant>,
}

impl DualSense {
    pub(super) fn poll(&mut self) -> Option<PadState> {
        if self.device.is_none() {
            let now = Instant::now();
            if self.retry_at.is_some_and(|retry| now < retry) {
                return None;
            }
            self.retry_at = Some(now + RECHECK);
            match open() {
                Ok(device) => self.device = device,
                Err(error) => tracing::debug!(%error, "could not open DualSense"),
            }
        }

        let device = self.device.as_ref()?;
        let mut report = [0; REPORT_LEN];
        // Drain queued reports without waiting. Keep the bound even when a
        // device submits reports faster than the host can consume them.
        for _ in 0..32 {
            match device.read_timeout(&mut report, 0) {
                Ok(0) => break,
                Ok(len) => {
                    if let Some(state) = report.get(..len).and_then(decode) {
                        if self.state.is_none() {
                            tracing::info!("wired DualSense input connected");
                        }
                        if self
                            .state
                            .is_none_or(|previous| previous.buttons != state.buttons)
                        {
                            tracing::debug!(buttons = state.buttons, "DualSense buttons");
                        }
                        self.state = Some(state);
                    }
                }
                Err(error) => {
                    tracing::debug!(%error, "DualSense input disconnected");
                    self.device = None;
                    self.state = None;
                    self.retry_at = Some(Instant::now() + RECHECK);
                    break;
                }
            }
        }
        self.state
    }
}

fn open() -> HidResult<Option<HidDevice>> {
    let api = HidApi::new()?;
    api.device_list()
        .find(|device| {
            device.vendor_id() == SONY
                && device.product_id() == DUALSENSE
                && matches!(device.bus_type(), BusType::Usb)
                && device.usage_page() == 0x01
                && device.usage() == 0x05
        })
        .map(|device| device.open_device(&api))
        .transpose()
}

fn axis(value: u8) -> f32 {
    let offset = f32::from(value) - 128.0;
    (offset / 127.0).max(-1.0)
}

/// The USB report begins with its ID, four stick axes, two trigger axes,
/// a sequence byte, and the hat/face and shoulder/system button fields.
fn decode(report: &[u8]) -> Option<PadState> {
    let &[0x01, lx, ly, rx, ry, lt, rt, _, face, shoulders, ..] = report else {
        return None;
    };
    if report.len() != REPORT_LEN {
        return None;
    }
    let hat = [
        UP,
        UP | RIGHT,
        RIGHT,
        DOWN | RIGHT,
        DOWN,
        DOWN | LEFT,
        LEFT,
        UP | LEFT,
    ];
    let buttons = hat.get(usize::from(face & 0xf)).copied().unwrap_or(0);
    let bits = u16::from(face) | u16::from(shoulders) << 8;
    let mapping = [
        (4, X),
        (5, A),
        (6, B),
        (7, Y),
        (8, LEFT_SHOULDER),
        (9, RIGHT_SHOULDER),
        (12, VIEW),
        (13, MENU),
        (14, LEFT_STICK),
        (15, RIGHT_STICK),
    ];
    Some(PadState {
        buttons: mapping.iter().fold(buttons, |all, (host, title)| {
            if bits & (1 << host) != 0 {
                all | title
            } else {
                all
            }
        }),
        left_trigger: f32::from(lt) / 255.0,
        right_trigger: f32::from(rt) / 255.0,
        left_x: axis(lx),
        left_y: -axis(ly),
        right_x: axis(rx),
        right_y: -axis(ry),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_usb_buttons_hat_and_axes() {
        let mut report = [0; REPORT_LEN];
        report[..10].copy_from_slice(&[1, 255, 0, 128, 255, 0, 255, 0, 0x21, 0x51]);
        assert_eq!(
            decode(&report),
            Some(PadState {
                buttons: A | UP | RIGHT | LEFT_SHOULDER | VIEW | LEFT_STICK,
                left_x: 1.0,
                left_y: 1.0,
                right_y: -1.0,
                right_trigger: 1.0,
                ..PadState::default()
            })
        );
        report[..10].copy_from_slice(&[1, 128, 128, 128, 128, 0, 0, 0, 8, 0]);
        assert_eq!(decode(&report), Some(PadState::default()));
        assert_eq!(decode(&report[..10]), None);
        report[0] = 0x31;
        assert_eq!(decode(&report), None);
    }
}
