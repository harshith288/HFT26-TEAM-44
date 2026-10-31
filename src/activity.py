"""Feature 4: day/night signals from the pixels, and hour of day from timestamps.

Night is decided from the image, not the clock, for two measured reasons:
- CCT stores 12 AM as hour 12 (141 of 170 hour-12 train/val images are IR).
- Camera 120 uses a white flash and its clock is unreliable: it has pitch-black
  frames at 8 AM to 4 PM and bright frames at 1 AM and 11 PM.
"""
import numpy as np


def channel_difference(image):
    """Mean |R-G| and |G-B| on a small copy, skipping the camera's black info bars."""
    small = np.asarray(image.convert("RGB").resize((64, 64)), dtype=np.float32)[4:-4]
    return float((np.abs(small[..., 0] - small[..., 1]).mean() +
                  np.abs(small[..., 1] - small[..., 2]).mean()) / 2)


def dark_share(image, dark_level):
    """Share of near-black pixels in the top half, where a white flash does not reach."""
    small = np.asarray(image.convert("L").resize((64, 64)), dtype=np.float32)[4:32]
    return float((small < dark_level).mean())


def is_ir_image(image, tolerance):
    """IR night images are grayscale, so R, G and B are almost equal."""
    return channel_difference(image) < tolerance


def is_dark_scene(image, activity_cfg):
    return dark_share(image, activity_cfg["dark_pixel_level"]) >= activity_cfg["dark_share_min"]


def is_night(is_ir, is_dark):
    """Night means an IR image or a dark scene. Needs no timestamp."""
    return bool(is_ir or is_dark)


def period(is_ir, is_dark):
    """Group used for every day/night metric."""
    if is_ir:
        return "ir_night"
    return "color_night" if is_dark else "day"


def corrected_hour(captured_at, night):
    """Hour of day, or None with no timestamp. Night at hour 12 is really 12 AM."""
    if not captured_at:
        return None
    hour = int(captured_at[11:13])
    return 0 if hour == 12 and night else hour
