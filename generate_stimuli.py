"""
Generate typographic stimulus images (text rendered as PNG).
All cultural sensitivity lives in the image; the text prompt to the VLM is neutral.
"""
import os
import platform
import textwrap
from PIL import Image, ImageDraw, ImageFont

from config import STIMULI, STIMULI_DIR, IMAGE_SIZE, IMAGE_BG, IMAGE_FG, FONT_SIZE


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    """Try system fonts, fall back to PIL default."""
    if platform.system() == "Darwin":
        candidates = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/SFNSText.ttf",
            "/Library/Fonts/Arial Unicode.ttf",
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        ]

    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue

    return ImageFont.load_default(size=size)


def make_typographic_image(
    text: str,
    output_path: str,
    size: tuple[int, int] = IMAGE_SIZE,
    bg: str = IMAGE_BG,
    fg: str = IMAGE_FG,
    font_size: int = FONT_SIZE,
) -> None:
    img = Image.new("RGB", size, color=bg)
    draw = ImageDraw.Draw(img)
    font = _get_font(font_size)

    margin_x, margin_y = 50, 40

    # Estimate characters per line from average glyph width
    try:
        avg_char_w = font.getlength("x")
    except AttributeError:
        avg_char_w = font_size * 0.6
    usable_width = size[0] - 2 * margin_x
    chars_per_line = max(20, int(usable_width / avg_char_w))

    wrapped = textwrap.fill(text, width=chars_per_line)

    # Measure line height
    bbox = font.getbbox("Ay")
    line_h = (bbox[3] - bbox[1]) + 8

    # Center vertically
    lines = wrapped.split("\n")
    total_h = line_h * len(lines)
    y = max(margin_y, (size[1] - total_h) // 2)

    for line in lines:
        draw.text((margin_x, y), line, fill=fg, font=font)
        y += line_h

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)


def generate_all() -> None:
    os.makedirs(STIMULI_DIR, exist_ok=True)
    for filename, category, axis, concept in STIMULI:
        path = os.path.join(STIMULI_DIR, filename)
        make_typographic_image(concept, path)
        print(f"  ✓ {filename:<25} [{axis}]")
    print(f"\nGenerated {len(STIMULI)} stimuli in {STIMULI_DIR}/")


if __name__ == "__main__":
    generate_all()
