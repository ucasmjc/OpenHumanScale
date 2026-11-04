"""Build the type-led SVG wordmark with fontTools and Noto Sans Display.

Usage: python build_wordmark.py --font-dir /path/to/noto/fonts
The final SVG uses outlines, with no runtime fonts or embedded images.
"""

import argparse
from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont


def outline(text, font_path, x, baseline, size, tracking):
    font = TTFont(font_path)
    glyphs = font.getGlyphSet()
    cmap = font.getBestCmap()
    scale = size / font['head'].unitsPerEm
    paths = []
    for char in text:
        glyph = glyphs[cmap[ord(char)]]
        pen = SVGPathPen(glyphs, ntos=lambda v: f'{v:.3f}'.rstrip('0').rstrip('.'))
        glyph.draw(TransformPen(pen, (scale, 0, 0, -scale, x, baseline)))
        if pen.getCommands():
            paths.append(
                f'    <path d="{pen.getCommands()}"/>'
            )
        x += glyph.width * scale + tracking
    font.close()
    return '\n'.join(paths), x - tracking


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font-dir', type=Path, default=Path('/usr/share/fonts/truetype/noto'))
    args = parser.parse_args()

    def font(weight):
        return args.font_dir / f'NotoSansDisplay-{weight}.ttf'

    # A quieter Open prefix and a heavier, continuous HumanScale wordmark.
    prefix, x = outline('Open', font('SemiBold'), 10, 76, 78, -1.8)
    hyphen_x = x + 5
    main_x = hyphen_x + 27
    main_text, end = outline('HumanScale', font('ExtraBold'), main_x, 76, 78, -1.8)
    width = round(end + 10)

    # The subtitle carries the research identity without a separate pictogram.
    caption = 'Scaling Human & Robot Experience'
    _, caption_width = outline(caption, font('Medium'), 0, 113, 20, 0.65)
    caption_x = (width - caption_width) / 2
    caption_paths, _ = outline(caption, font('Medium'), caption_x, 113, 20, 0.65)
    left_dot = caption_x - 20
    right_dot = width - left_dot

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="130" viewBox="0 0 {width} 130" role="img" aria-labelledby="title desc">
  <title id="title">Open-HumanScale</title>
  <desc id="desc">Open-HumanScale. Scaling Human and Robot Experience.</desc>
  <!-- Original vector composition; Noto Sans Display outlines (SIL OFL 1.1). -->
  <defs>
    <linearGradient id="experience" gradientUnits="userSpaceOnUse" x1="{main_x:.3f}" y1="0" x2="{end:.3f}" y2="0">
      <stop offset="0" stop-color="#087eb6"/>
      <stop offset="0.48" stop-color="#356bc0"/>
      <stop offset="1" stop-color="#7951bb"/>
    </linearGradient>
  </defs>
  <g id="open" fill="#526b82">
{prefix}
  </g>
  <path id="hyphen" d="M{hyphen_x:.3f} 54 H{hyphen_x + 19:.3f}" stroke="#91a0af" stroke-width="7"/>
  <g id="humanscale" fill="url(#experience)">
{main_text}
  </g>
  <g id="subtitle" fill="#68798b">
{caption_paths}
  </g>
  <g id="subtitle-rules" fill="none" stroke="#bdc9d4" stroke-width="1.3">
    <path d="M20 106 H{left_dot:.3f}"/>
    <path d="M{right_dot:.3f} 106 H{width - 20}"/>
  </g>
  <circle cx="{left_dot:.3f}" cy="106" r="3" fill="#087eb6"/>
  <circle cx="{right_dot:.3f}" cy="106" r="3" fill="#7951bb"/>
</svg>
'''
    output = Path(__file__).with_name('wordmark-v3.svg')
    output.write_text(svg)
    print(f'{output}: {width} × 130')


if __name__ == '__main__':
    main()
