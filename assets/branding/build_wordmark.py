"""Build the outlined SVG wordmark (requires fontTools and Noto Sans Display).

Usage: python build_wordmark.py --font-dir /path/to/noto/fonts
The output has no runtime font, image, script, or network dependencies.
"""

import argparse
from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont


def word(text, font_path, x, color):
    font = TTFont(font_path)
    glyphs = font.getGlyphSet()
    cmap = font.getBestCmap()
    scale = 59 / font['head'].unitsPerEm
    paths = []
    for char in text:
        glyph = glyphs[cmap[ord(char)]]
        pen = SVGPathPen(glyphs)
        glyph.draw(pen)
        paths.append(
            f'    <path transform="translate({x:.3f} 78) scale({scale} {-scale})" '
            f'd="{pen.getCommands()}"/>'
        )
        x += glyph.width * scale - 1.25
    font.close()
    return f'  <g id="{text.rstrip("-").lower()}" fill="{color}">\n' + '\n'.join(paths) + '\n  </g>', x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font-dir', type=Path, default=Path('/usr/share/fonts/truetype/noto'))
    args = parser.parse_args()
    groups = []
    x = 117
    for text, weight, color in [
        ('Open-', 'SemiBold', '#347da8'),
        ('Human', 'Bold', '#087fae'),
        ('Scale', 'Bold', '#8054c9'),
    ]:
        group, x = word(text, args.font_dir / f'NotoSansDisplay-{weight}.ttf', x, color)
        groups.append(group)
    width = round(x + 9)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="108" viewBox="0 0 {width} 108" role="img" aria-labelledby="title desc">
  <title id="title">Open-HumanScale</title>
  <desc id="desc">A human and robot form an H linked by an ascending motion path, beside the Open-HumanScale wordmark.</desc>
  <!-- Hand-authored emblem; typography outlined from Noto Sans Display (SIL OFL 1.1). -->
  <g id="human-robot-emblem" fill="none" stroke-linecap="round" stroke-linejoin="round">
    <!-- Human: round head and an organic, open stance. -->
    <circle cx="26" cy="23" r="8" fill="#087fae"/>
    <path d="M26 42 C18 51 15 65 15 86 M26 43 L35 62 L35 86" stroke="#087fae" stroke-width="9"/>
    <!-- Shared rising action trajectory forms the crossbar of H. -->
    <path d="M26 43 L43 54 L60 43 L76 43" stroke="#087fae" stroke-width="8"/>
    <path d="M50 49.5 L60 43 L76 43" stroke="#8054c9" stroke-width="8"/>
    <!-- Robot: square head and articulated, geometric stance. -->
    <rect x="68" y="15" width="16" height="16" rx="4" fill="#8054c9"/>
    <path d="M76 43 L76 62 L65 74 L65 86 M76 62 L87 74 L87 86" stroke="#8054c9" stroke-width="9"/>
    <circle cx="76" cy="62" r="3" fill="#fff"/>
    <!-- Three scaling increments, echoing the rising shared trajectory. -->
    <path d="M45 28 V25 M51 28 V20 M57 28 V15" stroke="#61b5bf" stroke-width="3"/>
  </g>
{chr(10).join(groups)}
</svg>
'''
    output = Path(__file__).with_name('wordmark-v2.svg')
    output.write_text(svg)
    print(f'{output}: {width} × 108')


if __name__ == '__main__':
    main()
