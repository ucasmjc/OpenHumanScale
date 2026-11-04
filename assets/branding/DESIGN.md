# Open-HumanScale wordmark

## Current: vector wordmark (v2)

`wordmark-v2.svg` is a hand-authored vector design with no embedded raster image,
external font, script, or network dependency. The human and robot share an
ascending action path; three small bars represent scaling. The 623 × 108 canvas
is displayed at 560 px wide (about 97 px high) in the README.

The text uses outlined Noto Sans Display SemiBold/Bold (Google, SIL OFL 1.1).
See [Noto fonts](https://github.com/notofonts/noto-fonts) for the font sources.
The outlined wordmark is a document made using the font, not a font distribution.

The SVG can be edited directly. To rebuild the lettering and emblem from source:

```bash
python assets/branding/build_wordmark.py --font-dir /path/to/noto/fonts
```

This requires `fontTools`, `NotoSansDisplay-SemiBold.ttf`, and
`NotoSansDisplay-Bold.ttf` at build time only.

## Archived: generated concept (v1)

The PNG below is retained as design history; it is no longer used by the README.

`wordmark-v1.png` is the original transparent PNG generated with the built-in
image generation tool on November 4, 2026. Preserve its alpha channel when reusing it.
The lettering uses the project's blue and violet palette, with a curved/structured
capital H suggesting the transfer from human motion to robot action.

Presentation reference: [Sana](https://github.com/NVlabs/Sana), for its bold,
centered wordmark treatment. This asset was generated independently, without
using Sana's logo as an image input.

## Generation prompt

```text
Use case: logo-brand.
Create ONE finished original typographic logo for the open-source embodied AI research project Open-HumanScale, suitable for the centered top of a GitHub README. The user likes the simple, bold, custom-lettered presentation of the SANA research repository logo. Take only that principle of bold memorable typography, not its letterforms, green/red colors, or lightning symbol.

Text, exactly and once: "Open-HumanScale". Preserve this exact capitalization and the single hyphen after Open; do not insert a space or another hyphen between Human and Scale.
Design: a confident, clean, custom geometric sans-serif wordmark. Main character comes from the lettering itself. Human is cobalt blue / teal-blue, Scale transitions to a restrained violet, Open- in matching blue. Crisp heavy strokes, generous open counters, refined kerning, one uninterrupted horizontal line. Integrate one subtle distinctive idea into the capital H: a flowing organic left stroke and a structured articulated right stroke connected by its crossbar, suggesting human motion transferring into robot action; the H must remain immediately readable. A small ascending step-cut detail in the S can allude to scaling. These are typographic details, not separate drawings.
Style: flat vector-like artwork, exceptionally clean edges, precise professional scientific project identity; no photorealism, no 3D, no glossy chrome, no heavy shadows, no thin circuitry, no standalone robot heads or hands, no decorative badges, no extra icons, no extra words, no slogan, no watermark.
Composition: a single wide logo, centered horizontally and vertically, approximately 3:1 landscape canvas, lettering uses nearly the entire width with about 5% clear margin on each side. Keep the canvas tightly fitted to the wordmark with little wasted vertical padding.
Background: genuinely transparent PNG alpha, no painted checkerboard. Colors should remain legible on both white and dark GitHub themes.
Deliver the final logo only, not a presentation board or multiple variants.
```
