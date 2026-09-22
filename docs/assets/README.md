# SLEEPJEV Visual Assets

This directory contains the visual identity and method diagrams used by the public
repository.

- `sleepjev-logo.svg` is the editable repository logo and wordmark.
- `icons/` contains the small line icons used by the README feature grid.
- `overall-framework-final.png` is the overview figure used on the README homepage.
- `overall-framework.svg` shows the end-to-end runtime decision path.
- `conventional-vs-sleepjev.svg` compares fixed-head and runtime-option serving.
- `query-serving.svg` shows one overnight encode reused across many queries.
- `plots/` contains R-generated release plots and the small CSV summary used to
  reproduce the README charts.
- `image-prompts.md` contains production prompts for GPT-IMAGE variants.

The SVG files are deterministic, text-readable fallbacks that can be reviewed in a
code review and rendered by GitHub. The PNG overview is the supplied figure selected
for the public homepage. These assets are explanatory figures, not model outputs and
do not encode a performance claim. If raster artwork is revised later, keep the SVG
diagrams as the source of truth for labels and method semantics.
