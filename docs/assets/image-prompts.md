# GPT-IMAGE Prompts

The repository includes editable SVG fallbacks so the diagrams remain reviewable in a text-only checkout. These prompts can be used to generate polished bitmap assets in GPT-IMAGE and then exported with the same filenames after human review.

The current README uses `overall-framework-final.png`, the project-provided overview
figure. The prompts below remain the generation specification for future visual
variants; generated artwork should preserve the same method semantics and terminology.

## Overall Framework

> Academic machine-learning system diagram for a project named SLEEPJEV, white background, restrained teal and cobalt with one coral accent, clean vector style, no gradients, no 3D, no decorative blobs. Show a left-to-right pipeline: "8-hour PSG" waveform input, "epoch / hour / coarse states", "shared overnight state", "predicted stage and event postings", a runtime clinical query card with time, stage, and event filters, "sparse retrieval", "option-conditioned scorer", and final probability bars over runtime options. Use very little text, generous whitespace, aligned arrows, consistent 2 px lines, publication-ready typography, readable at 1600 by 900 pixels. Do not add medical claims, patient imagery, logos, or invented metrics.

## Conventional versus SLEEPJEV

> Clean two-lane scientific comparison diagram on a white background. Upper lane: "conventional fixed-head" with PSG, repeated task-specific encoding, fixed stage/event heads, separate outputs. Lower lane: "SLEEPJEV" with PSG, one shared overnight encoding, reusable cache, many runtime queries, sparse retrieval, shared option scorer. Teal for shared computation, gray for fixed components, coral for repeated work. Minimal labels, no gradients, no cartoon icons, no hospital imagery, conference-paper visual language.

## High-query serving

> Scientific systems diagram showing one overnight PSG encoded exactly once into a reusable cache, then fanning out to Q=1, Q=8, Q=32, Q=128, and Q=512 runtime queries. Each query passes through a small sparse indexed retrieval block and a shared option-conditioned scorer. Include a simple qualitative latency curve that flattens for the cached path and a separate repeated-encoding path, without numeric claims. White background, navy text, teal lines, coral highlights, precise grid alignment, no gradients, no 3D, no decorative circles.

## Logo

> Minimal academic software logo for "SLEEPJEV" on transparent background. Combine a compact overnight waveform or EEG trace with a structured decision node and a small query bracket, suggesting sleep, PSG, runtime semantics, and reusable intelligence. Geometric monoline mark, teal plus deep navy plus a restrained coral accent, strong silhouette at 64 pixels, no hospital cross, no moon-and-stars cliché, no human or patient imagery, no gradient, no 3D, no tiny unreadable text. Provide light and dark background variants and a wordmark with exact spelling SLEEPJEV.
