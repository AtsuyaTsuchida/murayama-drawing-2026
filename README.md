# Murayama Drawing Generation

Learning the drawing process of artist Goro Murayama (stroke-by-stroke fixed-camera
captures, 602 works) to generate the next stroke in his style. The goal is
autoregressive generation: feeding the output back as input to produce a time series
of a drawing growing stroke by stroke.

## Structure

- `scripts/` — dataset construction, training launchers, generation, evaluation
- `src_modified/` — modified files from img2img-turbo (stochastic mode wired into training)

## Experiments

| # | Implementation | Base model | Description |
| --- | --- | --- | --- |
| 1 | img2img-turbo | stabilityai/sd-turbo | Full-state prediction. Desaturation and blur |
| 2 | img2img-turbo | stabilityai/sd-turbo | Diff prediction, deterministic. Blank became the loss optimum |
| 3 | img2img-turbo | stabilityai/sd-turbo | Diff prediction, stochastic. Same outcome |
| 4 | diffusers | timbrooks/instruct-pix2pix | N to N+1. Style successfully acquired |

Experiment 4 is the model used for autoregressive generation.

## Autoregressive generation

`scripts/autoregressive_k1_accum.py`

Naively feeding the output back as input breaks down within a few iterations because
VAE reconstruction error accumulates. The accumulative composition method solves this:
only newly drawn regions are extracted and stacked onto the canvas, and already-drawn
regions are never overwritten. Confirmed stable for 100 iterations, including
generation from a blank canvas created with `Image.new()`.

Key parameters: `LO=18, HI=42, BLUR=1.5, image_guidance_scale=1.2, guidance_scale=7.5`

Trade-off: overpainting is not possible. Partially relaxing the protection (30% / 50%)
was tested and still produced blur and bleeding, so full protection is kept.

## A note on evaluation

Evaluation in this project is primarily visual. Generated images are inspected directly.

`scripts/check_masked_snr.py` computes the signal-to-noise ratio inside the
changed-region mask. It is offered as one way to sanity-check whether a metric can
detect the change at all, not as a decisive quality measure.

## References

- img2img-turbo: https://github.com/GaParmar/img2img-turbo
- InstructPix2Pix: https://github.com/timothybrooks/instruct-pix2pix

## Note

Only the scripts on the path to the current result are kept here. Intermediate
variations (N to N+5 / N to N+10 datasets, alternative composition methods) were
explored but are not included.
