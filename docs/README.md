# Documentation

| folder | presentation | content |
|---|---|---|
| `howto/` | `howto_pipeline.pdf` | installation, input data structure, config file, the four steps, outputs, troubleshooting |
| `ting/` | `ting_fitting.pdf` | Ting viscoelastic fit: force conversion, contact mechanics, Lee–Radok / Ting, discretisation, fit, slope correction, parameters |
| `adhesion/` | `adhesion_properties.pdf` | every adhesion property on the retract curve: definition, how it is measured or computed |
| `variants/` | `model_variants.pdf` | model variants of the Ting fit (contact point, toe force, reference time): E, α, contrast, E–α relationship |
| `segmentation/` | `segmentation_tracking.pdf` | cell segmentation and tracking: every step, the math, worked examples |

Each folder: `<name>.tex` (Beamer, metropolis theme), `<name>.pdf`, `fig/` (figures used by the .tex),
`scripts/` (Python scripts that made the figures from the example data set data_213626 / the time-lapse;
paths inside are those of the analysis machine and have to be adjusted to rerun them; `facts*.json` = numbers quoted in the slides).

Compile: `latexmk -pdf <name>.tex` in the folder (needs TeX Live with beamer, metropolis, tikz, siunitx, listings).
