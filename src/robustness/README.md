# src/robustness — Perturbation grid

## Contract

**Produces:** One large CSV per (model, dataset) with columns:
`model, dataset, perturbation_family, perturbation_value, auc, eer, fpr_at_tpr95, ece, brier`

**Consumes:** Pre-existing model checkpoints; face-crop manifests from `src/data/`.

**Needs GPU:** Yes for inference; No for metric aggregation.

## Perturbation families

| Family | Parameters |
|--------|------------|
| JPEG | quality ∈ {95, 75, 55, 40} |
| Gaussian blur | σ ∈ {0.5, 1.0, 2.0, 3.0} |
| Denoise/sharpen | combinations: {none, denoise, sharpen, both} |
| Gamma | γ ∈ {0.5, 0.75, 1.0, 1.5, 2.0} |
| Bilinear resize | factor ∈ {0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0} |

## Key file

`perturbations.py` — five pure image→image transform functions, one per family.
Each takes an image array and a parameter value; returns the perturbed image.
The identity-parameter value (JPEG q=100, blur σ=0, gamma=1.0, resize=1.0)
must produce the exact input image unchanged (verified in unit tests).
