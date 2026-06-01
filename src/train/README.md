# src/train — Training loop and training utilities

## Contract

**Produces:** Model checkpoints (`.pt`) + sidecar JSON per checkpoint.

**Consumes:** PyTorch Datasets from `src/data/`, model classes from `src/models/`.

**Needs GPU:** Yes.

**Does NOT contain:** Dataset definitions, model architecture definitions, or metric computation.

## Key files

| File | Purpose |
|------|---------|
| `loop.py` | `train_one_epoch(model, loader, optimizer, device, loss_fn)` and `eval_one_epoch(...)`. Returns scalar loss and per-sample predictions. |
| `factory.py` | `build_optimizer(model, cfg)`, `build_scheduler(optimizer, cfg)`, `build_loss(cfg, class_weights)`. |
| `seed.py` | `set_all_seeds(seed)` — sets Python `random`, NumPy, PyTorch CPU+CUDA, `cudnn.deterministic=True`, `cudnn.benchmark=False`. |
| `callbacks.py` | `EarlyStopping(patience=7)`, `GradientClipper(max_norm=1.0)`, `ProgressiveUnfreeze(epochs=[3,6])`. |
| `sidecar.py` | `write_sidecar(checkpoint_path, cfg, seed)` — writes JSON with config, seed, git SHA, Python/lib versions, hostname, GPU model, wall-clock time. |
| `resume.py` | `find_checkpoint(run_dir)` and `load_checkpoint(path, model, optimizer)` — resume from interrupted run. |

## Stage 1 training protocol

- Optimizer: AdamW
- LR schedule: 3-epoch warmup → ReduceLROnPlateau on val macro-F1 (patience 7)
- Backbone unfreezing: epochs 3 and 6 (progressive)
- Early stopping: patience 7 on val macro-F1
- Gradient clipping: max norm 1.0
- Loss: class-weighted CrossEntropyLoss
- Max epochs: 30
- Seeds: 5 (42, 123, 456, 789, 1337)

## Stage 2 training protocol

- Optimizer: AdamW
- LR schedule: cosine warmup (1 epoch → 20 epochs)
- Batch sampling: class-balanced (half real / half fake per batch)
- Loss: BCEWithLogitsLoss
- JPEG aug: random quality in [75, 98] per image
- Max epochs: 20
- Seeds: 1 headline (42); 3 for multi-seed (42, 123, 235)
