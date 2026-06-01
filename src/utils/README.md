# src/utils — Shared utilities

## Contract

**Produces:** Helper functions used by multiple modules with no upward dependencies.

**Consumes:** Only stdlib and third-party packages. Never imports from other `src/` modules.

## Key files (to be created in later blocks)

| File | Purpose |
|------|---------|
| `paths.py` | `load_paths(paths_yaml=None) -> Namespace` — reads `paths.yaml` and applies env-var overrides |
| `hashing.py` | `config_hash(cfg_dict) -> str` — SHA-256 of a config dict for content-addressed output dirs |
| `logging.py` | `get_logger(name) -> logging.Logger` — shared log format |
