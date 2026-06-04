# _archive — retired / superseded material

Moved here 2026-06-04 to declutter the active tree. Nothing here is used by
the paper, the active `v5_rankbind` codebase, or any running job. Everything
remains in git history and on disk; restore with a plain `mv` if ever needed.

- `v4_residue_only/` — older standalone "residue-only" model experiment,
  superseded by the residue attention-pool extension (v5b) inside
  `v5_rankbind/`. ~12 GB (checkpoints/caches; git-ignored).
- `baselines_dropped/{deepdta,dualbind_nvidia,gign}/` — baseline models
  dropped after Phase 1 (the locked comparison set is graphdta / moltrans /
  drugban / gems, which stay under `baselines/`).

See `PROJECT_MAP.md` for the current structure.
