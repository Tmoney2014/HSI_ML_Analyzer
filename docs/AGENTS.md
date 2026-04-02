# docs/ - Design Spec Map

**Generated:** 2026-04-02  
**Scope:** `docs/` only

## OVERVIEW

Specification source-of-truth for pipeline invariants, runtime contract, and cache ownership rules shared by Python trainer and C# runtime.

## STRUCTURE

```
docs/
├── data_pipeline_spec.md      # End-to-end processing invariants
├── inference_runtime_spec.md  # C# runtime contract and performance constraints
└── cache_hierarchy_spec_ko.md # L1/L2 cache ownership + invalidation rules
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Raw masking invariant | `data_pipeline_spec.md` | Masking must run on Raw DN |
| Lazy processing rule | `data_pipeline_spec.md` | Cache base data; apply prep on demand |
| model.json metadata contract | `data_pipeline_spec.md`, `inference_runtime_spec.md` | `RequiredRawBands` + preprocessing config required |
| Runtime operator semantics | `inference_runtime_spec.md` | `MaskRules` parser + derivative direction |
| Runtime performance constraints | `inference_runtime_spec.md` | Line latency target, zero-allocation guidance |
| Cache ownership/thread model | `cache_hierarchy_spec_ko.md` | L1 lock policy, L2 main-thread ownership |
| Cache invalidation triggers | `cache_hierarchy_spec_ko.md` | mode/ref/mask/file/prep change behavior |

## CONVENTIONS

- Treat these specs as invariants; code changes should align here first.
- Pipeline order and parity requirements override local implementation convenience.
- Cache semantics are authoritative in `cache_hierarchy_spec_ko.md`.

## ANTI-PATTERNS

- Changing preprocessing formulas without updating parity expectations.
- Running masking on Reflectance/Absorbance instead of Raw DN.
- Writing L2 cache from worker threads directly (bypass signal protocol).
- Shipping runtime/export changes without updating spec references.

## NOTES

- Korean + English mixed docs are intentional; preserve key technical terms.
- Use this directory for architecture/specs, not generated runtime artifacts.
