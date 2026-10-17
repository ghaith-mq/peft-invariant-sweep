# peft-invariant-sweep

Property-based checks for [🤗 PEFT](https://github.com/huggingface/peft) that run across every method. PEFT has 40+ fine-tuning methods (LoRA, IA³, LoHa, VeRA, OFT, ...). Every one of them should satisfy the same mathematical and API invariants, but each method's tests only cover a few hand-picked layer types and configurations. These scripts check the invariants on every method × layer type × dtype combination, so they can find bugs in the gaps.

## What is checked

**`single_adapter_sweep.py`**: one adapter, on Linear, transformers `Conv1D`, `Embedding`, `Conv1d`, `Conv2d` (3×3 and 1×1), `Conv3d` and `MultiheadAttention` targets:

| Invariant | Property |
|---|---|
| init identity | the default initialization leaves the model output unchanged (where the method promises it) |
| merge | `merge_adapter()` gives the same output as the unmerged forward |
| merge → unmerge | restores the unmerged output |
| merge_and_unload / unload | gives the merged output and the base output respectively |
| disable_adapter | gives the base output, then re-enables cleanly |
| save / load | a `save_pretrained` → `from_pretrained` round trip preserves the output |
| bf16 / fp16 | half-precision forward keeps the dtype, and merging matches the unmerged output |

**`multi_adapter_sweep.py`**: two adapters on the same model:

| Invariant | Property |
|---|---|
| set_adapter | only the active adapter's parameters require grad |
| delete active / inactive adapter | the remaining adapter becomes active, stays trainable, and its output is unchanged |
| save / load (multi) | both adapters round-trip |
| mixed adapter batch | `model(x, adapter_names=["a", "b", "__base__", ...])` equals running each row with its own adapter (fp32 / bf16 / fp16) |

## Results (PEFT `main` @ `c2d5f69`, torch 2.14, CPU)

- **Single adapter:** 109 supported method × layer combinations. All merge, unload, disable, save/load and half-precision invariants hold, apart from documented limitations: Lily and OSF don't support merging, and OSF on non-Linear layers is already reported in huggingface/peft#3986.
- **Multi-adapter:** 53 supported combinations. set_adapter, delete_adapter and multi-adapter save/load all pass (53/53).
- **Bug found:** mixed adapter batches pass for only **LoRA, RoAd and GLoRA**. For every other method, `adapter_names` is **silently ignored** and the whole batch runs through the active adapter, with no error or warning. Minimal repro: [`repro_adapter_names_ignored.py`](repro_adapter_names_ignored.py).

  *Root cause:* `PeftModel._enable_peft_forward_hooks` only delegates to the tuner when the tuner defines that hook. Otherwise `adapter_names` is stripped from the kwargs and never reaches any layer.

Raw per-combination results are in [`results/`](results/).

## Running

```bash
git clone https://github.com/huggingface/peft
pip install -e "./peft[test]"
PEFT_REPO=./peft python single_adapter_sweep.py
PEFT_REPO=./peft python multi_adapter_sweep.py
```

The scripts reuse the tiny models and per-method configs from PEFT's own `tests/test_custom_models.py`, so new methods are picked up automatically.

## Notes on methodology

- Adapters are created with non-identity initialization and small random perturbations, so a passing invariant is not trivially satisfied by zero weights. An `adapters_differ` / `adapter_effective` sanity check flags cases where the adapter has no effect.
- A combination where PEFT raises a clear `TypeError`, `ValueError` or `NotImplementedError` when wrapping is recorded as *unsupported*, not as a failure.
- Tolerances: `1e-4` in fp32 and `2e-2`–`5e-2` in half precision.

Built with AI assistance (Claude Code); every reported finding was reproduced and checked by hand.
