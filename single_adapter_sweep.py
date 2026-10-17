"""Invariant sweep across PEFT methods x tiny model types x dtypes.

For every (method, model) pair where wrapping succeeds, check:
  init_identity   default init leaves output unchanged (only reported; not every method is identity at init)
  merge           merged output == unmerged output (non-identity init)
  merge_unmerge   merge() then unmerge() restores the unmerged output
  unload          unload() returns a model whose output == base model
  disable         disable_adapter() context gives base output
  save_load       save_pretrained / from_pretrained round trip preserves output
  half_forward    forward in bf16/fp16 runs and returns the base dtype
  half_merge      merge in bf16 roughly matches unmerged bf16 output

Combos already in tests/test_custom_models.py TEST_CASES are marked so we can focus on untested ones.
"""

import copy
import json
import sys
import tempfile
import traceback
import warnings

import torch

import os

sys.path.insert(0, os.environ.get("PEFT_REPO", "../peft"))
warnings.filterwarnings("ignore")

from tests.test_custom_models import TEST_CASES, MockTransformerWrapper  # noqa: E402
from tests.testing_utils import set_init_weights_false  # noqa: E402

from peft import PeftModel, get_peft_model  # noqa: E402
from peft.config import PromptLearningConfig  # noqa: E402

# model_id -> (target modules, input factory)
MODELS = {
    "MLP": (["lin0"], lambda: torch.arange(90).view(9, 10).float() / 90),
    "EmbConv1D": (["conv1d"], lambda: torch.arange(90).view(9, 10) % 100),
    "EmbConv1D-emb": (["emb"], lambda: torch.arange(90).view(9, 10) % 100),
    "Conv1d": (["conv1d"], lambda: torch.arange(90).view(9, 10).float() / 90),
    "Conv1dBigger": (["conv1d"], lambda: torch.arange(90).view(9, 10).float() / 90),
    "Conv2d": (["conv2d"], lambda: torch.arange(405).view(9, 45).float() / 405),
    "Conv2d1x1": (["conv2d"], lambda: torch.arange(81).view(9, 9).float() / 81),
    "Conv3d": (["conv3d"], lambda: torch.arange(405).view(9, 45).float() / 405),
    "Conv3d1x1": (["conv3d"], lambda: torch.arange(405).view(9, 45).float() / 405),
    "MHA": (["mha"], lambda: torch.arange(90).view(9, 10).float() / 90),
}
TOL = dict(atol=1e-4, rtol=1e-4)


def base_model_id(mid):
    return mid.split("-")[0]


def make_input(mid):
    return MODELS[mid][1]()


def get_tested_combos():
    tested = set()
    for _, model_id, cls, kwargs in TEST_CASES:
        tm = kwargs.get("target_modules")
        tm = [tm] if isinstance(tm, str) else (tm or [])
        tested.add((cls.__name__, model_id, tuple(sorted(tm))))
    return tested


def method_kwargs():
    # one representative kwargs dict per config class, taken from the MLP test cases when possible
    out = {}
    for _, model_id, cls, kwargs in TEST_CASES:
        if issubclass(cls, PromptLearningConfig):
            continue
        if any(k in kwargs for k in ("target_parameters", "alora_invocation_tokens", "layers_to_transform")):
            continue
        if kwargs.get("modules_to_save") or kwargs.get("trainable_token_indices"):
            continue
        if cls not in out or (model_id == "MLP" and out[cls][0] != "MLP"):
            out[cls] = (model_id, {k: v for k, v in kwargs.items() if k != "target_modules"})
    return {cls: kw for cls, (_, kw) in out.items()}


def load(mid, dtype=torch.float32):
    return MockTransformerWrapper.from_pretrained(base_model_id(mid), dtype=dtype).eval()


def fwd(model, x):
    with torch.no_grad():
        return model(x)


def close(a, b, **tol):
    tol = tol or TOL
    return torch.allclose(a.float(), b.float(), **tol), (a.float() - b.float()).abs().max().item()


def run_one(cls, kw, mid):
    res = {}
    target = MODELS[mid][0]
    x = make_input(mid)

    base = load(mid)
    y_base = fwd(base, x)

    # 1) default init identity
    try:
        m = get_peft_model(load(mid), cls(target_modules=target, **kw)).eval()
        ok, d = close(fwd(m, x), y_base)
        res["init_identity"] = "ok" if ok else f"diff={d:.2e}"
    except (ValueError, NotImplementedError, TypeError, KeyError) as e:
        return {"unsupported": f"{type(e).__name__}: {str(e)[:120]}"}
    except Exception as e:
        return {"wrap_crash": f"{type(e).__name__}: {str(e)[:200]}"}

    kw_nz = set_init_weights_false(cls, kw)
    torch.manual_seed(1)
    try:
        model = get_peft_model(load(mid), cls(target_modules=target, **kw_nz)).eval()
    except Exception as e:
        return {**res, "wrap_nonzero_crash": f"{type(e).__name__}: {str(e)[:200]}"}
    # make sure adapter weights are not trivially zero
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad and p.is_floating_point() and p.abs().sum() == 0:
                p.add_(torch.randn_like(p) * 0.1)
    y = fwd(model, x)
    res["adapter_effective"] = "ok" if not torch.allclose(y, y_base) else "NO EFFECT"

    def check(name, fn):
        try:
            res[name] = fn()
        except Exception as e:
            res[name] = f"CRASH {type(e).__name__}: {str(e)[:200]}"

    def merge():
        m = copy.deepcopy(model)
        m.merge_adapter()
        ok, d = close(fwd(m, x), y)
        return "ok" if ok else f"MISMATCH diff={d:.2e}"

    def merge_unmerge():
        m = copy.deepcopy(model)
        m.merge_adapter()
        m.unmerge_adapter()
        ok, d = close(fwd(m, x), y)
        return "ok" if ok else f"MISMATCH diff={d:.2e}"

    def merge_and_unload():
        m = copy.deepcopy(model).merge_and_unload()
        ok, d = close(fwd(m, x), y)
        return "ok" if ok else f"MISMATCH diff={d:.2e}"

    def unload():
        m = copy.deepcopy(model).unload()
        ok, d = close(fwd(m, x), y_base)
        return "ok" if ok else f"MISMATCH diff={d:.2e}"

    def disable():
        m = copy.deepcopy(model)
        with m.disable_adapter():
            ok, d = close(fwd(m, x), y_base)
        ok2, _ = close(fwd(m, x), y)
        if not ok:
            return f"MISMATCH diff={d:.2e}"
        return "ok" if ok2 else "NOT RE-ENABLED"

    def save_load():
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            m = PeftModel.from_pretrained(load(mid), tmp).eval()
            ok, d = close(fwd(m, x), y)
            return "ok" if ok else f"MISMATCH diff={d:.2e}"

    def half(dtype):
        def inner():
            torch.manual_seed(1)
            m = get_peft_model(load(mid, dtype), cls(target_modules=target, **kw_nz)).eval()
            m.load_state_dict(model.state_dict(), strict=False)
            if hasattr(m.base_model.model, "dtype"):
                m.base_model.model.dtype = dtype
            yh = fwd(m, x)
            if yh.dtype != dtype:
                return f"DTYPE {yh.dtype}"
            m.merge_adapter()
            ok, d = close(fwd(m, x), yh, atol=5e-2, rtol=5e-2)
            return "ok" if ok else f"MERGE MISMATCH diff={d:.2e}"

        return inner

    check("merge", merge)
    check("merge_unmerge", merge_unmerge)
    check("merge_and_unload", merge_and_unload)
    check("unload", unload)
    check("disable", disable)
    check("save_load", save_load)
    check("bf16", half(torch.bfloat16))
    check("fp16", half(torch.float16))
    return res


def main():
    tested = get_tested_combos()
    results = []
    for cls, kw in sorted(method_kwargs().items(), key=lambda t: t[0].__name__):
        for mid in MODELS:
            key = (cls.__name__, base_model_id(mid), tuple(sorted(MODELS[mid][0])))
            try:
                r = run_one(cls, kw, mid)
            except Exception:
                r = {"harness_error": traceback.format_exc(limit=2)[-300:]}
            results.append({"method": cls.__name__, "model": mid, "in_test_suite": key in tested, **r})
            bad = {k: v for k, v in r.items() if v != "ok" and k != "init_identity"}
            print(f"{cls.__name__:18s} {mid:14s} tested={key in tested!s:5s} {bad or 'all ok'}", flush=True)
    json.dump(results, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "single_adapter.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
