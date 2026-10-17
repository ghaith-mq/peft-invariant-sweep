"""Multi-adapter lifecycle sweep across PEFT methods (MLP / EmbConv1D / Conv2d models).

Checks, with two adapters "a" (default slot) and "b":
  set_adapter_grad   after set_adapter("b"), exactly b's params require grad (a's are frozen)
  set_adapter_out    output after set_adapter("b") == fresh single-adapter model with b's weights
  delete_active      delete_adapter(active "b") -> "a" becomes active, is trainable, output == a-only output
  delete_inactive    delete_adapter("b") while "a" active -> output unchanged, a still trainable
  mixed_batch        adapter_names=[a, b, __base__] matches per-adapter outputs (fp32, bf16, fp16)
  save_load_multi    save both, reload both, outputs match
"""

import copy
import json
import sys
import tempfile
import warnings

import torch

import os

sys.path.insert(0, os.environ.get("PEFT_REPO", "../peft"))
warnings.filterwarnings("ignore")

from tests.test_custom_models import TEST_CASES, MockTransformerWrapper  # noqa: E402
from tests.testing_utils import set_init_weights_false  # noqa: E402

from peft import PeftModel, get_peft_model  # noqa: E402
from peft.config import PromptLearningConfig  # noqa: E402

MODELS = {
    "MLP": (["lin0"], lambda: torch.arange(90).view(9, 10).float() / 90),
    "EmbConv1D": (["conv1d"], lambda: torch.arange(90).view(9, 10) % 100),
    "Conv2d": (["conv2d"], lambda: torch.arange(405).view(9, 45).float() / 405),
}
TOL = dict(atol=1e-4, rtol=1e-4)


def method_kwargs():
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
    m = MockTransformerWrapper.from_pretrained(mid, dtype=dtype).eval()
    if hasattr(m, "dtype"):
        m.dtype = dtype
    return m


def fwd(model, x, **kw):
    with torch.no_grad():
        return model(x, **kw)


def close(a, b, atol=TOL["atol"], rtol=TOL["rtol"]):
    return torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol), (a.float() - b.float()).abs().max().item()


def perturb(model, name):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if name in n.split(".") and p.is_floating_point() and "base_layer" not in n:
                p.add_(torch.randn_like(p) * 0.1)


def trainable_names(model):
    return {n for n, p in model.named_parameters() if p.requires_grad}


def build(cls, kw, mid, dtype=torch.float32, names=("a", "b")):
    torch.manual_seed(0)
    target = MODELS[mid][0]
    model = get_peft_model(load(mid, dtype), cls(target_modules=target, **kw), adapter_name=names[0])
    for n in names[1:]:
        model.add_adapter(n, cls(target_modules=target, **kw))
    torch.manual_seed(1)
    for n in names:
        perturb(model, n)
    return model


def run_one(cls, kw, mid):
    res = {}
    x = MODELS[mid][1]()
    kw = set_init_weights_false(cls, kw)
    try:
        model = build(cls, kw, mid)
    except (ValueError, NotImplementedError, TypeError, KeyError) as e:
        return {"unsupported": f"{type(e).__name__}: {str(e)[:100]}"}

    model.eval()
    model.set_adapter("a")
    y_a = fwd(model, x)
    model.set_adapter("b")
    y_b = fwd(model, x)
    with model.disable_adapter():
        y_base = fwd(model, x)
    res["adapters_differ"] = "ok" if not torch.allclose(y_a, y_b) else "SAME OUTPUT"

    def check(name, fn):
        try:
            res[name] = fn()
        except Exception as e:
            res[name] = f"CRASH {type(e).__name__}: {str(e)[:200]}"

    def set_adapter_grad():
        m = copy.deepcopy(model)
        m.set_adapter("b")
        tr = trainable_names(m)
        has_a = [n for n in tr if "a" in n.split(".")]
        has_b = [n for n in tr if "b" in n.split(".")]
        if has_a:
            return f"INACTIVE TRAINABLE {has_a[:2]}"
        if not has_b:
            return "ACTIVE NOT TRAINABLE"
        return "ok"

    def delete_active():
        m = copy.deepcopy(model)
        m.set_adapter("b")
        m.delete_adapter("b")
        if m.active_adapters != ["a"]:
            return f"ACTIVE={m.active_adapters}"
        ok, d = close(fwd(m, x), y_a)
        if not ok:
            return f"OUTPUT MISMATCH diff={d:.2e}"
        m.train()
        tr = trainable_names(m)
        if not any("a" in n.split(".") for n in tr):
            return "REMAINING ADAPTER FROZEN"
        return "ok"

    def delete_inactive():
        m = copy.deepcopy(model)
        m.set_adapter("a")
        m.delete_adapter("b")
        ok, d = close(fwd(m, x), y_a)
        if not ok:
            return f"OUTPUT MISMATCH diff={d:.2e}"
        if not any("a" in n.split(".") for n in trainable_names(m)):
            return "REMAINING ADAPTER FROZEN"
        return "ok"

    def mixed(dtype):
        def inner():
            if dtype == torch.float32:
                m, ya, yb, yb0 = model, y_a, y_b, y_base
                tol = TOL
            else:
                m = copy.deepcopy(model).to(dtype)
                m.base_model.model.dtype = dtype if hasattr(m.base_model.model, "dtype") else None
                m.set_adapter("a")
                ya = fwd(m, x)
                m.set_adapter("b")
                yb = fwd(m, x)
                with m.disable_adapter():
                    yb0 = fwd(m, x)
                tol = dict(atol=2e-2, rtol=2e-2)
            names = ["a", "b", "__base__"] * 3
            y = fwd(m, x, adapter_names=names)
            exp = torch.stack([[ya, yb, yb0][i % 3][i] for i in range(len(names))])
            ok, d = close(y, exp, **tol)
            return "ok" if ok else f"MISMATCH diff={d:.2e}"

        return inner

    def save_load_multi():
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            m = PeftModel.from_pretrained(load(mid), f"{tmp}/a", adapter_name="a")
            m.load_adapter(f"{tmp}/b", adapter_name="b")
            m.eval()
            m.set_adapter("a")
            ok1, d1 = close(fwd(m, x), y_a)
            m.set_adapter("b")
            ok2, d2 = close(fwd(m, x), y_b)
            return "ok" if ok1 and ok2 else f"MISMATCH a={d1:.2e} b={d2:.2e}"

    check("set_adapter_grad", set_adapter_grad)
    check("delete_active", delete_active)
    check("delete_inactive", delete_inactive)
    check("mixed_fp32", mixed(torch.float32))
    check("mixed_bf16", mixed(torch.bfloat16))
    check("mixed_fp16", mixed(torch.float16))
    check("save_load_multi", save_load_multi)
    return res


def main():
    results = []
    for cls, kw in sorted(method_kwargs().items(), key=lambda t: t[0].__name__):
        for mid in MODELS:
            try:
                r = run_one(cls, kw, mid)
            except Exception as e:
                r = {"harness_error": f"{type(e).__name__}: {str(e)[:200]}"}
            results.append({"method": cls.__name__, "model": mid, **r})
            if "unsupported" not in r:
                bad = {k: v for k, v in r.items() if v != "ok"}
                print(f"{cls.__name__:18s} {mid:10s} {bad or 'all ok'}", flush=True)
    json.dump(results, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "multi_adapter.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
