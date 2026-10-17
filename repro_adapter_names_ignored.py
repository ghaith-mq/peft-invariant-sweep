import torch
from torch import nn
from peft import IA3Config, LoHaConfig, VeraConfig, get_peft_model

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin0 = nn.Linear(10, 10)
        self.lin1 = nn.Linear(10, 2)
    def forward(self, x):
        return self.lin1(self.lin0(x).relu())

for cfg_cls, kw in [
    (IA3Config, dict(init_ia3_weights=False, feedforward_modules=[])),
    (LoHaConfig, dict(init_weights=False)),
    (VeraConfig, dict(init_weights=False, r=4)),
]:
    torch.manual_seed(0)
    model = get_peft_model(MLP(), cfg_cls(target_modules=["lin0"], **kw), adapter_name="first").eval()
    model.add_adapter("second", cfg_cls(target_modules=["lin0"], **kw))
    x = torch.randn(4, 10)
    with torch.no_grad():
        model.set_adapter("first");  y_first = model(x)
        model.set_adapter("second"); y_second = model(x)
        with model.disable_adapter():
            y_base = model(x)
        y_mixed = model(x, adapter_names=["first", "first", "__base__", "__base__"])

    print(
        cfg_cls.__name__,
        "| rows 0-1 use 'first':", torch.allclose(y_mixed[:2], y_first[:2]),     # expected True, got False
        "| rows 2-3 use base:", torch.allclose(y_mixed[2:], y_base[2:]),          # expected True, got False
        "| all rows use active 'second':", torch.allclose(y_mixed, y_second),    # expected False, got True
    )
