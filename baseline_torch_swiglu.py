import torch
import torch.nn as nn
import torch.nn.functional as F


class PyTorchSwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.w_proj = nn.Linear(in_features, hidden_features, bias=False)
        self.v_proj = nn.Linear(in_features, hidden_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up_proj = self.w_proj(x)
        up_proj_activated = F.silu(up_proj)
        gate_proj = self.v_proj(x)
        output = up_proj_activated * gate_proj
        return output

if __name__ == "__main__":
    BATCH_SIZE = 4
    SEQ_LEN = 2048
    IN_FEATURES = 1024  # K
    HIDDEN_FEATURES = 4096  # N
    DEVICE = 'cuda'
    DTYPE = torch.bfloat16

    model = PyTorchSwiGLU(IN_FEATURES, HIDDEN_FEATURES).to(DEVICE, DTYPE)

    input_tensor = torch.randn(
        (BATCH_SIZE * SEQ_LEN, IN_FEATURES),
        device=DEVICE,
        dtype=DTYPE
    )

    print("--- PyTorch Baseline ---")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Weight W shape: {model.w_proj.weight.T.shape}")
    print(f"Weight V shape: {model.v_proj.weight.T.shape}")

    output = model(input_tensor)

    print(f"Output shape: {output.shape}")
    assert output.shape == (BATCH_SIZE * SEQ_LEN, HIDDEN_FEATURES)
    print("Execution successful. Shapes are correct.")