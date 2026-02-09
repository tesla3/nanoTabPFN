"""
Validate that the new SDPA-based HypernetMultiheadAttention produces
identical outputs to the original einsum-based implementation.
"""
import math
import torch
import torch.nn.functional as F
from torch import nn


class OriginalHypernetMultiheadAttention(nn.Module):
    """Original implementation (before SDPA replacement) for reference."""

    def __init__(self, embed_dim, num_heads, target_network="default", attn_norm="softmax"):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.target_network = target_network
        self.attn_norm = attn_norm

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        B, Q_len, E = query.shape
        _, K_len, _ = key.shape
        H = self.num_heads
        D = self.head_dim

        # Original: reshape to [B, S, H, D] (NOT transposed)
        q = self.q_proj(query).view(B, Q_len, H, D)
        k = self.k_proj(key).view(B, K_len, H, D)
        v = self.v_proj(value).view(B, K_len, H, D)

        # Original einsum for scores
        scale = 1.0 / math.sqrt(D)
        attn_scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale

        if self.attn_norm == "softmax":
            attn_weights = torch.softmax(attn_scores, dim=-1)
        elif self.attn_norm == "rms_head":
            rms = torch.sqrt(torch.mean(attn_scores ** 2, dim=1, keepdim=True) + 1e-8)
            attn_weights = attn_scores / rms
        elif self.attn_norm == "none":
            attn_weights = attn_scores

        # Original einsums with v=[B, K, H, D]
        if self.target_network == "default":
            x = torch.einsum("bhqk,bkhd->bqhd", attn_weights, v)
        elif self.target_network == "mlp_silu":
            x = torch.einsum("bhqk,bkhd->bqkd", attn_weights, v)
            x = F.silu(x)
            x = torch.einsum("bhqk,bqkd->bqhd", attn_weights, x)
        elif self.target_network == "mlp_linear":
            x = torch.einsum("bhqk,bkhd->bqkd", attn_weights, v)
            x = torch.einsum("bhqk,bqkd->bqhd", attn_weights, x)

        x = x.reshape(B, Q_len, E)
        x = self.out_proj(x)
        return x, None


def copy_weights(src, dst):
    """Copy all projection weights from src to dst."""
    dst.q_proj.weight.data.copy_(src.q_proj.weight.data)
    dst.q_proj.bias.data.copy_(src.q_proj.bias.data)
    dst.k_proj.weight.data.copy_(src.k_proj.weight.data)
    dst.k_proj.bias.data.copy_(src.k_proj.bias.data)
    dst.v_proj.weight.data.copy_(src.v_proj.weight.data)
    dst.v_proj.bias.data.copy_(src.v_proj.bias.data)
    dst.out_proj.weight.data.copy_(src.out_proj.weight.data)
    dst.out_proj.bias.data.copy_(src.out_proj.bias.data)


def test_variant(target_network, attn_norm, B=2, Q=10, K=12, E=32, H=4):
    """Test one attention variant. Returns max absolute difference."""
    from model import HypernetMultiheadAttention as NewAttn

    torch.manual_seed(42)
    orig = OriginalHypernetMultiheadAttention(E, H, target_network, attn_norm)
    new = NewAttn(E, H, target_network, attn_norm)

    # Copy weights from original to new
    copy_weights(orig, new)

    orig.eval()
    new.eval()

    torch.manual_seed(123)
    query = torch.randn(B, Q, E)
    key = torch.randn(B, K, E)
    value = torch.randn(B, K, E)

    with torch.no_grad():
        out_orig, _ = orig(query, key, value)
        out_new, _ = new(query, key, value)

    diff = (out_orig - out_new).abs().max().item()
    return diff


def main():
    variants = [
        ("default", "softmax"),
        ("default", "rms_head"),
        ("default", "none"),
        ("mlp_silu", "softmax"),
        ("mlp_silu", "rms_head"),
        ("mlp_silu", "none"),
        ("mlp_linear", "softmax"),
        ("mlp_linear", "rms_head"),
        ("mlp_linear", "none"),
    ]

    print("Validating SDPA replacement against original einsum implementation")
    print("=" * 70)
    all_pass = True
    for tn, an in variants:
        diff = test_variant(tn, an)
        # SDPA uses a different numerical path, so allow small tolerance
        tol = 1e-5
        status = "PASS" if diff < tol else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {tn:12s} + {an:10s} : max_diff={diff:.2e}  [{status}]")

    print("=" * 70)
    if all_pass:
        print("All variants PASSED")
    else:
        print("Some variants FAILED")

    # Also test with self-attention (Q=K=V same tensor)
    print("\nSelf-attention (query=key=value):")
    print("-" * 70)
    from model import HypernetMultiheadAttention as NewAttn
    for tn, an in variants:
        torch.manual_seed(42)
        E, H, B, S = 32, 4, 2, 10
        orig = OriginalHypernetMultiheadAttention(E, H, tn, an)
        new = NewAttn(E, H, tn, an)
        copy_weights(orig, new)
        orig.eval(); new.eval()
        torch.manual_seed(123)
        x = torch.randn(B, S, E)
        with torch.no_grad():
            out_orig, _ = orig(x, x, x)
            out_new, _ = new(x, x, x)
        diff = (out_orig - out_new).abs().max().item()
        tol = 1e-5
        status = "PASS" if diff < tol else "FAIL"
        print(f"  {tn:12s} + {an:10s} : max_diff={diff:.2e}  [{status}]")


if __name__ == "__main__":
    main()
