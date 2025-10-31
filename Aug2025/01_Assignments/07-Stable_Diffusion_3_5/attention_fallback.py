
import torch
import torch.nn.functional as F

def flash_attention_fallback(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
    """
    Fallback implementation using PyTorch's native scaled dot product attention
    This provides similar functionality to flash_attn when it's not available
    """
    return F.scaled_dot_product_attention(
        q, k, v, 
        attn_mask=attn_mask, 
        dropout_p=dropout_p, 
        is_causal=is_causal
    )

# Test the fallback
def test_attention_fallback():
    # Create sample tensors
    batch_size, seq_len, head_dim = 2, 512, 64
    q = torch.randn(batch_size, 8, seq_len, head_dim, device='cuda' if torch.cuda.is_available() else 'cpu')
    k = torch.randn(batch_size, 8, seq_len, head_dim, device='cuda' if torch.cuda.is_available() else 'cpu')
    v = torch.randn(batch_size, 8, seq_len, head_dim, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        output = flash_attention_fallback(q, k, v)
        print(f"✅ Attention fallback works! Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"❌ Attention fallback failed: {e}")
        return False

# Save fallback to a file for reuse
fallback_working = test_attention_fallback()
if fallback_working:
    print("💾 You can use this fallback in your models when flash-attn is not available")
