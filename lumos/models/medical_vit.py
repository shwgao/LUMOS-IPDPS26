import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class MedicalVisionTransformer(nn.Module):
    """
    Vision Transformer specifically designed for medical image classification
    Optimized for small medical datasets with data augmentation and regularization
    """
    
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        num_classes=3,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0.1,
        emb_dropout=0.1,
        pool='cls',
        channels=3
    ):
        super().__init__()
        
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        # Patch embedding
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )
        
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        # Transformer blocks
        self.transformer = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)
        ])
        
        self.pool = pool
        self.to_latent = nn.Identity()
        
        # Classification head with dropout for regularization
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        
        # Add cls token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        
        # Apply transformer blocks
        for transformer in self.transformer:
            x = transformer(x)
        
        # Pool features
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        
        return self.mlp_head(x)


class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and MLP"""
    
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        self.skip_attn = False
        self.skip_mlp = False
        self.skip_forward = False
    
    def forward(self, x):
        # Self-attention with residual connection
        norm_x = self.norm1(x)
        attn_out = self.attn(norm_x)
        x = x + self.dropout(attn_out)
        
        # MLP with residual connection
        norm_x = self.norm2(x)
        mlp_out = self.mlp(norm_x)
        x = x + self.dropout(mlp_out)
        
        return x
    
    def skip_attn_forward(self, x):
        norm_x = self.norm2(x)
        mlp_out = self.mlp(norm_x)
        x = x + self.dropout(mlp_out)
        return x
    
    def skip_mlp_forward(self, x):
        norm_x = self.norm1(x)
        attn_out = self.attn(norm_x)
        x = x + self.dropout(attn_out)
        return x
    
    def skip_forward(self, x):
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        self.attn_dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        
        # Compute attention
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)


class MLP(nn.Module):
    """MLP block with GELU activation and dropout"""
    
    def __init__(self, dim, mlp_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class CompactMedicalViT(nn.Module):
    """
    Compact version of Medical ViT for smaller datasets
    Reduced parameters while maintaining performance
    """
    
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        num_classes=3,
        dim=384,
        depth=6,
        heads=6,
        mlp_dim=1536,
        dropout=0.1,
        emb_dropout=0.1
    ):
        super().__init__()
        
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        self.transformer = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)
        ])
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        
        for transformer in self.transformer:
            x = transformer(x)
        
        x = x[:, 0]  # Use cls token
        return self.mlp_head(x)


# Model factory functions
def medical_vit_small(num_classes=3):
    """Small Medical ViT for quick experiments"""
    return CompactMedicalViT(
        image_size=224,
        patch_size=16,
        num_classes=num_classes,
        dim=384,
        depth=6,
        heads=6,
        mlp_dim=1536,
        dropout=0.1
    )


def medical_vit_base(num_classes=3):
    """Base Medical ViT model"""
    return MedicalVisionTransformer(
        image_size=224,
        patch_size=16,
        num_classes=num_classes,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0.1
    )


def medical_vit_large(num_classes=3):
    """Large Medical ViT for high performance"""
    return MedicalVisionTransformer(
        image_size=224,
        patch_size=16,
        num_classes=num_classes,
        dim=1024,
        depth=24,
        heads=16,
        mlp_dim=4096,
        dropout=0.1
    )


# Default model
def medical_vit(num_classes=3):
    return medical_vit_base(num_classes)
