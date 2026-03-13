import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, batch_first=True, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, query, key, value, key_padding_mask=None):
        
        if key_padding_mask is not None:
            # 检查是否有样本是全 Mask 的 
            all_masked = key_padding_mask.all(dim=1)
            if all_masked.any():
                key_padding_mask = key_padding_mask.clone()
                # 让全 Mask 样本的第一个 token 可见 
                key_padding_mask[all_masked, 0] = False

        attn_output, _ = self.multihead_attn(
            query, key, value, key_padding_mask=key_padding_mask
        )

        if key_padding_mask is not None and all_masked.any():
            attn_output[all_masked] = 0.0

        # 残差连接
        query = self.norm1(query + self.dropout(attn_output))
        ffn_output = self.ffn(query)
        query = self.norm2(query + self.dropout(ffn_output))

        return query


class DualWindowAttModel(nn.Module):
    def __init__(
        self,
        visual_dim=1024,
        audio_dim=1024,
        d_model=256,
        nhead=8,
        num_layers=3,
        num_classes=8,
    ):
        super().__init__()
        self._keys_to_ignore_on_save = None
        self.d_model = d_model

        self.v_proj = nn.Linear(visual_dim, d_model)
        self.a_proj = nn.Linear(audio_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=512)

        # Encoder
        visual_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=d_model * 4
        )
        self.visual_encoder = nn.TransformerEncoder(
            visual_encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        audio_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=d_model * 4
        )
        self.audio_encoder = nn.TransformerEncoder(
            audio_encoder_layer, num_layers=num_layers
        )

        #Safe Attention
        self.cross_attn_v2a = CrossAttentionBlock(d_model, nhead)
        self.cross_attn_a2v = CrossAttentionBlock(d_model, nhead)

        #Gate
        self.gate_v = nn.Linear(d_model * 2, 1)
        self.gate_a = nn.Linear(d_model * 2, 1)

        #Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, v_in, a_in, v_mask, labels=None):
        
        padding_mask = v_mask == 0  
        v_feat = self.pos_encoder(self.v_proj(v_in))
        a_feat = self.pos_encoder(self.a_proj(a_in))

        # Self-Attention 
        v_context = self.visual_encoder(v_feat, src_key_padding_mask=padding_mask)
        a_context = self.audio_encoder(a_feat)  

        # Cross-Attention
        # V 查询 A (A 是完整的，不需要 mask)
        v2a_fused = self.cross_attn_v2a(query=v_context, key=a_context, value=a_context)

        # A 查询 V 
        a2v_fused = self.cross_attn_a2v(
            query=a_context,
            key=v_context,
            value=v_context,
            key_padding_mask=padding_mask,  
        )

        # Gating Fusion
        # V 分支
        concat_v = torch.cat([v_context, v2a_fused], dim=-1)
        gate_v_weight = torch.sigmoid(self.gate_v(concat_v))
        final_v = gate_v_weight * v_context + (1 - gate_v_weight) * v2a_fused

        # A 分支
        concat_a = torch.cat([a_context, a2v_fused], dim=-1)
        gate_a_weight = torch.sigmoid(self.gate_a(concat_a))
        final_a = gate_a_weight * a_context + (1 - gate_a_weight) * a2v_fused

       
        # 强制将缺失位置的 visual 特征置为 0，防止干扰分类器。
        mask_expanded = (v_mask > 0).float().unsqueeze(-1)
        final_v = final_v * mask_expanded

       
        final_multimodal_feat = torch.cat([final_v, final_a], dim=-1)

        logits = self.classifier(final_multimodal_feat)

        return logits
