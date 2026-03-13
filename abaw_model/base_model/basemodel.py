import torch
import torch.nn as nn
from transformers.modeling_outputs import ImageClassifierOutput

class BaseModel(nn.Module):
    def __init__(
        self,
        visual_input_dim: int = 1024,  
        audio_input_dim: int = 1024,  
        hidden_dim: int = 1024,
        num_classes: int = 8,  
        lambda_val: float = 0.5, 
        dropout_p: float = 0.24,  
    ):
        super().__init__()
        self.lambda_val = lambda_val

        self.visual_mlp = nn.Sequential(
            nn.Linear(visual_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_p),  
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.audio_mlp = nn.Sequential(
            nn.Linear(audio_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_p),  
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        visual_feat: torch.Tensor,  # [B, visual_input_dim]
        audio_feat: torch.Tensor,  # [B, audio_input_dim]
        labels = None,
    ) -> torch.Tensor:

        v_out = self.visual_mlp(visual_feat)  # [B, hidden_dim]
        a_out = self.audio_mlp(audio_feat)  # [B, hidden_dim]

        # output = λV + (1-λ)A
        combined = self.lambda_val * v_out + (1 - self.lambda_val) * a_out
        logits = self.classifier(combined)

        return ImageClassifierOutput(logits=logits)
