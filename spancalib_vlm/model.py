"""
SpanCalib-VLM Model Module
===========================
Defines SpanCalibVLM architecture with token-level classification head,
continuous probability calibrator, and specialized loss functions:
  - Focal Loss (token imbalance)
  - Soft-Dice Loss (Differentiable IoU)
  - Pearson Correlation Loss (Calibration metric optimization)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


class FocalLoss(nn.Module):
    """Focal Loss for addressing severe token-level class imbalance."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        logits: [B, N]
        targets: [B, N] binary float (0.0 or 1.0)
        mask: [B, N] boolean mask of valid response tokens
        """
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_factor = (1 - p_t) ** self.gamma
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_factor * focal_factor * bce
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss


class SoftDiceIoULoss(nn.Module):
    """Differentiable Soft-Dice Loss surrogate for Intersection-over-Union."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits) * mask
        targets = targets * mask
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum() - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1.0 - iou


class PearsonCorrelationLoss(nn.Module):
    """Differentiable Pearson Correlation Loss for direct calibration optimization."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred_probs: torch.Tensor, target_probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Extract masked tokens
        mask_bool = mask.bool()
        p = pred_probs[mask_bool]
        t = target_probs[mask_bool]

        if p.numel() < 2:
            return torch.tensor(0.0, device=pred_probs.device, requires_grad=True)

        p_mean = p.mean()
        t_mean = t.mean()

        p_centered = p - p_mean
        t_centered = t - t_mean

        cov = (p_centered * t_centered).sum()
        p_std = torch.sqrt((p_centered ** 2).sum() + self.eps)
        t_std = torch.sqrt((t_centered ** 2).sum() + self.eps)

        corr = cov / (p_std * t_std + self.eps)
        return 1.0 - corr


class SpanCalibHead(nn.Module):
    """Token-level classification & calibration head."""

    def __init__(self, hidden_size: int, num_categories: int = 5):
        super().__init__()
        # 1. Binary span detector (is token hallucinated?)
        self.span_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
        )

        # 2. Continuous probability calibrator
        self.prob_calibrator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

        # 3. Category classifier
        self.category_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_categories),
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        span_logits = self.span_classifier(hidden_states).squeeze(-1)       # [B, N]
        pred_probs = self.prob_calibrator(hidden_states).squeeze(-1)         # [B, N]
        cat_logits = self.category_classifier(hidden_states)                # [B, N, 5]
        return span_logits, pred_probs, cat_logits


class CrossAttentionVisionFusion(nn.Module):
    """Cross-Attention fusion between text token representations and visual patch tokens."""

    def __init__(self, text_hidden_size: int, vision_hidden_size: int = 768, num_heads: int = 8):
        super().__init__()
        self.proj_v = nn.Linear(vision_hidden_size, text_hidden_size) if vision_hidden_size != text_hidden_size else nn.Identity()
        self.cross_attn = nn.MultiheadAttention(embed_dim=text_hidden_size, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(text_hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, text_states: torch.Tensor, vision_states: torch.Tensor) -> torch.Tensor:
        v_proj = self.proj_v(vision_states)
        attn_out, _ = self.cross_attn(query=text_states, key=v_proj, value=v_proj)
        fused = self.norm(text_states + self.dropout(attn_out))
        return fused


class SpanCalibVLM(nn.Module):
    """SpanCalib-VLM: Full Architecture with optional Cross-Attention Vision Fusion."""

    def __init__(self, model_id: str = "xlm-roberta-base", num_categories: int = 5, use_vision: bool = False):
        super().__init__()
        self.model_id = model_id
        self.use_vision = use_vision
        self.config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(model_id, trust_remote_code=True)

        hidden_size = getattr(self.config, "hidden_size", getattr(self.config, "d_model", 768))

        if use_vision:
            from transformers import SiglipVisionModel
            self.vision_encoder = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
            v_hidden_size = self.vision_encoder.config.hidden_size
            self.vision_fusion = CrossAttentionVisionFusion(text_hidden_size=hidden_size, vision_hidden_size=v_hidden_size)
        else:
            self.vision_encoder = None
            self.vision_fusion = None

        self.head = SpanCalibHead(hidden_size=hidden_size, num_categories=num_categories)

        # Loss functions
        self.focal_loss = FocalLoss(alpha=0.75, gamma=2.0)
        self.dice_loss = SoftDiceIoULoss()
        self.pearson_loss = PearsonCorrelationLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        response_token_mask: torch.Tensor | None = None,
        binary_labels: torch.Tensor | None = None,
        prob_labels: torch.Tensor | None = None,
        category_labels: torch.Tensor | None = None,
    ) -> dict:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        text_hidden_states = outputs.last_hidden_state  # [B, N, hidden_size]

        if self.use_vision and pixel_values is not None and self.vision_encoder is not None:
            vision_outputs = self.vision_encoder(pixel_values=pixel_values)
            vision_hidden_states = vision_outputs.last_hidden_state  # [B, num_patches, v_hidden]
            hidden_states = self.vision_fusion(text_hidden_states, vision_hidden_states)
        else:
            hidden_states = text_hidden_states

        span_logits, pred_probs, cat_logits = self.head(hidden_states)

        loss = None
        if binary_labels is not None and response_token_mask is not None:
            mask = response_token_mask.float()

            # 1. Span Detection Loss = Focal Loss + Soft-Dice IoU Loss
            l_focal = self.focal_loss(span_logits, binary_labels, mask)
            l_dice = self.dice_loss(span_logits, binary_labels, mask)

            # 2. Calibration Loss = Pearson Correlation Loss + MSE
            l_pearson = self.pearson_loss(pred_probs, prob_labels, mask)
            l_mse = (F.mse_loss(pred_probs, prob_labels, reduction="none") * mask).sum() / (mask.sum() + 1e-8)

            # 3. Category Classification Loss
            l_cat = F.cross_entropy(
                cat_logits.view(-1, 5),
                category_labels.view(-1),
                reduction="none",
            ).view_as(binary_labels)
            l_cat = (l_cat * mask).sum() / (mask.sum() + 1e-8)

            loss = l_focal + 0.5 * l_dice + 0.5 * l_pearson + 0.2 * l_mse + 0.3 * l_cat

        return {
            "loss": loss,
            "span_logits": span_logits,
            "pred_probs": pred_probs,
            "cat_logits": cat_logits,
        }
