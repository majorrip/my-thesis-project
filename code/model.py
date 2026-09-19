import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_stable_local_loss(projected_student, teacher_target, layer_output,
                              variance_threshold=1.0, alpha=1.0, beta=0.01,
                              return_components=False):
    """
    Computes regularized local distillation loss using VicReg-style constraints
    to prevent dimensional collapse.

    The default coefficients (alpha=1.0, beta=0.01) reproduce the original
    formulation exactly. They are exposed as arguments so that the
    unregularized baseline (alpha=beta=0) required for the collapse ablation
    can share this single code path. Set return_components=True to obtain the
    per-term breakdown for logging.
    """
    # 1. Base Distillation Loss (MSE against target)
    distill_loss = F.mse_loss(projected_student, teacher_target)

    # 2. Variance Constraint (Hinge loss on batch standard deviation)
    std_student = torch.sqrt(layer_output.var(dim=0) + 1e-4)
    variance_loss = torch.mean(F.relu(variance_threshold - std_student))

    # 3. Covariance Regularization (Feature decorrelation)
    centered_student = projected_student - projected_student.mean(dim=0)
    batch_size = projected_student.size(0)
    cov_matrix = (centered_student.T @ centered_student) / (batch_size - 1)
    diag_mask = torch.eye(cov_matrix.size(0), device=projected_student.device)
    covariance_loss = (cov_matrix * (1 - diag_mask)).pow(2).sum() / cov_matrix.size(0)

    total = distill_loss + alpha * variance_loss + beta * covariance_loss
    if return_components:
        return total, {
            "distill": distill_loss.item(),
            "var": variance_loss.item(),
            "cov": covariance_loss.item(),
            "total": total.item(),
        }
    return total

class VisionJEPA(nn.Module):
    def __init__(self, img_channels=3, latent_dim=256, ema_decay=0.999):
        super().__init__()
        self.latent_dim = latent_dim
        self.ema_decay = ema_decay

        # Context Encoder (f_theta)
        self.context_encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # Target Encoder (f_theta_bar) - Updated via EMA
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Action/Bounding-Box Encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

        # Latent Predictor Block (p_psi)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

    @torch.no_grad()
    def update_target_encoder(self):
        for p_ctx, p_tgt in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            p_tgt.data.mul_(self.ema_decay).add_(p_ctx.data, alpha=1.0 - self.ema_decay)

    def forward(self, partial_images, full_images, spatial_actions):
        s_t = self.context_encoder(partial_images)
        with torch.no_grad():
            s_target = self.target_encoder(full_images)
        a_t = self.action_encoder(spatial_actions)

        combined_latent = torch.cat([s_t, a_t], dim=-1)
        s_predicted = self.predictor(combined_latent)

        return s_predicted, s_target, s_t
