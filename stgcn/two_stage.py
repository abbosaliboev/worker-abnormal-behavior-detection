"""
Two-Stage Fall Detector.

Stage 1 — ST-GCN  : skeleton-sequence classifier  (fall probability)
Stage 2 — Physics : hip kinematics rule filter

Decision logic (Physics Rescue mode):
  prob >= stage1_threshold         → FALL  (Stage 1 confident)
  rescue_threshold <= prob < t1    → FALL if physics confirms  (uncertain zone, physics rescues)
  prob < rescue_threshold          → NO-FALL

This way physics can only ADD detections (rescue Stage-1 misses), never remove them.
Falls already caught by Stage 1 are never filtered out.
"""

import numpy as np
import torch
from .physics import PhysicsFilter


class TwoStageDetector:
    """
    Args:
        model              : trained STGCN (torch.nn.Module), eval mode expected
        physics            : fitted PhysicsFilter
        stage1_threshold   : high-confidence fall threshold (default 0.5)
        rescue_threshold   : lower threshold; in [rescue, stage1) physics decides (default 0.3)
        device             : 'cpu' or 'cuda'
    """

    def __init__(
        self,
        model: torch.nn.Module,
        physics: PhysicsFilter,
        stage1_threshold: float = 0.5,
        rescue_threshold: float = 0.3,
        device: str = "cpu",
    ):
        self.model              = model
        self.physics            = physics
        self.stage1_threshold   = stage1_threshold
        self.rescue_threshold   = rescue_threshold
        self.device             = device

    def set_stage1_threshold(self, t: float):
        self.stage1_threshold = t

    # ── inference helpers ─────────────────────────────────────────────────────

    @torch.no_grad()
    def _stage1_probs(self, x: torch.Tensor) -> np.ndarray:
        self.model.eval()
        logits = self.model(x)
        probs  = torch.softmax(logits, dim=-1)[:, 1]
        return probs.cpu().numpy()

    def predict_batch(
        self,
        X_tensor: torch.Tensor,
        X_numpy:  np.ndarray,
    ) -> np.ndarray:
        """
        Args:
            X_tensor : (N, C, T, V, M) — ST-GCN input
            X_numpy  : (N, T, V, C)    — raw keypoints for physics filter
        Returns:
            preds (N,) — 0 or 1
        """
        probs = self._stage1_probs(X_tensor.to(self.device))

        final = np.zeros(len(probs), dtype=int)
        for i, p in enumerate(probs):
            if p >= self.stage1_threshold:
                # Stage 1 confident → FALL (don't filter with physics)
                final[i] = 1
            elif p >= self.rescue_threshold:
                # Uncertain zone → physics decides
                final[i] = self.physics.predict(X_numpy[i])
            # else: Stage 1 confident NO-FALL → stays 0
        return final

    def predict_one(
        self,
        x_tensor: torch.Tensor,
        seq_numpy: np.ndarray,
    ) -> int:
        x = x_tensor.unsqueeze(0).to(self.device)
        preds = self.predict_batch(x, seq_numpy[np.newaxis])
        return int(preds[0])

    # ── threshold search ──────────────────────────────────────────────────────

    def tune_thresholds(
        self,
        X_tensor: torch.Tensor,
        X_numpy:  np.ndarray,
        y:        np.ndarray,
    ) -> tuple:
        """
        Grid-search over (stage1_threshold, rescue_threshold) to maximise fall-F1.
        Physics thresholds must already be set.
        Returns (best_stage1_t, best_rescue_t, best_f1).
        """
        from sklearn.metrics import f1_score

        probs = self._stage1_probs(X_tensor.to(self.device))
        physics_preds = np.array([self.physics.predict(X_numpy[i]) for i in range(len(X_numpy))])

        best_f1, best_t1, best_tr = -1.0, 0.5, 0.3

        for t1 in np.arange(0.3, 0.95, 0.05):
            for tr in np.arange(0.05, t1, 0.05):
                preds = np.zeros(len(probs), dtype=int)
                for i, p in enumerate(probs):
                    if p >= t1:
                        preds[i] = 1
                    elif p >= tr:
                        preds[i] = physics_preds[i]
                f1 = f1_score(y, preds, pos_label=1, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t1 = t1
                    best_tr = tr

        self.stage1_threshold = float(best_t1)
        self.rescue_threshold = float(best_tr)
        print(
            f"[TwoStage] stage1_threshold={self.stage1_threshold:.2f}  "
            f"rescue_threshold={self.rescue_threshold:.2f}  "
            f"val_fall_F1={best_f1:.4f}"
        )
        return best_t1, best_tr, best_f1

    # backward compat
    def tune_stage1_threshold(self, X_tensor, X_numpy, y):
        t1, _, f1 = self.tune_thresholds(X_tensor, X_numpy, y)
        return t1
