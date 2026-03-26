# src/encoder/mpose_encoder.py
import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, List, Optional, Tuple
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


from src.features.joint_features import JointFeatureConfig

logger = logging.getLogger(__name__)

# Copy the Normalization class from the original (unchanged)
class Normalization:
    """Robust keypoint normalization with fallback and clipping."""

    @staticmethod
    def torso_centric(keypoints: torch.Tensor,
                      min_torso_height: float = 10.0,
                      max_norm_range: float = 5.0,
                      fallback_to_bbox: bool = True) -> torch.Tensor:
        device = keypoints.device
        batch_size = keypoints.shape[0]
        left_shoulder, right_shoulder = 5, 6
        left_hip, right_hip = 11, 12

        positions = keypoints[..., :2]      # (batch, num_joints, 2)
        confidence = keypoints[..., 2:3]    # (batch, num_joints, 1)

        # Confidence for torso joints
        shoulders_conf = confidence[:, [left_shoulder, right_shoulder], 0]
        hips_conf = confidence[:, [left_hip, right_hip], 0]

        valid_shoulders = (shoulders_conf > 0).sum(dim=1)
        valid_hips = (hips_conf > 0).sum(dim=1)

        use_torso = (valid_shoulders >= 1) & (valid_hips >= 1)

        if not use_torso.any() and fallback_to_bbox:
            return Normalization.bbox_centric(keypoints, max_norm_range=max_norm_range)

        # Compute midpoints
        shoulder_mid = torch.zeros(batch_size, 2, device=device)
        hip_mid = torch.zeros(batch_size, 2, device=device)

        for b in range(batch_size):
            if use_torso[b]:
                # Shoulder midpoint
                s_left = positions[b, left_shoulder, :]
                s_right = positions[b, right_shoulder, :]
                s_conf_left = shoulders_conf[b, 0]
                s_conf_right = shoulders_conf[b, 1]
                if s_conf_left > 0 and s_conf_right > 0:
                    shoulder_mid[b] = (s_left + s_right) / 2
                elif s_conf_left > 0:
                    shoulder_mid[b] = s_left
                else:
                    shoulder_mid[b] = s_right

                # Hip midpoint
                h_left = positions[b, left_hip, :]
                h_right = positions[b, right_hip, :]
                h_conf_left = hips_conf[b, 0]
                h_conf_right = hips_conf[b, 1]
                if h_conf_left > 0 and h_conf_right > 0:
                    hip_mid[b] = (h_left + h_right) / 2
                elif h_conf_left > 0:
                    hip_mid[b] = h_left
                else:
                    hip_mid[b] = h_right

        torso_center = (shoulder_mid + hip_mid) / 2        # (batch, 2)
        torso_height = torch.norm(shoulder_mid - hip_mid, dim=1, keepdim=True)  # (batch, 1)

        # Check which samples have torso height below threshold
        small_height = torso_height.squeeze() < min_torso_height

        if small_height.any() and fallback_to_bbox:
            # Mixed case: some samples are valid, others need fallback
            normalized = torch.zeros_like(keypoints)
            valid_mask = ~small_height & use_torso
            if valid_mask.any():
                pos = positions[valid_mask]                         # (n_valid, num_joints, 2)
                center = torso_center[valid_mask].unsqueeze(1)      # (n_valid, 1, 2)
                height = torso_height[valid_mask].unsqueeze(1)      # (n_valid, 1, 1)
                norm_pos = (pos - center) / (height + 1e-6)
                norm_pos = torch.clamp(norm_pos, -max_norm_range, max_norm_range)
                normalized[valid_mask, :, :2] = norm_pos
                normalized[valid_mask, :, 2:3] = confidence[valid_mask]
            fallback_mask = ~valid_mask
            if fallback_mask.any():
                fallback_norm = Normalization.bbox_centric(keypoints[fallback_mask], max_norm_range=max_norm_range)
                normalized[fallback_mask] = fallback_norm
            return normalized
        else:
            # All samples use torso (or we don't fall back)
            center = torso_center.unsqueeze(1)          # (batch, 1, 2)
            height = torso_height.unsqueeze(1)          # (batch, 1, 1)
            norm_pos = (positions - center) / (height + 1e-6)
            norm_pos = torch.clamp(norm_pos, -max_norm_range, max_norm_range)
            return torch.cat([norm_pos, confidence], dim=-1)

    @staticmethod
    def bbox_centric(keypoints: torch.Tensor, bbox: Optional[torch.Tensor] = None,
                     max_norm_range: float = 5.0) -> torch.Tensor:
        """
        Normalize using bounding box (x_min, y_min, width, height).
        If bbox not provided, compute from keypoints.
        """
        if bbox is None:
            # Compute bbox from keypoints: (batch, 1, 2) for min and max
            min_xy = keypoints[..., :2].min(dim=1, keepdim=True)[0]  # (batch, 1, 2)
            max_xy = keypoints[..., :2].max(dim=1, keepdim=True)[0]  # (batch, 1, 2)
            width = (max_xy[..., 0] - min_xy[..., 0]).unsqueeze(-1)   # (batch, 1, 1)
            height = (max_xy[..., 1] - min_xy[..., 1]).unsqueeze(-1)  # (batch, 1, 1)
            denom = torch.cat([width+ 1e-6, height+ 1e-6], dim=-1)  # (batch, 1, 2)
        else:
            # bbox shape: (batch, 4) or (batch, 1, 4)
            if bbox.dim() == 2:
                bbox = bbox.unsqueeze(1)  # (batch, 1, 4)
            min_xy = bbox[..., :2]        # (batch, 1, 2)
            width = bbox[..., 2:3]        # (batch, 1, 1)
            height = bbox[..., 3:4]       # (batch, 1, 1)
            denom = torch.cat([width + + 1e-6, height+ + 1e-6], dim=-1)  # (batch, 1, 2)

        normalized_pos = (keypoints[..., :2] - min_xy) / denom
        normalized_pos = torch.clamp(normalized_pos, -max_norm_range, max_norm_range)
        return torch.cat([normalized_pos, keypoints[..., 2:3]], dim=-1)


class MPOSEFeatureExtractor(nn.Module):
    """
    Adapted JointFeatureExtractor for MPOSE2021 dataset (5 channels: x,y,conf,dx,dy).
    Uses pre‑computed linear velocities from the dataset.
    """
    def __init__(
        self,
        num_joints: int = 17,
        joint_type_embedding_dim: int = 16,
        max_angles: int = 3,
        use_angles: bool = True,
        use_velocities: bool = True,      # uses pre‑computed dx,dy
        use_relative_pos: bool = True,
        use_confidence: bool = True,
        use_temporal: bool = False,
        temporal_encoding_dim: int = 8,
        output_dim: int = 64,
        normalize: str = 'torso',
        augmentations: Optional[Dict] = None,
        return_raw: bool = False,
        min_torso_height: float = 10.0,
        max_norm_range: float = 5.0,
        fallback_to_bbox: bool = True,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.output_dim = output_dim
        self.max_angles = max_angles
        self.use_angles = use_angles
        self.use_velocities = use_velocities
        self.use_relative_pos = use_relative_pos
        self.use_confidence = use_confidence
        self.use_temporal = use_temporal
        self.normalize = normalize
        self.augmentations = augmentations or {}
        self.return_raw = return_raw
        self.min_torso_height = min_torso_height
        self.max_norm_range = max_norm_range
        self.fallback_to_bbox = fallback_to_bbox

        # Joint type embedding
        self.joint_type_embedding = nn.Embedding(num_joints, joint_type_embedding_dim)

        # Angle mask
        angle_mask = torch.zeros(num_joints, max_angles)
        for j in range(num_joints):
            config = JointFeatureConfig.JOINT_ANGLE_CONFIG.get(j, {'angles': [False, False, False]})
            for a, relevant in enumerate(config['angles']):
                if relevant and a < max_angles:
                    angle_mask[j, a] = 1.0
        self.register_buffer('angle_mask', angle_mask)

        # Position encoder (for normalized x,y)
        self.pos_encoder = nn.Linear(2, 16)

        # Angle encoder
        if use_angles:
            self.angle_encoder = nn.Linear(max_angles, 16)

        # Velocity encoders
        if use_velocities:
            self.velocity_encoder = nn.Linear(2, 8)          # linear velocity (dx,dy)
            # Angular velocity encoder – we won't use it for now (would require previous angles)
            self.angular_velocity_encoder = nn.Linear(max_angles, 8)

        # Relative position encoder
        if use_relative_pos:
            self.relative_pos_encoder = nn.Linear(2, 8)

        # Confidence encoder
        if use_confidence:
            self.confidence_encoder = nn.Linear(1, 4)

        # Temporal encoder (unused)
        if use_temporal:
            self.temporal_encoder = nn.Linear(temporal_encoding_dim, 8)

        # Total feature dimension
        total_dim = joint_type_embedding_dim + 16   # type + position
        if use_angles:
            total_dim += 16
        if use_velocities:
            total_dim += 8   # linear velocity
            # We'll also add angular velocity dimension if we compute it later
            # For simplicity, we'll add zero for angular velocity and its encoder
            total_dim += 8   # angular velocity (placeholder)
        if use_relative_pos:
            total_dim += 8
        if use_confidence:
            total_dim += 4
        if use_temporal:
            total_dim += 8

        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(total_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )

        logger.info(f"MPOSEFeatureExtractor initialized:")
        logger.info(f"  - Normalization: {normalize}")
        logger.info(f"  - Output dim: {output_dim}")
        logger.info(f"  - Use angles: {use_angles}, velocities: {use_velocities}")
        logger.info(f"  - Normalization params: min_height={min_torso_height}, max_range={max_norm_range}, fallback={fallback_to_bbox}")

    def forward(self, x: torch.Tensor, apply_augmentations: bool = False, return_raw: Optional[bool] = None):
        """
        x: (batch, T, J, C) where C = 5 (x,y,conf,dx,dy) or (x,y,conf) if velocities not used.
        We'll process each frame independently and return (batch, T, J, D) embeddings.
        """
        if return_raw is None:
            return_raw = self.return_raw

        batch, T, J, C = x.shape
        # Reshape to (batch*T, J, C)
        x_flat = x.view(batch * T, J, C)

        # Separate components
        positions = x_flat[..., :2]          # (B*T, J, 2)
        confidence = x_flat[..., 2:3]        # (B*T, J, 1)
        if C >= 5:
            linear_vel = x_flat[..., 3:5]    # (B*T, J, 2)
        else:
            linear_vel = None

        # Augmentations (simplified – not implemented for velocities)
        if apply_augmentations and self.training:
            # We'll skip augmentations for now to keep code simple.
            pass

        # Normalization of positions and confidence
        if self.normalize == 'torso':
            kp_for_norm = torch.cat([positions, confidence], dim=-1)
            normalized = Normalization.torso_centric(
                kp_for_norm,
                min_torso_height=self.min_torso_height,
                max_norm_range=self.max_norm_range,
                fallback_to_bbox=self.fallback_to_bbox
            )
            positions_norm = normalized[..., :2]
            confidence_norm = normalized[..., 2:3]
        elif self.normalize == 'bbox':
            kp_for_norm = torch.cat([positions, confidence], dim=-1)
            normalized = Normalization.bbox_centric(kp_for_norm, max_norm_range=self.max_norm_range)
            positions_norm = normalized[..., :2]
            confidence_norm = normalized[..., 2:3]
        else:
            positions_norm = positions
            confidence_norm = confidence

        # Compute angles (if needed)
        if self.use_angles:
            kp_for_angles = torch.cat([positions_norm, confidence_norm], dim=-1)
            angles = self._compute_angles(kp_for_angles)
            angles = angles * self.angle_mask.unsqueeze(0)
            angle_emb = self.angle_encoder(angles)
        else:
            angles = None
            angle_emb = torch.zeros(batch*T, J, 0, device=x.device)

        # Linear velocities (pre‑computed)
        if self.use_velocities and linear_vel is not None:
            # Note: velocities are in original image coordinates; they should be normalized consistently.
            # For simplicity, we'll pass them as‑is; the network can learn to adapt.
            velocity_emb = self.velocity_encoder(linear_vel)
        else:
            velocity_emb = torch.zeros(batch*T, J, 8, device=x.device)

        # Angular velocity – not computed, set to zero
        angular_velocity_emb = torch.zeros_like(velocity_emb)

        # Relative position to torso
        if self.use_relative_pos:
            torso_idx = 5
            torso_pos = positions_norm[:, torso_idx:torso_idx+1, :]
            relative_pos = positions_norm - torso_pos
            relative_pos_emb = self.relative_pos_encoder(relative_pos)
        else:
            relative_pos_emb = None

        # Confidence encoding
        if self.use_confidence:
            confidence_emb = self.confidence_encoder(confidence_norm)
        else:
            confidence_emb = None

        # Temporal encoding (skip)
        temporal_emb = None

        # Joint type embeddings
        joint_indices = torch.arange(self.num_joints, device=x.device)
        type_emb = self.joint_type_embedding(joint_indices)
        type_emb = type_emb.unsqueeze(0).expand(batch*T, -1, -1)

        # Position encoding
        pos_emb = self.pos_encoder(positions_norm)

        # Concatenate features
        features = [type_emb, pos_emb]
        if self.use_angles:
            features.append(angle_emb)
        if self.use_velocities:
            features.append(velocity_emb)
            features.append(angular_velocity_emb)
        if self.use_relative_pos and relative_pos_emb is not None:
            features.append(relative_pos_emb)
        if self.use_confidence and confidence_emb is not None:
            features.append(confidence_emb)

        combined = torch.cat(features, dim=-1)

        # Project to final embedding
        joint_embeddings = self.projection(combined)
        joint_embeddings = torch.nan_to_num(joint_embeddings, nan=0.0, posinf=0.0, neginf=0.0)

        # Reshape back to (batch, T, J, D)
        joint_embeddings = joint_embeddings.view(batch, T, J, -1)

        output = {'joint_embeddings': joint_embeddings}

        if return_raw:
            output['raw_features'] = {
                'positions_norm': positions_norm,
                'angles': angles,
                'linear_vel': linear_vel,
                'confidence': confidence_norm,
            }
        return output

    def _compute_angles(self, keypoints: torch.Tensor) -> torch.Tensor:
        """
        Compute biomechanical angles for each joint in a fully vectorized way.
        Args:
            keypoints: (batch, num_joints, 3) with x, y, confidence
        Returns:
            angles: (batch, num_joints, max_angles) with zeros for undefined joints
            """
        batch_size, num_joints, _ = keypoints.shape
        device = keypoints.device
        angles = torch.zeros(batch_size, num_joints, self.max_angles, device=device)

    # Extract positions and confidence
        pos = keypoints[..., :2]            # (batch, num_joints, 2)
        conf = keypoints[..., 2]            # (batch, num_joints)

    # Precompute torso center (midpoint of shoulders and hips)
        left_shoulder, right_shoulder = 5, 6
        left_hip, right_hip = 11, 12

    # Shoulder and hip midpoints
        shoulder_mid = (pos[:, left_shoulder] + pos[:, right_shoulder]) / 2   # (batch, 2)
        hip_mid = (pos[:, left_hip] + pos[:, right_hip]) / 2                 # (batch, 2)
        torso_center = (shoulder_mid + hip_mid) / 2                          # (batch, 2)

    # Confidence for torso joints (used to mask angles if low)
        torso_conf = (conf[:, left_shoulder] + conf[:, right_shoulder] +
                      conf[:, left_hip] + conf[:, right_hip]) / 4            # (batch,)
        valid_torso = torso_conf > 0.5                                      # (batch,)

    # Helper to compute angle between two vectors
        def angle_between(v1, v2, eps=1e-8):
            dot = (v1 * v2).sum(dim=-1)
            norms = v1.norm(dim=-1) * v2.norm(dim=-1) + eps
            cos = torch.clamp(dot / norms, -1.0, 1.0)
            return torch.acos(cos)

    # -------------------- Shoulders (indices 5, 6) --------------------
        shoulder_indices = [5, 6]
        elbow_indices = [7, 8]          # left elbow, right elbow
        for s_idx, e_idx in zip(shoulder_indices, elbow_indices):
        # Get positions and confidences
            sh_pos = pos[:, s_idx]       # (batch, 2)
            el_pos = pos[:, e_idx]       # (batch, 2)
            sh_conf = conf[:, s_idx]
            el_conf = conf[:, e_idx]
        # Arm vector and reference (torso center - shoulder)
            arm_vec = el_pos - sh_pos
            ref_vec = torso_center - sh_pos
        # Compute angles only if both joints have sufficient confidence
            valid = (sh_conf > 0.5) & (el_conf > 0.5) & valid_torso
        # Flexion: angle between arm and reference
            flexion = torch.where(valid, angle_between(arm_vec, ref_vec), torch.tensor(0.0, device=device))
        # Abduction: angle of arm in XY plane (simplified)
            abduction = torch.where(valid, torch.atan2(arm_vec[:, 1], arm_vec[:, 0]), torch.tensor(0.0, device=device))
        # Rotation placeholder
            rotation = torch.zeros_like(flexion)
        # Store
            angles[:, s_idx, 0] = flexion
            angles[:, s_idx, 1] = abduction
            angles[:, s_idx, 2] = rotation

    # -------------------- Elbows (indices 7, 8) --------------------
        shoulder_elbow = [(5,7), (6,8)]   # (shoulder_idx, elbow_idx)
        wrist_indices = [9, 10]           # left wrist, right wrist
        for (sh_idx, el_idx), wr_idx in zip(shoulder_elbow, wrist_indices):
            sh_pos = pos[:, sh_idx]
            el_pos = pos[:, el_idx]
            wr_pos = pos[:, wr_idx]
            sh_conf = conf[:, sh_idx]
            el_conf = conf[:, el_idx]
            wr_conf = conf[:, wr_idx]
            valid = (sh_conf > 0.5) & (el_conf > 0.5) & (wr_conf > 0.5)
            v1 = sh_pos - el_pos      # upper arm
            v2 = wr_pos - el_pos      # forearm
            angle = torch.where(valid, angle_between(v1, v2), torch.tensor(0.0, device=device))
            angles[:, el_idx, 0] = angle

    # -------------------- Hips (indices 11, 12) --------------------
        hip_indices = [11, 12]
        knee_indices = [13, 14]       # left knee, right knee
        for h_idx, k_idx in zip(hip_indices, knee_indices):
            hip_pos = pos[:, h_idx]
            knee_pos = pos[:, k_idx]
            hip_conf = conf[:, h_idx]
            knee_conf = conf[:, k_idx]
            leg_vec = knee_pos - hip_pos
            ref_vec = torso_center - hip_pos
            valid = (hip_conf > 0.5) & (knee_conf > 0.5) & valid_torso
            flexion = torch.where(valid, angle_between(leg_vec, ref_vec), torch.tensor(0.0, device=device))
            abduction = torch.where(valid, torch.atan2(leg_vec[:, 1], leg_vec[:, 0]), torch.tensor(0.0, device=device))
            rotation = torch.zeros_like(flexion)
            angles[:, h_idx, 0] = flexion
            angles[:, h_idx, 1] = abduction
            angles[:, h_idx, 2] = rotation

    # -------------------- Knees (indices 13, 14) --------------------
        hip_knee = [(11,13), (12,14)]   # (hip_idx, knee_idx)
        ankle_indices = [15, 16]        # left ankle, right ankle
        for (h_idx, k_idx), a_idx in zip(hip_knee, ankle_indices):
            hip_pos = pos[:, h_idx]
            knee_pos = pos[:, k_idx]
            ankle_pos = pos[:, a_idx]
            hip_conf = conf[:, h_idx]
            knee_conf = conf[:, k_idx]
            ankle_conf = conf[:, a_idx]
            valid = (hip_conf > 0.5) & (knee_conf > 0.5) & (ankle_conf > 0.5)
            v1 = hip_pos - knee_pos   # thigh
            v2 = ankle_pos - knee_pos # shank
            angle = torch.where(valid, angle_between(v1, v2), torch.tensor(0.0, device=device))
            angles[:, k_idx, 0] = angle

        return angles