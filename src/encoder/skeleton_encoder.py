

# src/joint_embedding/joint_feature_extractor.py
import torch
import sys
import os
import torch.nn as nn
import numpy as np
import math
from typing import Dict, List, Optional, Tuple
import logging
from src.features.joint_features import JointAngles, JointFeatureConfig

logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


# =============================================================================
# Normalization helpers (Robust version)

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


# =============================================================================
# Edge feature extractor for GNNs
# =============================================================================
class EdgeFeatureExtractor:
    @staticmethod
    def compute_edge_features(keypoints: torch.Tensor,
                              connections: List[Tuple[int, int]]) -> torch.Tensor:
        """
        Compute bone vectors for each edge.
        Args:
            keypoints: (batch, num_joints, 3) (x,y,confidence)
            connections: list of (i, j) joint indices (directed or undirected)
        Returns:
            edge_features: (batch, num_edges, 2) (dx, dy)
        """
        positions = keypoints[..., :2]  # (batch, num_joints, 2)
        edge_features = []
        for i, j in connections:
            vec = positions[:, i, :] - positions[:, j, :]   # (batch, 2)
            edge_features.append(vec.unsqueeze(1))          # (batch, 1, 2)
        return torch.cat(edge_features, dim=1)              # (batch, num_edges, 2)


# =============================================================================
# Main JointFeatureExtractor
# =============================================================================
class JointFeatureExtractor(nn.Module):
    """
    Enhanced feature extractor with:
    - Joint type embedding (learnable)
    - Normalization (torso / bbox)
    - Biomechanical angles (flexion, abduction, rotation)
    - Linear & angular velocities
    - Relative position to torso
    - Confidence encoding
    - Temporal encoding
    - Augmentations (rotation, scale, noise, masking)
    """

    def __init__(
        self,
        num_joints: int = 17,
        joint_type_embedding_dim: int = 16,
        max_angles: int = 3,
        use_angles: bool = False,
        use_velocities: bool = True,
        use_relative_pos: bool = True,
        use_confidence: bool = True,
        use_temporal: bool = False,
        temporal_encoding_dim: int = 8,
        output_dim: int = 64,
        normalize: str = 'torso',           # 'torso', 'bbox', or None
        augmentations: Optional[Dict] = None,
        return_raw: bool = False,
        # Normalization parameters
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
        # Store normalization parameters for use in forward
        self.min_torso_height = min_torso_height
        self.max_norm_range = max_norm_range
        self.fallback_to_bbox = fallback_to_bbox

        # --- Joint type embedding ---
        self.joint_type_embedding = nn.Embedding(num_joints, joint_type_embedding_dim)

        # --- Angle mask (which angles are relevant per joint) ---
        angle_mask = torch.zeros(num_joints, max_angles)
        for j in range(num_joints):
            config = JointFeatureConfig.JOINT_ANGLE_CONFIG.get(j, {'angles': [False, False, False]})
            for a, relevant in enumerate(config['angles']):
                if relevant and a < max_angles:
                    angle_mask[j, a] = 1.0
        self.register_buffer('angle_mask', angle_mask)

        # --- Position encoder ---
        self.pos_encoder = nn.Linear(2, 16)

        # --- Angle encoder ---
        if use_angles:
            self.angle_encoder = nn.Linear(max_angles, 16)

        # --- Velocity encoders ---
        if use_velocities:
            self.velocity_encoder = nn.Linear(2, 8)
            self.angular_velocity_encoder = nn.Linear(max_angles, 8)

        # --- Relative position encoder ---
        if use_relative_pos:
            self.relative_pos_encoder = nn.Linear(2, 8)

        # --- Confidence encoder ---
        if use_confidence:
            self.confidence_encoder = nn.Linear(1, 4)

        # --- Temporal encoder ---
        if use_temporal:
            self.temporal_encoder = nn.Linear(temporal_encoding_dim, 8)

        # --- Calculate total feature dimension ---
        total_dim = joint_type_embedding_dim + 16   # type + position
        if use_angles:
            total_dim += 16
        if use_velocities:
            total_dim += 8 + 8    # linear + angular
        if use_relative_pos:
            total_dim += 8
        if use_confidence:
            total_dim += 4
        if use_temporal:
            total_dim += 8

        # --- Final projection ---
        self.projection = nn.Sequential(
            nn.Linear(total_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )

        logger.info(f"JointFeatureExtractor initialized:")
        logger.info(f"  - Normalization: {normalize}")
        logger.info(f"  - Augmentations: {augmentations}")
        logger.info(f"  - Output dim: {output_dim}")
        logger.info(f"  - Use angles: {use_angles}, velocities: {use_velocities}")
        logger.info(f"  - Normalization params: min_height={min_torso_height}, max_range={max_norm_range}, fallback={fallback_to_bbox}")

    def forward(
        self,
        keypoints: torch.Tensor,
        prev_keypoints: Optional[torch.Tensor] = None,
        frame_idx: Optional[torch.Tensor] = None,
        bbox: Optional[torch.Tensor] = None,
        apply_augmentations: bool = False,
        return_raw: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        if return_raw is None:
            return_raw = self.return_raw

        # --- Step 0: Augmentations (if enabled) ---
        if apply_augmentations and self.training:
            keypoints, prev_keypoints = self._apply_augmentations(keypoints, prev_keypoints)

        # --- Step 1: Normalization (now with robust parameters) ---
        if self.normalize == 'torso':
            keypoints = Normalization.torso_centric(
                keypoints,
                min_torso_height=self.min_torso_height,
                max_norm_range=self.max_norm_range,
                fallback_to_bbox=self.fallback_to_bbox
            )
            if prev_keypoints is not None:
                prev_keypoints = Normalization.torso_centric(
                    prev_keypoints,
                    min_torso_height=self.min_torso_height,
                    max_norm_range=self.max_norm_range,
                    fallback_to_bbox=self.fallback_to_bbox
                )
        elif self.normalize == 'bbox':
            keypoints = Normalization.bbox_centric(keypoints, bbox, max_norm_range=self.max_norm_range)
            if prev_keypoints is not None:
                prev_keypoints = Normalization.bbox_centric(prev_keypoints, bbox, max_norm_range=self.max_norm_range)
        # else: no normalization

        # --- Extract components ---
        positions = keypoints[..., :2]
        confidence = keypoints[..., 2:3]

        # --- Step 2: Compute angles (if needed) ---
        if self.use_angles:
            angles = self._compute_angles(keypoints)
            angles = angles * self.angle_mask.unsqueeze(0)
            angle_emb = self.angle_encoder(angles)
        else:
            angles = None
            angle_emb = torch.zeros(keypoints.size(0), self.num_joints, 0, device=keypoints.device)

        # --- Step 3: Compute velocities (if previous frame available) ---
        if self.use_velocities and prev_keypoints is not None:
            prev_positions = prev_keypoints[..., :2]
            velocity = positions - prev_positions
            velocity_emb = self.velocity_encoder(velocity)

            if self.use_angles:
                prev_angles = self._compute_angles(prev_keypoints)
                prev_angles = prev_angles * self.angle_mask.unsqueeze(0)
                angular_velocity = angles - prev_angles
                angular_velocity_emb = self.angular_velocity_encoder(angular_velocity)
            else:
                angular_velocity = None
                angular_velocity_emb = torch.zeros_like(velocity_emb)
        else:
            velocity = None
            velocity_emb = torch.zeros(keypoints.size(0), self.num_joints, 8, device=keypoints.device)
            angular_velocity = None
            angular_velocity_emb = torch.zeros_like(velocity_emb)

        # --- Step 4: Relative position to torso ---
        if self.use_relative_pos:
            torso_idx = 5
            torso_pos = positions[:, torso_idx:torso_idx+1, :]
            relative_pos = positions - torso_pos
            relative_pos_emb = self.relative_pos_encoder(relative_pos)
        else:
            relative_pos_emb = None

        # --- Step 5: Confidence encoding ---
        if self.use_confidence:
            confidence_emb = self.confidence_encoder(confidence)
        else:
            confidence_emb = None

        # --- Step 6: Temporal encoding ---
        if self.use_temporal:
            if frame_idx is None:
                frame_idx = torch.zeros(keypoints.size(0), 1, device=keypoints.device)
            temporal_feat = self._temporal_encoding(frame_idx)
            temporal_emb = self.temporal_encoder(temporal_feat)
            temporal_emb = temporal_emb.unsqueeze(1).expand(-1, self.num_joints, -1)
        else:
            temporal_emb = None

        # --- Step 7: Joint type embeddings ---
        joint_indices = torch.arange(self.num_joints, device=keypoints.device)
        type_emb = self.joint_type_embedding(joint_indices)
        type_emb = type_emb.unsqueeze(0).expand(keypoints.size(0), -1, -1)

        # --- Step 8: Position encoding ---
        pos_emb = self.pos_encoder(positions)

        # --- Step 9: Concatenate all features ---
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
        if self.use_temporal and temporal_emb is not None:
            features.append(temporal_emb)

        combined = torch.cat(features, dim=-1)

        # --- Step 10: Project to final embedding ---
        joint_embeddings = self.projection(combined)
        joint_embeddings = torch.nan_to_num(joint_embeddings, nan=0.0, posinf=0.0, neginf=0.0)


        output = {'joint_embeddings': joint_embeddings}

        if return_raw:
            output['raw_features'] = {
                'positions': positions,
                'angles': angles,
                'velocity': velocity,
                'angular_velocity': angular_velocity,
                'confidence': confidence,
                'type_emb': type_emb,
                'pos_emb': pos_emb,
                'angle_emb': angle_emb if self.use_angles else None,
            }

        return output


    # -------------------------------------------------------------------------
    # Angle computation using JointAngles (numpy, loop over joints)
    # -------------------------------------------------------------------------
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
    # -------------------------------------------------------------------------
    # Augmentation methods (only applied when apply_augmentations=True and training)
    # -------------------------------------------------------------------------
    def _apply_augmentations(self,
                         keypoints: torch.Tensor,
                         prev_keypoints: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
    Apply per‑sample augmentations to keypoints and optionally to prev_keypoints.
    The same random parameters are used for both to preserve motion.
        """
        if not self.training:
            return keypoints, prev_keypoints

        batch_size, num_joints, _ = keypoints.shape
        device = keypoints.device

    # Clone to avoid in-place modifications of the original tensor
        kp = keypoints.clone()
        prev_kp = prev_keypoints.clone() if prev_keypoints is not None else None

    # ------------------------------------------------------------
    # Rotation (per‑sample)
    # ------------------------------------------------------------
        if 'rotation' in self.augmentations and self.augmentations['rotation'] > 0:
            angle_range = self.augmentations['rotation']
            angles = torch.empty(batch_size, device=device).uniform_(-angle_range, angle_range)
            angle_rad = angles * math.pi / 180.0
            cos_a = torch.cos(angle_rad)
            sin_a = torch.sin(angle_rad)

        # Build rotation matrices of shape (batch, 2, 2)
            rot_mats = torch.stack([
                torch.stack([cos_a, -sin_a], dim=-1),
                torch.stack([sin_a,  cos_a], dim=-1)
            ], dim=-2)   # (batch, 2, 2)

        # Apply to current frame
            center = kp[..., :2].mean(dim=1, keepdim=True)          # (batch, 1, 2)
            centered = kp[..., :2] - center
            rotated = torch.matmul(centered, rot_mats.transpose(-1, -2))   # (batch, joints, 2)
            kp[..., :2] = rotated + center

        # Apply the same rotation to previous frame
            if prev_kp is not None:
                center_prev = prev_kp[..., :2].mean(dim=1, keepdim=True)
                centered_prev = prev_kp[..., :2] - center_prev
                rotated_prev = torch.matmul(centered_prev, rot_mats.transpose(-1, -2))
                prev_kp[..., :2] = rotated_prev + center_prev

    # ------------------------------------------------------------
    # Scale (per‑sample)
    # ------------------------------------------------------------
        if 'scale' in self.augmentations and self.augmentations['scale'] > 0:
            scale_range = self.augmentations['scale']
            scales = torch.empty(batch_size, device=device).uniform_(1 - scale_range, 1 + scale_range)
        # Expand to (batch, 1, 1) for broadcasting
            scales = scales.view(batch_size, 1, 1)

            kp[..., :2] = kp[..., :2] * scales
            if prev_kp is not None:
                prev_kp[..., :2] = prev_kp[..., :2] * scales

    # ------------------------------------------------------------
    # Gaussian noise (per‑sample per‑joint)
    # ------------------------------------------------------------
        if 'noise' in self.augmentations and self.augmentations['noise'] > 0:
            noise = torch.randn_like(kp[..., :2]) * self.augmentations['noise']
            kp[..., :2] = kp[..., :2] + noise
            if prev_kp is not None:
                prev_kp[..., :2] = prev_kp[..., :2] + noise

    # ------------------------------------------------------------
    # Masking (per‑sample per‑joint)
    # ------------------------------------------------------------
        mask_prob = self.augmentations.get('mask', 0)
        if mask_prob > 0:
            # Create mask of shape (batch, num_joints, 1)
            mask = torch.rand(batch_size, num_joints, 1, device=device) < mask_prob
        # Set confidence to 0 where mask is True
            kp[..., 2:3] = kp[..., 2:3] * (~mask).float()
            if prev_kp is not None:
                prev_kp[..., 2:3] = prev_kp[..., 2:3] * (~mask).float()

        return kp, prev_kp