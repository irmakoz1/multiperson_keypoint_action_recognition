# src/joint_embedding/joint_features.py
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class JointAngles:
    """
    Calculate anatomical angles for each joint type.

    Based on biomechanical conventions:
    - Flexion/extension: Decreasing/increasing joint angle
    - Abduction/adduction: Movement away/toward midline
    - Rotation: Internal/external rotation
    """

    @staticmethod
    def calculate_elbow_angle(shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray) -> float:
        """Calculate elbow flexion/extension angle."""
        v1 = elbow - shoulder
        v2 = wrist - elbow
        return JointAngles._angle_between(v1, v2)

    @staticmethod
    def calculate_shoulder_angles(
        torso: np.ndarray, shoulder: np.ndarray, elbow: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Calculate shoulder angles:
        - Flexion/extension: arm forward/backward
        - Abduction/adduction: arm away/toward body
        - Rotation: internal/external rotation (simplified)
        """
        # Vector from torso to shoulder (reference)
        v_torso_shoulder = shoulder - torso

        # Vector from shoulder to elbow (arm direction)
        v_arm = elbow - shoulder

        # Flexion/extension: angle in sagittal plane (x-z if x is forward)
        # Simplified: using projection
        flexion = JointAngles._angle_between(
            np.array([v_arm[0], 0, v_arm[2]]),
            np.array([1, 0, 0])
        )

        # Abduction/adduction: angle in coronal plane (y-z)
        abduction = JointAngles._angle_between(
            np.array([0, v_arm[1], v_arm[2]]),
            np.array([0, 1, 0])
        )

        # Simplified rotation (based on elbow position relative to shoulder)
        rotation = np.arctan2(v_arm[0], v_arm[2])  # Rough estimate

        return flexion, abduction, rotation

    @staticmethod
    def calculate_knee_angle(hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray) -> float:
        """Calculate knee flexion/extension angle."""
        v1 = knee - hip
        v2 = ankle - knee
        return JointAngles._angle_between(v1, v2)

    @staticmethod
    def calculate_hip_angles(
        torso: np.ndarray, hip: np.ndarray, knee: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Calculate hip angles:
        - Flexion/extension: leg forward/backward
        - Abduction/adduction: leg away/toward midline
        - Rotation: internal/external rotation
        """
        # Vector from torso to hip (reference)
        v_torso_hip = hip - torso

        # Vector from hip to knee (leg direction)
        v_leg = knee - hip

        # Flexion/extension
        flexion = JointAngles._angle_between(
            np.array([v_leg[0], 0, v_leg[2]]),
            np.array([1, 0, 0])
        )

        # Abduction/adduction
        abduction = JointAngles._angle_between(
            np.array([0, v_leg[1], v_leg[2]]),
            np.array([0, 1, 0])
        )

        # Rotation (simplified)
        rotation = np.arctan2(v_leg[0], v_leg[2])

        return flexion, abduction, rotation

    @staticmethod
    def calculate_torso_angles(
        left_shoulder: np.ndarray, right_shoulder: np.ndarray,
        left_hip: np.ndarray, right_hip: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Calculate torso angles:
        - Forward/backward lean
        - Lateral bend
        - Rotation
        """
        # Midpoints
        shoulder_mid = (left_shoulder + right_shoulder) / 2
        hip_mid = (left_hip + right_hip) / 2

        # Torso vector
        v_torso = shoulder_mid - hip_mid

        # Lean forward/backward (angle from vertical in sagittal plane)
        lean = JointAngles._angle_between(
            np.array([v_torso[0], 0, v_torso[2]]),
            np.array([0, 0, 1])  # vertical
        )

        # Lateral bend (angle from vertical in coronal plane)
        lateral = JointAngles._angle_between(
            np.array([0, v_torso[1], v_torso[2]]),
            np.array([0, 0, 1])
        )

        # Rotation (difference in shoulder vs hip orientation)
        shoulder_vector = right_shoulder - left_shoulder
        hip_vector = right_hip - left_hip
        rotation = JointAngles._angle_between(
            shoulder_vector[:2],  # Project to horizontal plane
            hip_vector[:2]
        )

        return lean, lateral, rotation

    @staticmethod
    def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate angle between two vectors in radians."""
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.arccos(cos_angle)


class JointFeatureConfig:
    """Configuration for joint features per joint type."""

    # COCO joint indices with their relevant angles
    JOINT_ANGLE_CONFIG = {
        # Joint: [flexion, abduction, rotation] - which angles are relevant
        0: {'name': 'nose', 'angles': [False, False, False]},  # No angles
        1: {'name': 'left_eye', 'angles': [False, False, False]},
        2: {'name': 'right_eye', 'angles': [False, False, False]},
        3: {'name': 'left_ear', 'angles': [False, False, False]},
        4: {'name': 'right_ear', 'angles': [False, False, False]},
        5: {'name': 'left_shoulder', 'angles': [True, True, True]},  # All 3 angles
        6: {'name': 'right_shoulder', 'angles': [True, True, True]},
        7: {'name': 'left_elbow', 'angles': [True, False, False]},   # Only flexion
        8: {'name': 'right_elbow', 'angles': [True, False, False]},
        9: {'name': 'left_wrist', 'angles': [True, True, False]},    # Flexion, abduction
        10: {'name': 'right_wrist', 'angles': [True, True, False]},
        11: {'name': 'left_hip', 'angles': [True, True, True]},      # All 3
        12: {'name': 'right_hip', 'angles': [True, True, True]},
        13: {'name': 'left_knee', 'angles': [True, False, False]},   # Only flexion
        14: {'name': 'right_knee', 'angles': [True, False, False]},
        15: {'name': 'left_ankle', 'angles': [True, True, False]},   # Flexion, abduction
        16: {'name': 'right_ankle', 'angles': [True, True, False]},
    }

    # Joint hierarchy for relative positioning
    JOINT_HIERARCHY = {
        'root': 'torso_mid',  # Reference point
        'left_arm': [5, 7, 9],  # shoulder -> elbow -> wrist
        'right_arm': [6, 8, 10],
        'left_leg': [11, 13, 15],  # hip -> knee -> ankle
        'right_leg': [12, 14, 16],
        'head': [0, 1, 2, 3, 4],  # nose, eyes, ears
    }

