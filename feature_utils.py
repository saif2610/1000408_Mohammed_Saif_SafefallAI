"""
Pose-based feature extraction for SafeFall AI.

Reconstructs the 13 features the RandomForest model in safe_fall_model.pkl
expects, from live MediaPipe Pose landmarks:

  8 per-frame ("mediapipe") features:
    aspect_ratio, torso_angle, hip_y, torso_ratio, bbox_w, bbox_h,
    v_hip_y, v_torso_angle

  5 rolling-window ("temporal") features, computed over the last
  `window_size` frames:
    rolling_max_v_hip, rolling_mean_angle, rolling_max_aspect,
    rolling_min_hip_y, angle_change_range

NOTE: these formulas are a best-effort reconstruction based on the
feature names stored in the model bundle, not the original training
script. If predictions look off once you test with real footage, the
fix is almost always here (tune a formula to match training), not in
the model itself.
"""

from collections import deque

import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24


class FallDetector:
    def __init__(self, model_bundle):
        self.model = model_bundle["model"]
        self.feature_columns = model_bundle["feature_columns"]
        self.labels = model_bundle["labels"]
        self.window_size = model_bundle.get("window_size", 10)

        self.pose = mp_pose.Pose(
            static_image_mode=True, model_complexity=1, min_detection_confidence=0.5
        )
        self.history = deque(maxlen=self.window_size)
        self.prev_raw = None

    def reset(self):
        """Call this between separate videos/streams so history doesn't leak across them."""
        self.history.clear()
        self.prev_raw = None

    def _raw_features(self, landmarks):
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        bbox_w = max(xs) - min(xs)
        bbox_h = max(ys) - min(ys) + 1e-6
        aspect_ratio = bbox_w / bbox_h

        ls, rs = landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER]
        lh, rh = landmarks[LEFT_HIP], landmarks[RIGHT_HIP]
        shoulder_mid = ((ls.x + rs.x) / 2, (ls.y + rs.y) / 2)
        hip_mid = ((lh.x + rh.x) / 2, (lh.y + rh.y) / 2)

        dx = hip_mid[0] - shoulder_mid[0]
        dy = hip_mid[1] - shoulder_mid[1]
        torso_angle = float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6)))  # 0=upright, 90=horizontal
        torso_len = float(np.hypot(dx, dy))
        torso_ratio = torso_len / bbox_h

        return {
            "aspect_ratio": aspect_ratio,
            "torso_angle": torso_angle,
            "hip_y": hip_mid[1],
            "torso_ratio": torso_ratio,
            "bbox_w": bbox_w,
            "bbox_h": bbox_h,
        }

    def process_frame(self, frame_bgr):
        """Returns (label, confidence, feature_dict) or (None, None, None) if no person found."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        if not results.pose_landmarks:
            return None, None, None

        raw = self._raw_features(results.pose_landmarks.landmark)

        if self.prev_raw is not None:
            v_hip_y = raw["hip_y"] - self.prev_raw["hip_y"]
            v_torso_angle = raw["torso_angle"] - self.prev_raw["torso_angle"]
        else:
            v_hip_y = 0.0
            v_torso_angle = 0.0

        raw["v_hip_y"] = v_hip_y
        raw["v_torso_angle"] = v_torso_angle

        self.history.append(raw)
        self.prev_raw = raw

        hip_ys = [h["hip_y"] for h in self.history]
        angles = [h["torso_angle"] for h in self.history]
        aspects = [h["aspect_ratio"] for h in self.history]
        v_hips = [h["v_hip_y"] for h in self.history]

        features = dict(raw)
        features["rolling_max_v_hip"] = max(v_hips)
        features["rolling_mean_angle"] = float(np.mean(angles))
        features["rolling_max_aspect"] = max(aspects)
        features["rolling_min_hip_y"] = min(hip_ys)
        features["angle_change_range"] = max(angles) - min(angles)

        vector = np.array([[features[col] for col in self.feature_columns]])
        pred = self.model.predict(vector)[0]
        proba = self.model.predict_proba(vector)[0]
        label = self.labels[pred]
        confidence = float(proba[pred])

        return label, confidence, features
