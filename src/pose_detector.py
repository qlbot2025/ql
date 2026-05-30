"""姿态识别模块 - 使用 MediaPipe 检测人体关键点"""

import numpy as np
import cv2
import mediapipe as mp
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class KeyPoint:
    """关键点数据结构"""
    x: float
    y: float
    z: float
    confidence: float


@dataclass
class PoseFrame:
    """单帧的姿态数据"""
    landmarks: List[KeyPoint]
    frame_id: int
    timestamp: float
    height: int
    width: int


class PoseDetector:
    """使用 MediaPipe Pose 进行实时姿态检测"""

    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.5):
        """
        初始化姿态检测器
        
        Args:
            model_complexity: 模型复杂度 (0=轻量, 1=标准, 2=重型)
            min_detection_confidence: 最小检测置信度
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )
        
        # MediaPipe 关键点顺序
        self.keypoint_names = [
            'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear',
            'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_pinky', 'right_pinky',
            'left_index', 'right_index',
            'left_thumb', 'right_thumb',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle',
            'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index',
        ]

    def detect(self, frame: np.ndarray, frame_id: int = 0, timestamp: float = 0.0) -> Optional[PoseFrame]:
        """
        检测单帧中的人体姿态
        
        Args:
            frame: 输入图像 (BGR format)
            frame_id: 帧编号
            timestamp: 时间戳
            
        Returns:
            PoseFrame 对象或 None（无检测结果）
        """
        height, width = frame.shape[:2]
        
        # 转换为 RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 进行姿态检测
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks is None:
            return None
        
        # 转换关键点格式
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append(KeyPoint(
                x=lm.x,
                y=lm.y,
                z=lm.z,
                confidence=lm.visibility
            ))
        
        return PoseFrame(
            landmarks=landmarks,
            frame_id=frame_id,
            timestamp=timestamp,
            height=height,
            width=width
        )

    def draw_landmarks(self, frame: np.ndarray, pose_frame: PoseFrame) -> np.ndarray:
        """
        在图像上绘制关键点和骨架
        
        Args:
            frame: 输入图像
            pose_frame: 姿态数据
            
        Returns:
            绘制后的图像
        """
        result_frame = frame.copy()
        h, w = pose_frame.height, pose_frame.width
        
        # 定义骨架连接
        connections = [
            (11, 12),  # shoulders
            (11, 13), (13, 15),  # left arm
            (12, 14), (14, 16),  # right arm
            (11, 23), (12, 24),  # torso
            (23, 24),  # hips
            (23, 25), (25, 27), (27, 29), (29, 31),  # left leg
            (24, 26), (26, 28), (28, 30), (30, 32),  # right leg
        ]
        
        # 绘制骨架
        for start, end in connections:
            start_lm = pose_frame.landmarks[start]
            end_lm = pose_frame.landmarks[end]
            
            if start_lm.confidence > 0.3 and end_lm.confidence > 0.3:
                start_pos = (int(start_lm.x * w), int(start_lm.y * h))
                end_pos = (int(end_lm.x * w), int(end_lm.y * h))
                cv2.line(result_frame, start_pos, end_pos, (0, 255, 0), 2)
        
        # 绘制关键点
        for lm in pose_frame.landmarks:
            if lm.confidence > 0.3:
                pos = (int(lm.x * w), int(lm.y * h))
                cv2.circle(result_frame, pos, 4, (0, 0, 255), -1)
        
        return result_frame

    def get_keypoint_by_name(self, pose_frame: PoseFrame, name: str) -> Optional[KeyPoint]:
        """
        通过名称获取关键点
        
        Args:
            pose_frame: 姿态数据
            name: 关键点名称
            
        Returns:
            关键点或 None
        """
        try:
            idx = self.keypoint_names.index(name)
            return pose_frame.landmarks[idx]
        except (ValueError, IndexError):
            return None
