"""跌倒状态分类模块 - 识别跌倒后的身体姿态"""

from enum import Enum
from typing import Tuple
import numpy as np
from .pose_detector import PoseFrame


class PostureType(Enum):
    """跌倒姿态枚举"""
    UPRIGHT = "upright"            # 直立
    SUPINE = "supine"              # 仰卧位（背部朝下）
    PRONE = "prone"                # 俯卧位（脸朝下）
    SIDE_LEFT = "side_left"        # 左侧卧位
    SIDE_RIGHT = "side_right"      # 右侧卧位
    SITTING = "sitting"            # 坐着
    CURLED = "curled"              # 蜷缩


class PostureClassifier:
    """跌倒状态分类器"""

    def __init__(self):
        """
        初始化姿态分类器
        """
        self.current_posture = PostureType.UPRIGHT
        self.posture_confidence = 0.0

    def classify(self, pose_frame: PoseFrame) -> Tuple[PostureType, float]:
        """
        分类跌倒姿态
        
        Args:
            pose_frame: 姿态帧
            
        Returns:
            (姿态类型, 置信度)
        """
        features = self._extract_features(pose_frame)
        
        if features is None:
            return PostureType.UPRIGHT, 0.0
        
        # 使用特征进行分类
        posture, confidence = self._classify_by_features(features)
        
        self.current_posture = posture
        self.posture_confidence = confidence
        
        return posture, confidence

    def _extract_features(self, pose_frame: PoseFrame) -> dict:
        """
        从姿态帧提取特征
        
        Args:
            pose_frame: 姿态帧
            
        Returns:
            特征字典
        """
        landmarks = pose_frame.landmarks
        
        # 检查必要的关键点
        required_indices = [11, 12, 23, 24, 25, 26, 27, 28]  # 肩膀、髋、膝、踝
        if not all(landmarks[i].confidence > 0.3 for i in required_indices):
            return None
        
        features = {}
        
        # 关键点位置
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        left_knee = landmarks[25]
        right_knee = landmarks[26]
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        
        # 身体中心
        body_center_x = (left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x) / 4
        body_center_y = (left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y) / 4
        features['body_center'] = (body_center_x, body_center_y)
        
        # 肩膀高度（平均Y坐标）
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        features['shoulder_y'] = shoulder_y
        
        # 髋部高度
        hip_y = (left_hip.y + right_hip.y) / 2
        features['hip_y'] = hip_y
        
        # 膝盖高度
        knee_y = (left_knee.y + right_knee.y) / 2
        features['knee_y'] = knee_y
        
        # 踝部高度
        ankle_y = (left_ankle.y + right_ankle.y) / 2
        features['ankle_y'] = ankle_y
        
        # 肩膀与髋部的水平距离
        shoulder_hip_dx = abs(
            (left_shoulder.x + right_shoulder.x) / 2 -
            (left_hip.x + right_hip.x) / 2
        )
        features['shoulder_hip_dx'] = shoulder_hip_dx
        
        # 肩膀与髋部的竖直距离
        shoulder_hip_dy = abs(shoulder_y - hip_y)
        features['shoulder_hip_dy'] = shoulder_hip_dy
        
        # 左右肩膀的倾斜
        shoulder_tilt = abs(left_shoulder.y - right_shoulder.y)
        features['shoulder_tilt'] = shoulder_tilt
        
        # 左右髋部的倾斜
        hip_tilt = abs(left_hip.y - right_hip.y)
        features['hip_tilt'] = hip_tilt
        
        # 膝盖弯曲程度（身体高度）
        body_height = abs(shoulder_y - ankle_y)
        features['body_height'] = body_height
        
        # 腿部伸展程度
        leg_extension = abs(knee_y - ankle_y)
        features['leg_extension'] = leg_extension
        
        return features

    def _classify_by_features(self, features: dict) -> Tuple[PostureType, float]:
        """
        根据特征分类姿态
        
        Args:
            features: 特征字典
            
        Returns:
            (姿态类型, 置信度)
        """
        shoulder_y = features['shoulder_y']
        hip_y = features['hip_y']
        knee_y = features['knee_y']
        ankle_y = features['ankle_y']
        shoulder_hip_dy = features['shoulder_hip_dy']
        shoulder_tilt = features['shoulder_tilt']
        hip_tilt = features['hip_tilt']
        body_height = features['body_height']
        leg_extension = features['leg_extension']
        
        # 直立：肩膀 < 髋 < 膝盖 < 踝，身体高度大
        if (shoulder_y < hip_y < knee_y < ankle_y and body_height > 0.5):
            return PostureType.UPRIGHT, 0.9
        
        # 仰卧位：肩膀 ≈ 髋 ≈ 膝盖 ≈ 踝（都在下方），肩膀和髋部倾斜小
        if (abs(shoulder_y - hip_y) < 0.15 and
            abs(hip_y - ankle_y) < 0.15 and
            shoulder_tilt < 0.1 and hip_tilt < 0.1):
            return PostureType.SUPINE, 0.85
        
        # 坐着：膝盖弯曲，身体高度中等
        if (hip_y > knee_y > ankle_y and
            0.2 < body_height < 0.4 and
            leg_extension < 0.1):
            return PostureType.SITTING, 0.80
        
        # 侧卧位：肩膀或髋部有明显倾斜
        if (shoulder_tilt > 0.15 or hip_tilt > 0.15) and body_height < 0.3:
            if shoulder_y < hip_y:
                return PostureType.SIDE_LEFT, 0.75
            else:
                return PostureType.SIDE_RIGHT, 0.75
        
        # 俯卧位：身体非常贴近地面，腿部伸展
        if (body_height < 0.25 and leg_extension > 0.1 and
            shoulder_y > 0.7):
            return PostureType.PRONE, 0.80
        
        # 蜷缩：身体紧凑，高度很低
        if (body_height < 0.2 and leg_extension < 0.1):
            return PostureType.CURLED, 0.75
        
        # 默认分类
        return PostureType.UPRIGHT, 0.5

    def get_posture_description(self) -> str:
        """
        获取姿态描述
        
        Returns:
            姿态描述文本
        """
        descriptions = {
            PostureType.UPRIGHT: "直立",
            PostureType.SUPINE: "仰卧位（背部朝下）",
            PostureType.PRONE: "俯卧位（脸朝下）",
            PostureType.SIDE_LEFT: "左侧卧位",
            PostureType.SIDE_RIGHT: "右侧卧位",
            PostureType.SITTING: "坐着",
            PostureType.CURLED: "蜷缩",
        }
        return descriptions.get(self.current_posture, "未知")
