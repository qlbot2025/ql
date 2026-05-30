"""跌倒分类模块 - 判断是否发生跌倒"""

import numpy as np
from typing import List, Tuple
from .pose_detector import PoseFrame


class FallClassifier:
    """基于姿态数据的跌倒分类器"""

    def __init__(self, window_size: int = 10, height_threshold: float = 0.5):
        """
        初始化跌倒分类器
        
        Args:
            window_size: 用于特征提取的时间窗口大小（帧数）
            height_threshold: 身体高度下降的阈值（相对值）
        """
        self.window_size = window_size
        self.height_threshold = height_threshold
        self.pose_history: List[PoseFrame] = []

    def update(self, pose_frame: PoseFrame) -> None:
        """
        更新姿态历史记录
        
        Args:
            pose_frame: 新的姿态帧
        """
        self.pose_history.append(pose_frame)
        
        # 保持窗口大小
        if len(self.pose_history) > self.window_size * 2:
            self.pose_history = self.pose_history[-self.window_size*2:]

    def extract_features(self) -> dict:
        """
        从姿态序列中提取特征
        
        Returns:
            特征字典
        """
        if len(self.pose_history) < 2:
            return None
        
        features = {}
        
        # 计算身体高度变化
        current_frame = self.pose_history[-1]
        height_drop = self._compute_height_drop()
        features['height_drop'] = height_drop
        
        # 计算关键点速度
        velocity = self._compute_velocity()
        features['velocity'] = velocity
        
        # 计算加速度
        acceleration = self._compute_acceleration()
        features['acceleration'] = acceleration
        
        # 计算身体倾斜角度
        tilt_angle = self._compute_body_tilt(current_frame)
        features['tilt_angle'] = tilt_angle
        
        # 计算身体姿态得分
        features['posture_score'] = self._compute_posture_score(current_frame)
        
        return features

    def classify(self, pose_frame: PoseFrame) -> Tuple[bool, float]:
        """
        判断是否发生跌倒
        
        Args:
            pose_frame: 姿态帧
            
        Returns:
            (是否跌倒, 置信度)
        """
        self.update(pose_frame)
        
        if len(self.pose_history) < self.window_size:
            return False, 0.0
        
        features = self.extract_features()
        if features is None:
            return False, 0.0
        
        # 简单的规则决策逻辑
        fall_score = 0.0
        
        # 规则1: 身体高度快速下降
        if features['height_drop'] > self.height_threshold:
            fall_score += 0.4
        
        # 规则2: 高速度（快速下落）
        if features['velocity'] > 0.3:
            fall_score += 0.3
        
        # 规则3: 身体倾斜角度大
        if features['tilt_angle'] > 45:  # 度
            fall_score += 0.2
        
        # 规则4: 低姿态得分（身体接近地面）
        if features['posture_score'] < 0.3:
            fall_score += 0.1
        
        is_fall = fall_score > 0.6
        confidence = min(fall_score, 1.0)
        
        return is_fall, confidence

    def _compute_height_drop(self) -> float:
        """
        计算身体高度下降的比例
        
        Returns:
            高度下降比例 (0-1)
        """
        if len(self.pose_history) < 2:
            return 0.0
        
        # 使用肩膀和髋部的平均高度作为身体高度
        window = self.pose_history[-self.window_size:]
        
        if len(window) < 2:
            return 0.0
        
        # 计算起始高度
        start_frame = window[0]
        left_shoulder = start_frame.landmarks[11]
        right_shoulder = start_frame.landmarks[12]
        left_hip = start_frame.landmarks[23]
        right_hip = start_frame.landmarks[24]
        
        if all([left_shoulder.confidence > 0.3, right_shoulder.confidence > 0.3,
                left_hip.confidence > 0.3, right_hip.confidence > 0.3]):
            start_height = min((left_shoulder.y + right_shoulder.y) / 2,
                             (left_hip.y + right_hip.y) / 2)
        else:
            return 0.0
        
        # 计算当前高度
        end_frame = window[-1]
        left_shoulder = end_frame.landmarks[11]
        right_shoulder = end_frame.landmarks[12]
        left_hip = end_frame.landmarks[23]
        right_hip = end_frame.landmarks[24]
        
        if all([left_shoulder.confidence > 0.3, right_shoulder.confidence > 0.3,
                left_hip.confidence > 0.3, right_hip.confidence > 0.3]):
            end_height = min((left_shoulder.y + right_shoulder.y) / 2,
                           (left_hip.y + right_hip.y) / 2)
        else:
            return 0.0
        
        # 计算下降比例
        height_drop = max(0, (start_height - end_height) / (start_height + 1e-6))
        return min(height_drop, 1.0)

    def _compute_velocity(self) -> float:
        """
        计算平均运动速度
        
        Returns:
            归一化的速度 (0-1)
        """
        if len(self.pose_history) < 2:
            return 0.0
        
        velocities = []
        for i in range(1, len(self.pose_history)):
            prev_frame = self.pose_history[i-1]
            curr_frame = self.pose_history[i]
            
            # 使用肩膀中点作为身体位置
            prev_x = (prev_frame.landmarks[11].x + prev_frame.landmarks[12].x) / 2
            prev_y = (prev_frame.landmarks[11].y + prev_frame.landmarks[12].y) / 2
            
            curr_x = (curr_frame.landmarks[11].x + curr_frame.landmarks[12].x) / 2
            curr_y = (curr_frame.landmarks[11].y + curr_frame.landmarks[12].y) / 2
            
            dx = curr_x - prev_x
            dy = curr_y - prev_y
            velocity = np.sqrt(dx**2 + dy**2)
            velocities.append(velocity)
        
        avg_velocity = np.mean(velocities) if velocities else 0.0
        return min(avg_velocity, 1.0)

    def _compute_acceleration(self) -> float:
        """
        计算平均加速度
        
        Returns:
            归一化的加速度
        """
        if len(self.pose_history) < 3:
            return 0.0
        
        velocities = []
        for i in range(1, len(self.pose_history)):
            prev_frame = self.pose_history[i-1]
            curr_frame = self.pose_history[i]
            
            prev_x = (prev_frame.landmarks[11].x + prev_frame.landmarks[12].x) / 2
            prev_y = (prev_frame.landmarks[11].y + prev_frame.landmarks[12].y) / 2
            
            curr_x = (curr_frame.landmarks[11].x + curr_frame.landmarks[12].x) / 2
            curr_y = (curr_frame.landmarks[11].y + curr_frame.landmarks[12].y) / 2
            
            dx = curr_x - prev_x
            dy = curr_y - prev_y
            velocity = np.sqrt(dx**2 + dy**2)
            velocities.append(velocity)
        
        if len(velocities) < 2:
            return 0.0
        
        accelerations = []
        for i in range(1, len(velocities)):
            acc = abs(velocities[i] - velocities[i-1])
            accelerations.append(acc)
        
        avg_acceleration = np.mean(accelerations) if accelerations else 0.0
        return min(avg_acceleration, 1.0)

    def _compute_body_tilt(self, pose_frame: PoseFrame) -> float:
        """
        计算身体倾斜角度
        
        Args:
            pose_frame: 姿态帧
            
        Returns:
            倾斜角度（度）
        """
        left_shoulder = pose_frame.landmarks[11]
        right_shoulder = pose_frame.landmarks[12]
        left_hip = pose_frame.landmarks[23]
        right_hip = pose_frame.landmarks[24]
        
        if not all([left_shoulder.confidence > 0.3, right_shoulder.confidence > 0.3,
                    left_hip.confidence > 0.3, right_hip.confidence > 0.3]):
            return 0.0
        
        # 计算脊柱方向向量
        spine_x = ((left_shoulder.x + right_shoulder.x) / 2) - ((left_hip.x + right_hip.x) / 2)
        spine_y = ((left_shoulder.y + right_shoulder.y) / 2) - ((left_hip.y + right_hip.y) / 2)
        
        # 计算与竖直方向的夹角
        angle = np.arctan2(spine_x, -spine_y) * 180 / np.pi
        return abs(angle)

    def _compute_posture_score(self, pose_frame: PoseFrame) -> float:
        """
        计算身体姿态得分（高=直立，低=躺地）
        
        Args:
            pose_frame: 姿态帧
            
        Returns:
            姿态得分 (0-1)
        """
        # 使用肩膀和髋部的Y坐标差异来判断
        left_shoulder = pose_frame.landmarks[11]
        right_shoulder = pose_frame.landmarks[12]
        left_hip = pose_frame.landmarks[23]
        right_hip = pose_frame.landmarks[24]
        
        if not all([left_shoulder.confidence > 0.3, right_shoulder.confidence > 0.3,
                    left_hip.confidence > 0.3, right_hip.confidence > 0.3]):
            return 0.5
        
        # 肩膀应该在髋部上方（Y坐标更小）
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_y = (left_hip.y + right_hip.y) / 2
        
        height_diff = hip_y - shoulder_y
        
        # 正常直立时，shoulder_y < hip_y，所以 height_diff > 0
        # 躺地时，height_diff 接近 0 或负值
        score = max(0, min(height_diff, 0.5)) / 0.5
        
        return score
