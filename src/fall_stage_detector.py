"""跌倒阶段检测模块 - 识别跌倒过程的各个阶段"""

from enum import Enum
from typing import List, Tuple
from .pose_detector import PoseFrame


class FallStage(Enum):
    """跌倒阶段枚举"""
    NORMAL = "normal"              # 正常
    BALANCE_LOSS = "balance_loss"  # 平衡失去
    FALLING = "falling"            # 下落中
    IMPACT = "impact"              # 撞击
    LYING = "lying"                # 静止躺着


class FallStageDetector:
    """跌倒阶段检测器"""

    def __init__(self, window_size: int = 30):
        """
        初始化阶段检测器
        
        Args:
            window_size: 时间窗口大小（帧数）
        """
        self.window_size = window_size
        self.pose_history: List[PoseFrame] = []
        self.current_stage = FallStage.NORMAL
        self.stage_confidence = 0.0
        self.stage_start_frame = 0

    def update(self, pose_frame: PoseFrame, is_falling: bool) -> Tuple[FallStage, float]:
        """
        更新当前阶段
        
        Args:
            pose_frame: 姿态帧
            is_falling: 是否判定为跌倒
            
        Returns:
            (当前阶段, 置信度)
        """
        self.pose_history.append(pose_frame)
        
        if len(self.pose_history) > self.window_size * 2:
            self.pose_history = self.pose_history[-self.window_size*2:]
        
        # 检测阶段
        if not is_falling:
            self.current_stage = FallStage.NORMAL
            self.stage_confidence = 1.0
            return self.current_stage, self.stage_confidence
        
        # 分析跌倒阶段
        self.current_stage, self.stage_confidence = self._detect_stage()
        
        return self.current_stage, self.stage_confidence

    def _detect_stage(self) -> Tuple[FallStage, float]:
        """
        检测当前的跌倒阶段
        
        Returns:
            (阶段, 置信度)
        """
        if len(self.pose_history) < 2:
            return FallStage.NORMAL, 0.0
        
        # 计算身体运动特征
        velocity = self._compute_velocity()
        height_drop = self._compute_height_drop()
        body_contact = self._compute_body_ground_contact()
        tilt_angle = self._compute_tilt_angle()
        
        # 根据特征判断阶段
        if velocity > 0.5 and height_drop < 0.3:
            # 高速度，身体未明显下降 -> 平衡失去
            return FallStage.BALANCE_LOSS, min(velocity, 1.0)
        
        elif velocity > 0.3 and height_drop > 0.2 and not body_contact:
            # 高速度，身体下降，未接触地面 -> 下落中
            return FallStage.FALLING, min(velocity * 0.8, 1.0)
        
        elif velocity > 0.4 and body_contact and height_drop > 0.4:
            # 高速度，接触地面，明显下降 -> 撞击
            return FallStage.IMPACT, min(velocity * 0.6, 1.0)
        
        elif velocity < 0.1 and body_contact and height_drop > 0.3:
            # 低速度，接触地面，明显下降 -> 静止躺着
            return FallStage.LYING, min(0.8, 1.0)
        
        else:
            # 默认为下落中
            return FallStage.FALLING, 0.5

    def _compute_velocity(self) -> float:
        """
        计算当前速度
        
        Returns:
            归一化速度
        """
        if len(self.pose_history) < 2:
            return 0.0
        
        velocities = []
        window = self.pose_history[-10:]  # 最近10帧
        
        for i in range(1, len(window)):
            prev = window[i-1]
            curr = window[i]
            
            prev_x = (prev.landmarks[11].x + prev.landmarks[12].x) / 2
            prev_y = (prev.landmarks[11].y + prev.landmarks[12].y) / 2
            
            curr_x = (curr.landmarks[11].x + curr.landmarks[12].x) / 2
            curr_y = (curr.landmarks[11].y + curr.landmarks[12].y) / 2
            
            dx = curr_x - prev_x
            dy = curr_y - prev_y
            vel = (dx**2 + dy**2) ** 0.5
            velocities.append(vel)
        
        return sum(velocities) / len(velocities) if velocities else 0.0

    def _compute_height_drop(self) -> float:
        """
        计算身体高度下降比例
        
        Returns:
            高度下降比例
        """
        if len(self.pose_history) < 2:
            return 0.0
        
        start = self.pose_history[max(0, len(self.pose_history) - self.window_size)]
        end = self.pose_history[-1]
        
        start_y = (start.landmarks[11].y + start.landmarks[12].y) / 2
        end_y = (end.landmarks[11].y + end.landmarks[12].y) / 2
        
        drop = (end_y - start_y) / (start_y + 1e-6)
        return min(drop, 1.0)

    def _compute_body_ground_contact(self) -> bool:
        """
        检测身体是否接触地面
        
        Returns:
            是否接触地面
        """
        if len(self.pose_history) == 0:
            return False
        
        frame = self.pose_history[-1]
        
        # 检查脚、膝盖、髋部的Y坐标（接近屏幕下方）
        left_ankle = frame.landmarks[27]
        right_ankle = frame.landmarks[28]
        left_knee = frame.landmarks[25]
        right_knee = frame.landmarks[26]
        left_hip = frame.landmarks[23]
        right_hip = frame.landmarks[24]
        
        # 身体下半部分的平均Y坐标
        lower_body_y = (
            left_ankle.y + right_ankle.y +
            left_knee.y + right_knee.y +
            left_hip.y + right_hip.y
        ) / 6
        
        # 如果下半身Y > 0.8（接近屏幕底部），认为接触地面
        return lower_body_y > 0.8

    def _compute_tilt_angle(self) -> float:
        """
        计算身体倾斜角度
        
        Returns:
            倾斜角度（度）
        """
        if len(self.pose_history) == 0:
            return 0.0
        
        frame = self.pose_history[-1]
        
        left_shoulder = frame.landmarks[11]
        right_shoulder = frame.landmarks[12]
        left_hip = frame.landmarks[23]
        right_hip = frame.landmarks[24]
        
        spine_x = ((left_shoulder.x + right_shoulder.x) / 2) - ((left_hip.x + right_hip.x) / 2)
        spine_y = ((left_shoulder.y + right_shoulder.y) / 2) - ((left_hip.y + right_hip.y) / 2)
        
        import numpy as np
        angle = np.arctan2(spine_x, -spine_y) * 180 / np.pi
        return abs(angle)

    def get_stage_description(self) -> str:
        """
        获取阶段描述
        
        Returns:
            阶段描述文本
        """
        descriptions = {
            FallStage.NORMAL: "正常活动",
            FallStage.BALANCE_LOSS: "平衡失去 - 身体开始失控",
            FallStage.FALLING: "下落中 - 快速向地面移动",
            FallStage.IMPACT: "撞击阶段 - 与地面接触",
            FallStage.LYING: "静止躺着 - 倒在地面上",
        }
        return descriptions.get(self.current_stage, "未知")
