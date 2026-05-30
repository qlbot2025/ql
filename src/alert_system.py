"""告警系统 - 处理跌倒检测结果并发送告警"""

from dataclasses import dataclass
from typing import Optional, Callable, List
from datetime import datetime
from .fall_stage_detector import FallStage
from .posture_classifier import PostureType


@dataclass
class AlertEvent:
    """告警事件"""
    timestamp: float
    event_time: str
    is_fall: bool
    fall_confidence: float
    fall_stage: FallStage
    stage_confidence: float
    posture_type: PostureType
    posture_confidence: float
    description: str


class AlertSystem:
    """告警系统"""

    def __init__(self, fall_threshold: float = 0.6, alert_cooldown: float = 5.0):
        """
        初始化告警系统
        
        Args:
            fall_threshold: 跌倒置信度阈值
            alert_cooldown: 告警冷却时间（秒）
        """
        self.fall_threshold = fall_threshold
        self.alert_cooldown = alert_cooldown
        self.last_alert_time = 0.0
        self.alert_handlers: List[Callable] = []
        self.alert_history: List[AlertEvent] = []

    def register_handler(self, handler: Callable) -> None:
        """
        注册告警处理器
        
        Args:
            handler: 回调函数，签名为 handler(alert_event)
        """
        self.alert_handlers.append(handler)

    def process(self, timestamp: float, is_fall: bool, fall_confidence: float,
                fall_stage: FallStage, stage_confidence: float,
                posture_type: PostureType, posture_confidence: float) -> Optional[AlertEvent]:
        """
        处理检测结果并生成告警
        
        Args:
            timestamp: 时间戳
            is_fall: 是否跌倒
            fall_confidence: 跌倒置信度
            fall_stage: 跌倒阶段
            stage_confidence: 阶段置信度
            posture_type: 姿态类型
            posture_confidence: 姿态置信度
            
        Returns:
            告警事件或 None
        """
        alert_event = None
        
        # 检查是否需要生成告警
        if is_fall and fall_confidence > self.fall_threshold:
            # 检查冷却时间
            if timestamp - self.last_alert_time > self.alert_cooldown:
                alert_event = AlertEvent(
                    timestamp=timestamp,
                    event_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    is_fall=is_fall,
                    fall_confidence=fall_confidence,
                    fall_stage=fall_stage,
                    stage_confidence=stage_confidence,
                    posture_type=posture_type,
                    posture_confidence=posture_confidence,
                    description=self._generate_description(
                        is_fall, fall_confidence, fall_stage,
                        posture_type, posture_confidence
                    )
                )
                
                self.last_alert_time = timestamp
                self.alert_history.append(alert_event)
                
                # 调用所有告警处理器
                self._trigger_handlers(alert_event)
        
        return alert_event

    def _generate_description(self, is_fall: bool, fall_confidence: float,
                            fall_stage: FallStage, posture_type: PostureType,
                            posture_confidence: float) -> str:
        """
        生成告警描述
        
        Returns:
            描述文本
        """
        if not is_fall:
            return "正常活动"
        
        stage_desc = {
            FallStage.NORMAL: "正常",
            FallStage.BALANCE_LOSS: "平衡失去",
            FallStage.FALLING: "下落中",
            FallStage.IMPACT: "撞击阶段",
            FallStage.LYING: "静止躺着",
        }
        
        posture_desc = {
            PostureType.UPRIGHT: "直立",
            PostureType.SUPINE: "仰卧位",
            PostureType.PRONE: "俯卧位",
            PostureType.SIDE_LEFT: "左侧卧位",
            PostureType.SIDE_RIGHT: "右侧卧位",
            PostureType.SITTING: "坐着",
            PostureType.CURLED: "蜷缩",
        }
        
        description = (
            f"⚠️ 检测到跌倒！\n"
            f"跌倒置信度: {fall_confidence:.2%}\n"
            f"阶段: {stage_desc.get(fall_stage, '未知')} (置信度: {fall_confidence:.2%})\n"
            f"姿态: {posture_desc.get(posture_type, '未知')} (置信度: {posture_confidence:.2%})"
        )
        
        return description

    def _trigger_handlers(self, alert_event: AlertEvent) -> None:
        """
        触发所有告警处理器
        
        Args:
            alert_event: 告警事件
        """
        for handler in self.alert_handlers:
            try:
                handler(alert_event)
            except Exception as e:
                print(f"警告: 告警处理器执行失败: {e}")

    def get_alert_history(self, limit: int = 10) -> List[AlertEvent]:
        """
        获取告警历史
        
        Args:
            limit: 返回的最大记录数
            
        Returns:
            告警事件列表
        """
        return self.alert_history[-limit:]

    def print_alert(self, alert_event: AlertEvent) -> None:
        """
        打印告警信息到控制台（默认处理器）
        
        Args:
            alert_event: 告警事件
        """
        print(f"\n{'='*60}")
        print(f"[{alert_event.event_time}] 跌倒告警")
        print(alert_event.description)
        print(f"{'='*60}\n")
