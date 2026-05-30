"""Fall detection system package"""

from .pose_detector import PoseDetector
from .fall_classifier import FallClassifier
from .fall_stage_detector import FallStageDetector
from .posture_classifier import PostureClassifier
from .alert_system import AlertSystem
from .point_cloud_detector import PointCloudDetector
from .radar_detector import RadarDetector
from .multi_modal_fusion import MultiModalFusion, ModalityType, FusionStrategy, ModalityResult

__version__ = "0.2.0"
__all__ = [
    # RGB视频处理
    "PoseDetector",
    "FallClassifier",
    "FallStageDetector",
    "PostureClassifier",
    "AlertSystem",
    
    # 点云处理
    "PointCloudDetector",
    
    # 雷达处理
    "RadarDetector",
    
    # 多模态融合
    "MultiModalFusion",
    "ModalityType",
    "FusionStrategy",
    "ModalityResult",
]
