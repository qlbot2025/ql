"""多模态融合模块 - 融合RGB、点云、雷达数据进行跌倒检测"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from .fall_stage_detector import FallStage
from .posture_classifier import PostureType


class ModalityType(Enum):
    """模态类型"""
    RGB = "rgb"
    POINT_CLOUD = "point_cloud"
    RADAR = "radar"


class FusionStrategy(Enum):
    """融合策略"""
    EARLY_FUSION = "early"  # 早期融合（特征级）
    LATE_FUSION = "late"    # 后期融合（决策级）
    HYBRID_FUSION = "hybrid"  # 混合融合


@dataclass
class ModalityResult:
    """单个模态的检测结果"""
    modality: ModalityType
    is_fall: bool
    confidence: float
    stage: FallStage
    stage_confidence: float
    posture: PostureType
    posture_confidence: float
    features: Dict
    timestamp: float


@dataclass
class FusionResult:
    """融合结果"""
    is_fall: bool
    confidence: float
    stage: FallStage
    stage_confidence: float
    posture: PostureType
    posture_confidence: float
    modality_results: Dict[ModalityType, ModalityResult]
    fusion_scores: Dict[str, float]
    timestamp: float


class MultiModalFusion:
    """多模态传感器融合器"""

    def __init__(self, strategy: FusionStrategy = FusionStrategy.HYBRID_FUSION,
                weights: Dict[ModalityType, float] = None):
        """
        初始化多模态融合器
        
        Args:
            strategy: 融合策略
            weights: 各模态的权重
        """
        self.strategy = strategy
        
        if weights is None:
            # 默认权重
            self.weights = {
                ModalityType.RGB: 0.4,
                ModalityType.POINT_CLOUD: 0.35,
                ModalityType.RADAR: 0.25,
            }
        else:
            self.weights = weights
        
        # 归一化权重
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        self.modality_history = []

    def fuse(self, results: Dict[ModalityType, ModalityResult],
            timestamp: float = 0.0) -> FusionResult:
        """
        融合多个模态的检测结果
        
        Args:
            results: {模态类型: 检测结果}
            timestamp: 时间戳
            
        Returns:
            融合结果
        """
        if len(results) == 0:
            return None
        
        if self.strategy == FusionStrategy.LATE_FUSION:
            return self._late_fusion(results, timestamp)
        elif self.strategy == FusionStrategy.EARLY_FUSION:
            return self._early_fusion(results, timestamp)
        else:  # HYBRID_FUSION
            return self._hybrid_fusion(results, timestamp)

    def _late_fusion(self, results: Dict[ModalityType, ModalityResult],
                    timestamp: float) -> FusionResult:
        """
        后期融合 - 在决策级进行融合
        
        Args:
            results: 各模态结果
            timestamp: 时间戳
            
        Returns:
            融合结果
        """
        fusion_scores = {}
        
        # 加权求和各模态的跌倒置信度
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for modality, result in results.items():
            if modality in self.weights:
                weight = self.weights[modality]
                weighted_confidence += result.confidence * weight
                total_weight += weight
                fusion_scores[f"{modality.value}_confidence"] = result.confidence
        
        if total_weight > 0:
            fusion_confidence = weighted_confidence / total_weight
        else:
            fusion_confidence = 0.0
        
        # 跌倒决策：任何模态置信度超过阈值，或多个模态同意
        is_fall = self._decide_fall_late(results, fusion_confidence)
        
        # 聚合阶段信息
        stage, stage_conf = self._aggregate_stage(results)
        
        # 聚合姿态信息
        posture, posture_conf = self._aggregate_posture(results)
        
        fusion_scores['is_fall'] = float(is_fall)
        fusion_scores['overall_confidence'] = fusion_confidence
        
        return FusionResult(
            is_fall=is_fall,
            confidence=fusion_confidence,
            stage=stage,
            stage_confidence=stage_conf,
            posture=posture,
            posture_confidence=posture_conf,
            modality_results=results,
            fusion_scores=fusion_scores,
            timestamp=timestamp
        )

    def _early_fusion(self, results: Dict[ModalityType, ModalityResult],
                     timestamp: float) -> FusionResult:
        """
        早期融合 - 在特征级进行融合
        
        Args:
            results: 各模态结果
            timestamp: 时间戳
            
        Returns:
            融合结果
        """
        # 提取各模态的特征
        fused_features = self._extract_fused_features(results)
        
        # 基于融合特征进行决策
        fusion_confidence, is_fall = self._classify_fused_features(fused_features)
        
        # 聚合阶段和姿态信息
        stage, stage_conf = self._aggregate_stage(results)
        posture, posture_conf = self._aggregate_posture(results)
        
        fusion_scores = {
            'is_fall': float(is_fall),
            'overall_confidence': fusion_confidence,
            'fused_features': fused_features
        }
        
        return FusionResult(
            is_fall=is_fall,
            confidence=fusion_confidence,
            stage=stage,
            stage_confidence=stage_conf,
            posture=posture,
            posture_confidence=posture_conf,
            modality_results=results,
            fusion_scores=fusion_scores,
            timestamp=timestamp
        )

    def _hybrid_fusion(self, results: Dict[ModalityType, ModalityResult],
                      timestamp: float) -> FusionResult:
        """
        混合融合 - 结合早期和后期融合的优点
        
        Args:
            results: 各模态结果
            timestamp: 时间戳
            
        Returns:
            融合结果
        """
        # 第一阶段：特征级融合
        fused_features = self._extract_fused_features(results)
        feature_confidence, feature_is_fall = self._classify_fused_features(fused_features)
        
        # 第二阶段：决策级融合
        decision_confidence = 0.0
        decision_agree_count = 0
        
        for modality, result in results.items():
            if result.is_fall:
                decision_agree_count += 1
            decision_confidence += result.confidence * self.weights.get(modality, 0.33)
        
        # 融合两个阶段的结果
        # 权重：特征级60%，决策级40%
        final_confidence = feature_confidence * 0.6 + decision_confidence * 0.4
        
        # 如果两个阶段都同意跌倒，或置信度很高，判定为跌倒
        is_fall = (feature_is_fall and decision_agree_count >= 1) or final_confidence > 0.7
        
        # 聚合阶段和姿态
        stage, stage_conf = self._aggregate_stage(results)
        posture, posture_conf = self._aggregate_posture(results)
        
        fusion_scores = {
            'is_fall': float(is_fall),
            'overall_confidence': final_confidence,
            'feature_level_confidence': feature_confidence,
            'decision_level_confidence': decision_confidence,
            'modality_agreement': decision_agree_count / max(len(results), 1),
        }
        
        return FusionResult(
            is_fall=is_fall,
            confidence=final_confidence,
            stage=stage,
            stage_confidence=stage_conf,
            posture=posture,
            posture_confidence=posture_conf,
            modality_results=results,
            fusion_scores=fusion_scores,
            timestamp=timestamp
        )

    def _extract_fused_features(self, results: Dict[ModalityType, ModalityResult]) -> dict:
        """
        提取融合特征
        
        Args:
            results: 各模态结果
            
        Returns:
            融合特征字典
        """
        fused_features = {}
        
        for modality, result in results.items():
            prefix = f"{modality.value}_"
            for key, value in result.features.items():
                if isinstance(value, (int, float)):
                    fused_features[prefix + key] = value
        
        return fused_features

    def _classify_fused_features(self, fused_features: dict) -> Tuple[float, bool]:
        """
        基于融合特征进行分类
        
        Args:
            fused_features: 融合特征
            
        Returns:
            (置信度, 是否跌倒)
        """
        score = 0.0
        
        # RGB相关特征
        if 'rgb_height_drop' in fused_features and fused_features['rgb_height_drop'] > 0.4:
            score += 0.2
        
        if 'rgb_velocity' in fused_features and fused_features['rgb_velocity'] > 0.4:
            score += 0.15
        
        if 'rgb_tilt_angle' in fused_features and fused_features['rgb_tilt_angle'] > 50:
            score += 0.15
        
        # 点云相关特征
        if 'point_cloud_body_height' in fused_features and fused_features['point_cloud_body_height'] < 0.8:
            score += 0.2
        
        if 'point_cloud_width_height_ratio' in fused_features and fused_features['point_cloud_width_height_ratio'] > 2.0:
            score += 0.15
        
        # 雷达相关特征
        if 'radar_vertical_velocity' in fused_features and fused_features['radar_vertical_velocity'] < -0.5:
            score += 0.15
        
        is_fall = score > 0.5
        confidence = min(score, 1.0)
        
        return confidence, is_fall

    def _decide_fall_late(self, results: Dict[ModalityType, ModalityResult],
                         fusion_confidence: float) -> bool:
        """
        后期融合的跌倒决策
        
        Args:
            results: 各模态结果
            fusion_confidence: 融合置信度
            
        Returns:
            是否跌倒
        """
        # 规则：多数投票或高置信度
        fall_votes = sum(1 for r in results.values() if r.is_fall)
        total_votes = len(results)
        
        # 至少2个模态同意，或置信度>0.7
        return (fall_votes >= 2) or (fusion_confidence > 0.7 and fall_votes >= 1)

    def _aggregate_stage(self, results: Dict[ModalityType, ModalityResult]) -> Tuple[FallStage, float]:
        """
        聚合跌倒阶段信息
        
        Args:
            results: 各模态结果
            
        Returns:
            (阶段, 置信度)
        """
        if len(results) == 0:
            return FallStage.NORMAL, 0.0
        
        # 加权平均置��度
        stage_confidences = {}
        for modality, result in results.items():
            stage = result.stage
            conf = result.stage_confidence * self.weights.get(modality, 0.33)
            if stage not in stage_confidences:
                stage_confidences[stage] = 0.0
            stage_confidences[stage] += conf
        
        if len(stage_confidences) == 0:
            return FallStage.NORMAL, 0.0
        
        best_stage = max(stage_confidences, key=stage_confidences.get)
        return best_stage, stage_confidences[best_stage]

    def _aggregate_posture(self, results: Dict[ModalityType, ModalityResult]) -> Tuple[PostureType, float]:
        """
        聚合姿态信息
        
        Args:
            results: 各模态结果
            
        Returns:
            (姿态, 置信度)
        """
        if len(results) == 0:
            return PostureType.UPRIGHT, 0.0
        
        # 加权平均置信度
        posture_confidences = {}
        for modality, result in results.items():
            posture = result.posture
            conf = result.posture_confidence * self.weights.get(modality, 0.33)
            if posture not in posture_confidences:
                posture_confidences[posture] = 0.0
            posture_confidences[posture] += conf
        
        if len(posture_confidences) == 0:
            return PostureType.UPRIGHT, 0.0
        
        best_posture = max(posture_confidences, key=posture_confidences.get)
        return best_posture, posture_confidences[best_posture]

    def get_fusion_report(self, fusion_result: FusionResult) -> str:
        """
        生成融合报告
        
        Args:
            fusion_result: 融合结果
            
        Returns:
            报告文本
        """
        report = (
            f"\\n{'='*70}\\n"
            f"多模态融合检测报告\\n"
            f"{'='*70}\\n"
            f"跌倒检测: {'是' if fusion_result.is_fall else '否'}\\n"
            f"总体置信度: {fusion_result.confidence:.2%}\\n"
            f"跌倒阶段: {fusion_result.stage.value} ({fusion_result.stage_confidence:.2%})\\n"
            f"身体姿态: {fusion_result.posture.value} ({fusion_result.posture_confidence:.2%})\\n"
            f"\\n各模态结果:\\n"
        )
        
        for modality, result in fusion_result.modality_results.items():
            report += (
                f"  {modality.value.upper()}:\\n"
                f"    - 跌倒: {result.is_fall}, 置信度: {result.confidence:.2%}\\n"
                f"    - 阶段: {result.stage.value}\\n"
                f"    - 姿态: {result.posture.value}\\n"
            )
        
        report += (
            f"\\n融合得分:\\n"
        )
        
        for key, value in fusion_result.fusion_scores.items():
            if isinstance(value, float):
                report += f"  - {key}: {value:.2%}\\n"
        
        report += f"{'='*70}\\n"
        
        return report
