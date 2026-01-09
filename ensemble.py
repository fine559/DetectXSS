import numpy as np
from models import XGBoostModel, BiLSTMModel, TransformerModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XSSDetectorEnsemble:
    """XSS检测集成模型"""
    
    def __init__(self):
        self.xgboost_model = XGBoostModel()
        self.bilstm_model = BiLSTMModel()
        self.transformer_model = TransformerModel()
        
    def load_models(self):
        """加载所有模型"""
        logger.info("加载模型...")
        
        success = True
        success = self.xgboost_model.load() and success
        success = self.bilstm_model.load() and success
        success = self.transformer_model.load() and success
        
        if success:
            logger.info("所有模型加载成功")
        else:
            logger.warning("部分模型加载失败")
        
        return success
    
    def detect(self, text):
        """检测文本是否为XSS攻击"""
        try:
            # 使用各模型进行预测
            xgb_pred, xgb_prob = self.xgboost_model.predict(text)
            bilstm_pred, bilstm_prob = self.bilstm_model.predict(text)
            transformer_pred, transformer_prob = self.transformer_model.predict(text)
            
            # 加权平均集成
            weights = {
                'xgboost': 0.3,
                'bilstm': 0.35,
                'transformer': 0.35
            }
            
            ensemble_prob = (
                weights['xgboost'] * xgb_prob +
                weights['bilstm'] * bilstm_prob +
                weights['transformer'] * transformer_prob
            )
            
            ensemble_pred = 1 if ensemble_prob > 0.5 else 0
            
            result = {
                'is_xss': bool(ensemble_pred),
                'xgboost': {
                    'prediction': bool(xgb_pred),
                    'probability': float(xgb_prob)
                },
                'bilstm': {
                    'prediction': bool(bilstm_pred),
                    'probability': float(bilstm_prob)
                },
                'transformer': {
                    'prediction': bool(transformer_pred),
                    'probability': float(transformer_prob)
                },
                'ensemble': {
                    'prediction': bool(ensemble_pred),
                    'probability': float(ensemble_prob)
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"检测失败: {e}")
            return {
                'is_xss': False,
                'error': str(e)
            }
    
    def detect_batch(self, texts):
        """批量检测"""
        try:
            # 使用各模型进行批量预测
            xgb_preds, xgb_probs = self.xgboost_model.predict_batch(texts)
            bilstm_preds, bilstm_probs = self.bilstm_model.predict_batch(texts)
            transformer_preds, transformer_probs = self.transformer_model.predict_batch(texts)
            
            results = []
            weights = {
                'xgboost': 0.3,
                'bilstm': 0.35,
                'transformer': 0.35
            }
            
            for i in range(len(texts)):
                ensemble_prob = (
                    weights['xgboost'] * xgb_probs[i] +
                    weights['bilstm'] * bilstm_probs[i] +
                    weights['transformer'] * transformer_probs[i]
                )
                ensemble_pred = 1 if ensemble_prob > 0.5 else 0
                
                results.append({
                    'text': texts[i],
                    'is_xss': bool(ensemble_pred),
                    'xgboost_prob': float(xgb_probs[i]),
                    'bilstm_prob': float(bilstm_probs[i]),
                    'transformer_prob': float(transformer_probs[i]),
                    'ensemble_prob': float(ensemble_prob)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"批量检测失败: {e}")
            return []
    
    def voting_detect(self, text):
        """投票法检测"""
        try:
            xgb_pred, _ = self.xgboost_model.predict(text)
            bilstm_pred, _ = self.bilstm_model.predict(text)
            transformer_pred, _ = self.transformer_model.predict(text)
            
            # 多数投票
            votes = [xgb_pred, bilstm_pred, transformer_pred]
            ensemble_pred = 1 if sum(votes) >= 2 else 0
            
            return {
                'is_xss': bool(ensemble_pred),
                'votes': {
                    'xgboost': bool(xgb_pred),
                    'bilstm': bool(bilstm_pred),
                    'transformer': bool(transformer_pred)
                }
            }
            
        except Exception as e:
            logger.error(f"投票检测失败: {e}")
            return {
                'is_xss': False,
                'error': str(e)
            }


# 创建全局检测器实例
detector = XSSDetectorEnsemble()


def init_detector():
    """初始化检测器"""
    return detector.load_models()


def detect_xss(text):
    """检测XSS攻击的便捷函数"""
    return detector.detect(text)
