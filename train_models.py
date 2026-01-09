import os
import logging
from data_processor import XSSDataProcessor
from models import XGBoostModel, BiLSTMModel, TransformerModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_all_models():
    """训练所有模型"""
    logger.info("=" * 60)
    logger.info("开始训练所有模型")
    logger.info("=" * 60)
    
    # 创建模型保存目录
    os.makedirs('models', exist_ok=True)
    
    # 1. 准备数据
    logger.info("\n[1/5] 准备训练数据...")
    processor = XSSDataProcessor()
    X_train, X_test, y_train, y_test = processor.prepare_training_data()
    
    # 保存训练数据预处理器
    processor.save_preprocessor('models/processor.pkl')
    logger.info(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
    
    # 2. 训练XGBoost模型
    logger.info("\n[2/5] 训练XGBoost模型...")
    xgboost_model = XGBoostModel()
    xgboost_metrics = xgboost_model.train(X_train, y_train, X_test, y_test)
    xgboost_model.save()
    logger.info(f"XGBoost模型评估: {xgboost_metrics}")
    
    # 3. 训练BiLSTM模型
    logger.info("\n[3/5] 训练BiLSTM模型...")
    bilstm_model = BiLSTMModel()
    bilstm_metrics = bilstm_model.train(X_train, y_train, X_test, y_test, epochs=10, batch_size=32)
    bilstm_model.save()
    logger.info(f"BiLSTM模型评估: {bilstm_metrics}")
    
    # 4. 训练Transformer模型
    logger.info("\n[4/5] 训练Transformer模型...")
    transformer_model = TransformerModel()
    transformer_metrics = transformer_model.train(X_train, y_train, X_test, y_test, epochs=10, batch_size=32)
    transformer_model.save()
    logger.info(f"Transformer模型评估: {transformer_metrics}")
    
    # 5. 测试集成模型
    logger.info("\n[5/5] 测试集成模型...")
    from ensemble import XSSDetectorEnsemble
    detector = XSSDetectorEnsemble()
    
    if detector.load_models():
        # 测试几个样本
        test_texts = [
            "<script>alert('xss')</script>",
            "Hello world",
            "<img src=x onerror=alert(1)>",
            "This is normal text",
            "<body onload=alert('xss')>",
            "Check out this website: https://example.com"
        ]
        
        logger.info("\n测试样本检测结果:")
        for text in test_texts:
            result = detector.detect(text)
            logger.info(f"输入: {text[:50]}")
            logger.info(f"  -> 检测结果: {'XSS攻击' if result['is_xss'] else '正常'}")
            logger.info(f"  -> 集成概率: {result['ensemble']['probability']:.4f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("所有模型训练完成!")
    logger.info("=" * 60)
    
    # 打印模型性能对比
    logger.info("\n模型性能对比:")
    logger.info("-" * 60)
    logger.info(f"{'模型':<15} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10}")
    logger.info("-" * 60)
    logger.info(f"{'XGBoost':<15} {xgboost_metrics['accuracy']:.4f}    {xgboost_metrics['precision']:.4f}    {xgboost_metrics['recall']:.4f}    {xgboost_metrics['f1']:.4f}")
    logger.info(f"{'BiLSTM':<15} {bilstm_metrics['accuracy']:.4f}    {bilstm_metrics['precision']:.4f}    {bilstm_metrics['recall']:.4f}    {bilstm_metrics['f1']:.4f}")
    logger.info(f"{'Transformer':<15} {transformer_metrics['accuracy']:.4f}    {transformer_metrics['precision']:.4f}    {transformer_metrics['recall']:.4f}    {transformer_metrics['f1']:.4f}")
    logger.info("-" * 60)
    
    # 保存模型性能到文件
    import json
    metrics_summary = {
        'xgboost': xgboost_metrics,
        'bilstm': bilstm_metrics,
        'transformer': transformer_metrics
    }
    
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    logger.info("\n模型性能已保存到 models/metrics.json")
    logger.info("\n提示: 训练完成后，重启Flask应用即可加载训练好的模型进行检测")


def test_models():
    """测试模型"""
    logger.info("加载已训练的模型进行测试...")
    
    from ensemble import XSSDetectorEnsemble
    detector = XSSDetectorEnsemble()
    
    if not detector.load_models():
        logger.error("模型加载失败，请先运行训练脚本")
        return
    
    # 测试用例
    test_cases = [
        ("<script>alert('xss')</script>", True),
        ("<img src=x onerror=alert('xss')>", True),
        ("<body onload=alert('xss')>", True),
        ("javascript:alert('xss')", True),
        ("Hello world", False),
        ("This is normal text", False),
        ("<div onclick=\"alert('xss')\">Click</div>", True),
        ("User input: test@example.com", False),
        ("\\x3cscript\\x3ealert('xss')\\x3c/script\\x3e", True),
        ("Regular HTML content", False),
    ]
    
    correct = 0
    total = len(test_cases)
    
    logger.info("\n开始测试...")
    logger.info("-" * 80)
    logger.info(f"{'输入':<40} {'预期':<10} {'实际':<10} {'概率':<10} {'结果'}")
    logger.info("-" * 80)
    
    for text, expected in test_cases:
        result = detector.detect(text)
        predicted = result['is_xss']
        probability = result['ensemble']['probability']
        
        is_correct = predicted == expected
        if is_correct:
            correct += 1
        
        logger.info(f"{text[:40]:<40} {str(expected):<10} {str(predicted):<10} {probability:.4f}    {'✓' if is_correct else '✗'}")
    
    logger.info("-" * 80)
    accuracy = correct / total * 100
    logger.info(f"\n测试结果: {correct}/{total} 正确, 准确率: {accuracy:.2f}%")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # 测试模式
        test_models()
    else:
        # 训练模式
        train_all_models()
