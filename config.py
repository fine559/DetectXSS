# 数据库配置
DATABASE_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',
    'database': 'xss_detection'
}

# 模型配置
MODEL_CONFIG = {
    'max_length': 200,
    'vocab_size': 10000,
    'embedding_dim': 64,
    'lstm_units': 64,
    'transformer_heads': 4,
    'ff_dim': 128
}

# 集成权重
ENSEMBLE_WEIGHTS = {
    'xgboost': 0.3,
    'bilstm': 0.35,
    'transformer': 0.35
}

# 训练配置
TRAINING_CONFIG = {
    'epochs': 10,
    'batch_size': 32,
    'test_size': 0.2,
    'random_state': 42
}

# 模型保存路径
MODEL_PATHS = {
    'xgboost': 'models/xgboost_model.pkl',
    'bilstm': 'models/bilstm_model.h5',
    'transformer': 'models/transformer_model.h5',
    'processor': 'models/processor.pkl',
    'metrics': 'models/metrics.json'
}
