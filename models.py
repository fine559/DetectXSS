import numpy as np
import xgboost as xgb
import tensorflow as tf
try:
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    import keras
    from keras import layers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score
import pickle
import logging
import os
from data_processor import XSSDataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostModel:
    """XGBoost模型"""
    
    def __init__(self, model_path='models/xgboost_model.pkl'):
        self.model = None
        self.model_path = model_path
        self.processor = XSSDataProcessor()
        
    def train(self, X_train, y_train, X_test=None, y_test=None):
        """训练XGBoost模型"""
        logger.info("开始训练XGBoost模型...")
        
        # 构建特征
        X_train_features = self.processor.build_combined_features(X_train, fit=True)
        if X_test is not None:
            X_test_features = self.processor.build_combined_features(X_test, fit=False)
        
        # 创建XGBoost模型
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        # 训练模型
        self.model.fit(X_train_features, y_train)
        
        # 评估模型
        metrics = {}
        if X_test is not None and y_test is not None:
            y_pred = self.model.predict(X_test_features)
            y_pred_proba = self.model.predict_proba(X_test_features)[:, 1]

            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }

            # 计算AUC
            try:
                metrics['auc'] = roc_auc_score(y_test, y_pred_proba)
            except:
                metrics['auc'] = None

            logger.info(f"XGBoost模型评估结果: {metrics}")

        logger.info("XGBoost模型训练完成")
        return metrics
    
    def predict(self, text):
        """预测单个文本"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        # 清理文本
        cleaned_text = self.processor.clean_text(text)
        
        # 构建特征
        features = self.processor.build_combined_features([cleaned_text], fit=False)
        
        # 预测
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0, 1]
        
        return prediction, probability
    
    def predict_batch(self, texts):
        """批量预测"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        cleaned_texts = [self.processor.clean_text(text) for text in texts]
        features = self.processor.build_combined_features(cleaned_texts, fit=False)
        
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)[:, 1]
        
        return predictions, probabilities
    
    def save(self):
        """保存模型"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'processor': self.processor
            }, f)
        logger.info(f"XGBoost模型已保存到 {self.model_path}")
    
    def load(self):
        """加载模型"""
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.processor = data['processor']
            logger.info(f"XGBoost模型已从 {self.model_path} 加载")
            return True
        except Exception as e:
            logger.error(f"加载XGBoost模型失败: {e}")
            return False

    def evaluate(self, X_test, y_test):
        """评估模型"""
        if self.model is None:
            raise ValueError("模型未训练")

        X_test_features = self.processor.build_combined_features(X_test, fit=False)
        y_pred = self.model.predict(X_test_features)

        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }

    def predict_proba_batch(self, texts):
        """批量预测概率"""
        if self.model is None:
            raise ValueError("模型未训练")

        cleaned_texts = [self.processor.clean_text(text) for text in texts]
        features = self.processor.build_combined_features(cleaned_texts, fit=False)

        return self.model.predict_proba(features)[:, 1]


class BiLSTMModel:
    """BiLSTM模型"""
    
    def __init__(self, model_path='models/bilstm_model.h5'):
        self.model = None
        self.model_path = model_path
        self.processor = XSSDataProcessor()
        self.max_length = 200
        
    def build_model(self, vocab_size=100, embedding_dim=64):
        """构建BiLSTM模型"""
        inputs = keras.Input(shape=(self.max_length,))
        
        # 嵌入层
        x = layers.Embedding(vocab_size, embedding_dim)(inputs)
        
        # 双向LSTM层
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Bidirectional(layers.LSTM(32))(x)
        x = layers.Dropout(0.3)(x)
        
        # 全连接层
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # 输出层
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_test=None, y_test=None, epochs=10, batch_size=32):
        """训练BiLSTM模型"""
        logger.info("开始训练BiLSTM模型...")
        
        # 为深度学习分词
        X_train_seq = self.processor.tokenize_for_deep_learning(X_train)
        if X_test is not None:
            X_test_seq = self.processor.tokenize_for_deep_learning(X_test)
        
        # 构建模型
        vocab_size = len(self.processor.word_to_idx)
        self.model = self.build_model(vocab_size=vocab_size)
        
        self.model.summary()
        
        # 训练模型
        history = self.model.fit(
            X_train_seq, y_train,
            validation_data=(X_test_seq, y_test) if X_test is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # 评估模型
        metrics = {}
        if X_test is not None and y_test is not None:
            y_pred_proba = self.model.predict(X_test_seq).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)

            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }

            # 计算AUC
            try:
                metrics['auc'] = roc_auc_score(y_test, y_pred_proba)
            except:
                metrics['auc'] = None

            logger.info(f"BiLSTM模型评估结果: {metrics}")

        logger.info("BiLSTM模型训练完成")
        return metrics, history
    
    def predict(self, text):
        """预测单个文本"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        # 清理文本
        cleaned_text = self.processor.clean_text(text)
        
        # 分词
        seq = self.processor.tokenize_for_deep_learning([cleaned_text])
        
        # 预测
        probability = self.model.predict(seq)[0, 0]
        prediction = 1 if probability > 0.5 else 0
        
        return prediction, probability
    
    def predict_batch(self, texts):
        """批量预测"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        cleaned_texts = [self.processor.clean_text(text) for text in texts]
        seq = self.processor.tokenize_for_deep_learning(cleaned_texts)
        
        probabilities = self.model.predict(seq).flatten()
        predictions = (probabilities > 0.5).astype(int)
        
        return predictions, probabilities
    
    def save(self):
        """保存模型"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)
        
        # 保存预处理器
        self.processor.save_preprocessor('models/bilstm_processor.pkl')
        
        logger.info(f"BiLSTM模型已保存到 {self.model_path}")
    
    def load(self):
        """加载模型"""
        try:
            self.model = keras.models.load_model(self.model_path)
            self.processor.load_preprocessor('models/bilstm_processor.pkl')
            logger.info(f"BiLSTM模型已从 {self.model_path} 加载")
            return True
        except Exception as e:
            logger.error(f"加载BiLSTM模型失败: {e}")
            return False


class TransformerModel:
    """Transformer模型"""
    
    def __init__(self, model_path='models/transformer_model.h5'):
        self.model = None
        self.model_path = model_path
        self.processor = XSSDataProcessor()
        self.max_length = 200
        
    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        """Transformer编码器块"""
        # 多头注意力
        x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # 残差连接
        res = x + inputs
        
        # 前馈网络
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')(res)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        return x + res
    
    def build_model(self, vocab_size=100, num_transformer_blocks=2, head_size=64, num_heads=4, ff_dim=128):
        """构建Transformer模型"""
        inputs = keras.Input(shape=(self.max_length,))
        
        # 嵌入层和位置编码
        embedding = layers.Embedding(vocab_size, 64)(inputs)
        positions = layers.Embedding(input_dim=self.max_length, output_dim=64)(tf.range(start=0, limit=self.max_length, delta=1))
        x = embedding + positions
        
        # Transformer编码器块
        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout=0.1)
        
        # 全局平均池化
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.2)(x)
        
        # 全连接层
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # 输出层
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_test=None, y_test=None, epochs=10, batch_size=32):
        """训练Transformer模型"""
        logger.info("开始训练Transformer模型...")
        
        # 为深度学习分词
        X_train_seq = self.processor.tokenize_for_deep_learning(X_train)
        if X_test is not None:
            X_test_seq = self.processor.tokenize_for_deep_learning(X_test)
        
        # 构建模型
        vocab_size = len(self.processor.word_to_idx)
        self.model = self.build_model(vocab_size=vocab_size)
        
        self.model.summary()
        
        # 训练模型
        history = self.model.fit(
            X_train_seq, y_train,
            validation_data=(X_test_seq, y_test) if X_test is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # 评估模型
        metrics = {}
        if X_test is not None and y_test is not None:
            y_pred_proba = self.model.predict(X_test_seq).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)

            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }

            # 计算AUC
            try:
                metrics['auc'] = roc_auc_score(y_test, y_pred_proba)
            except:
                metrics['auc'] = None

            logger.info(f"Transformer模型评估结果: {metrics}")

        logger.info("Transformer模型训练完成")
        return metrics, history
    
    def predict(self, text):
        """预测单个文本"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        # 清理文本
        cleaned_text = self.processor.clean_text(text)
        
        # 分词
        seq = self.processor.tokenize_for_deep_learning([cleaned_text])
        
        # 预测
        probability = self.model.predict(seq)[0, 0]
        prediction = 1 if probability > 0.5 else 0
        
        return prediction, probability
    
    def predict_batch(self, texts):
        """批量预测"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        cleaned_texts = [self.processor.clean_text(text) for text in texts]
        seq = self.processor.tokenize_for_deep_learning(cleaned_texts)
        
        probabilities = self.model.predict(seq).flatten()
        predictions = (probabilities > 0.5).astype(int)
        
        return predictions, probabilities
    
    def save(self):
        """保存模型"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)
        
        # 保存预处理器
        self.processor.save_preprocessor('models/transformer_processor.pkl')
        
        logger.info(f"Transformer模型已保存到 {self.model_path}")
    
    def load(self):
        """加载模型"""
        try:
            self.model = keras.models.load_model(self.model_path)
            self.processor.load_preprocessor('models/transformer_processor.pkl')
            logger.info(f"Transformer模型已从 {self.model_path} 加载")
            return True
        except Exception as e:
            logger.error(f"加载Transformer模型失败: {e}")
            return False
