import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XSSDataProcessor:
    """XSS数据预处理类"""
    
    def __init__(self, max_length=200, vocab_size=10000):
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.tfidf_vectorizer = None
        self.tokenizer = None
        self.word_to_idx = {}
        self.idx_to_word = {}
        
    @staticmethod
    def is_xss_pattern(text):
        """检查文本是否包含XSS特征模式"""
        xss_patterns = [
            r'<script[^>]*>.*?</script>',
            r'on\w+\s*=\s*["\'].*?["\']',
            r'javascript:',
            r'vbscript:',
            r'onload\s*=',
            r'onerror\s*=',
            r'onclick\s*=',
            r'onmouseover\s*=',
            r'eval\s*\(',
            r'expression\s*\(',
            r'alert\s*\(',
            r'document\.cookie',
            r'window\.location',
            r'fromCharCode',
            r'\\x[0-9a-fA-F]{2}',
            r'&#\d+;',
            r'<iframe[^>]*>',
            r'<object[^>]*>',
            r'<embed[^>]*>',
            r'src\s*=\s*["\']javascript:',
        ]
        
        for pattern in xss_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    @staticmethod
    def clean_text(text):
        """清理文本"""
        if not isinstance(text, str):
            return ""
        
        # 转小写
        text = text.lower()
        
        # 替换HTML实体
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'&quot;', '"', text)
        text = re.sub(r'&apos;', "'", text)
        text = re.sub(r'&amp;', '&', text)
        
        # 替换十六进制编码
        text = re.sub(r'\\x[0-9a-fA-F]{2}', lambda m: chr(int(m.group(0)[2:], 16)), text)
        
        return text
    
    @staticmethod
    def extract_features(text):
        """提取XSS特征"""
        features = {}
        
        # 特征1: 包含script标签
        features['has_script'] = 1 if '<script' in text.lower() else 0
        
        # 特征2: 包含事件处理器
        events = ['onload', 'onerror', 'onclick', 'onmouseover', 'onfocus', 'onblur']
        features['has_event'] = sum(1 for event in events if event in text.lower())
        
        # 特征3: 包含javascript伪协议
        features['has_javascript'] = 1 if 'javascript:' in text.lower() else 0
        
        # 特征4: 包含eval
        features['has_eval'] = 1 if 'eval(' in text.lower() else 0
        
        # 特征5: 包含document.cookie
        features['has_document'] = 1 if 'document.' in text.lower() else 0
        
        # 特征6: 特殊字符数量
        features['special_chars'] = len(re.findall(r'[<>"\'()]', text))
        
        # 特征7: 文本长度
        features['length'] = len(text)
        
        # 特征8: 编码模式数量
        features['hex_encoding'] = len(re.findall(r'\\x[0-9a-fA-F]{2}', text))
        features['html_entity'] = len(re.findall(r'&#[0-9]+;', text))
        features['html_hex_entity'] = len(re.findall(r'&#x[0-9a-fA-F]+', text))
        
        return list(features.values())
    
    def build_tfidf_features(self, texts, fit=True):
        """构建TF-IDF特征"""
        if fit:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.vocab_size,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            tfidf_features = self.tfidf_vectorizer.fit_transform(texts)
        else:
            if self.tfidf_vectorizer is None:
                raise ValueError("TF-IDF向量化器未训练")
            tfidf_features = self.tfidf_vectorizer.transform(texts)
        
        return tfidf_features
    
    def build_combined_features(self, texts, fit=True):
        """构建组合特征（TF-IDF + 手工特征）"""
        # TF-IDF特征
        tfidf_features = self.build_tfidf_features(texts, fit=fit)
        
        # 手工特征
        manual_features = np.array([self.extract_features(text) for text in texts])
        
        # 组合特征
        combined_features = np.hstack([tfidf_features.toarray(), manual_features])
        
        return combined_features
    
    def tokenize_for_deep_learning(self, texts):
        """为深度学习模型进行分词"""
        # 构建词汇表
        if not self.word_to_idx:
            word_freq = {}
            for text in texts:
                # 简单的字符级别分词
                for char in text:
                    word_freq[char] = word_freq.get(char, 0) + 1
            
            # 选择最常见的字符
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
            for word, _ in sorted_words[:self.vocab_size-2]:
                self.word_to_idx[word] = len(self.word_to_idx)
            
            self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}
        
        # 转换文本为序列
        sequences = []
        for text in texts:
            seq = [self.word_to_idx.get(char, self.word_to_idx['<UNK>']) for char in text]
            # 截断或填充
            if len(seq) > self.max_length:
                seq = seq[:self.max_length]
            else:
                seq = seq + [self.word_to_idx['<PAD>']] * (self.max_length - len(seq))
            sequences.append(seq)
        
        return np.array(sequences)
    
    def generate_sample_data(self, n_samples=1000):
        """生成示例数据用于测试"""
        xss_samples = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "<body onload=alert('xss')>",
            "javascript:alert('xss')",
            "<svg/onload=alert('xss')>",
            "<iframe src='javascript:alert(1)'>",
            "<input onfocus=alert('xss') autofocus>",
            "<details open ontoggle=alert('xss')>",
            "<marquee onstart=alert('xss')>",
            "eval('alert(\"xss\")')",
            "<div onclick=\"alert('xss')\">Click</div>",
            "<a href='javascript:alert(1)'>Link</a>",
            "<object data='javascript:alert(1)'>",
            "<embed src='javascript:alert(1)'>",
            "<form><button formaction=javascript:alert(1)>Click</button></form>",
            "document.cookie",
            "\\x3cscript\\x3ealert('xss')\\x3c/script\\x3e",
            "<script>alert(String.fromCharCode(88,83,83))</script>",
            "<img src=x onerror=\"javascript:alert('xss')\">",
            "<body onkeypress=alert('xss')>",
        ]
        
        normal_samples = [
            "Hello world",
            "This is a normal text",
            "Check out this link: https://example.com",
            "User name: John",
            "Password: 123456",
            "Email: test@example.com",
            "Normal HTML content",
            "<div>This is normal content</div>",
            "Regular string with < > symbols",
            "Some text with numbers 12345",
            "Testing the system",
            "No XSS here",
            "Just a regular input",
            "Check this out!",
            "Normal user comment",
        ]
        
        # 扩充数据集
        xss_data = []
        normal_data = []
        
        for _ in range(n_samples // 2):
            # 随机选择XSS样本并添加变异
            base = xss_samples[_ % len(xss_samples)]
            xss_data.append(base)
            # 添加一些简单的变异
            if _ % 3 == 0:
                xss_data.append(base.upper())
            elif _ % 3 == 1:
                xss_data.append(base + " extra text")
        
        for _ in range(n_samples // 2):
            base = normal_samples[_ % len(normal_samples)]
            normal_data.append(base)
            if _ % 3 == 0:
                normal_data.append(base + " " + str(_))
            elif _ % 3 == 1:
                normal_data.append(base.upper())
        
        # 创建DataFrame
        xss_df = pd.DataFrame({
            'payload': xss_data,
            'label': 1
        })
        
        normal_df = pd.DataFrame({
            'payload': normal_data,
            'label': 0
        })
        
        df = pd.concat([xss_df, normal_df], ignore_index=True)
        df = df.sample(frac=1).reset_index(drop=True)  # 打乱顺序
        
        return df
    
    def load_data_from_csv(self, filepath):
        """从CSV文件加载数据"""
        try:
            df = pd.read_csv(filepath)
            if 'payload' in df.columns and 'label' in df.columns:
                return df
            elif 'text' in df.columns and 'label' in df.columns:
                df.rename(columns={'text': 'payload'}, inplace=True)
                return df
            else:
                raise ValueError("CSV文件必须包含'payload'和'label'列")
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            return None
    
    def prepare_training_data(self, df=None, test_size=0.2, random_state=42):
        """准备训练数据"""
        if df is None:
            logger.info("生成示例数据...")
            df = self.generate_sample_data(n_samples=2000)
        
        # 清理文本
        df['cleaned_payload'] = df['payload'].apply(self.clean_text)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            df['cleaned_payload'].values,
            df['label'].values,
            test_size=test_size,
            random_state=random_state,
            stratify=df['label'].values
        )
        
        logger.info(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
        logger.info(f"XSS样本: {sum(y_train)}, 正常样本: {len(y_train) - sum(y_train)}")
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessor(self, filepath):
        """保存预处理器"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'word_to_idx': self.word_to_idx,
                'idx_to_word': self.idx_to_word,
                'max_length': self.max_length,
                'vocab_size': self.vocab_size
            }, f)
        logger.info(f"预处理器已保存到 {filepath}")
    
    def load_preprocessor(self, filepath):
        """加载预处理器"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.tfidf_vectorizer = data['tfidf_vectorizer']
                self.word_to_idx = data['word_to_idx']
                self.idx_to_word = data['idx_to_word']
                self.max_length = data['max_length']
                self.vocab_size = data['vocab_size']
            logger.info(f"预处理器已从 {filepath} 加载")
            return True
        except Exception as e:
            logger.error(f"加载预处理器失败: {e}")
            return False
