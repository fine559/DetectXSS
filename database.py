import pymysql
from pymysql.cursors import DictCursor
from dbutils.pooled_db import PooledDB
import logging
from decimal import Decimal
import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Database:
    """数据库连接管理类 - 使用连接池"""

    def __init__(self, host='localhost', user='root', password='123456', database='xss_detection'):
        self.connection_pool = PooledDB(
            creator=pymysql,
            host=host,
            user=user,
            password=password,
            database=database,
            maxconnections=10,
            ping=1,
            cursorclass=DictCursor,
            charset='utf8mb4'
        )
        logger.info("数据库连接池初始化成功")

    def get_connection(self):
        """从连接池获取一个连接"""
        return self.connection_pool.connection()

    def create_database(self):
        """创建数据库"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS xss_detection CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
                conn.commit()
                logger.info("数据库创建成功或已存在")
            return True
        except Exception as e:
            logger.error(f"创建数据库失败: {e}")
            return False
        finally:
            conn.close()

    def create_tables(self):
        """创建数据表"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS detection_records (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        input_text TEXT NOT NULL,
                        is_xss INT NOT NULL,
                        xgboost_prob DECIMAL(5,4),
                        bilstm_prob DECIMAL(5,4),
                        transformer_prob DECIMAL(5,4),
                        ensemble_prob DECIMAL(5,4),
                        detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_detection_time (detection_time),
                        INDEX idx_is_xss (is_xss)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS training_data (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        payload TEXT NOT NULL,
                        label INT NOT NULL COMMENT '0: 正常, 1: XSS',
                        source VARCHAR(100) DEFAULT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_label (label)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS model_info (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        model_name VARCHAR(50) NOT NULL,
                        model_version VARCHAR(20) NOT NULL,
                        accuracy DECIMAL(5,4),
                        precision_score DECIMAL(5,4),
                        recall_score DECIMAL(5,4),
                        f1_score DECIMAL(5,4),
                        auc_score DECIMAL(5,4),
                        training_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        model_path VARCHAR(255),
                        UNIQUE KEY unique_model (model_name, model_version)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS training_history (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        model_name VARCHAR(50) NOT NULL,
                        epoch INT NOT NULL,
                        loss DECIMAL(10,6),
                        accuracy DECIMAL(5,4),
                        val_loss DECIMAL(10,6),
                        val_accuracy DECIMAL(5,4),
                        training_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_model_name (model_name),
                        INDEX idx_epoch (epoch)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """)

                # 检查并添加缺失的列（用于数据库迁移）
                cursor.execute("SHOW COLUMNS FROM model_info LIKE 'auc_score'")
                if not cursor.fetchone():
                    cursor.execute("ALTER TABLE model_info ADD COLUMN auc_score DECIMAL(5,4) AFTER f1_score")
                    logger.info("添加 auc_score 列到 model_info 表")

                conn.commit()
                logger.info("数据表创建成功")
            return True
        except Exception as e:
            logger.error(f"创建数据表失败: {e}")
            return False
        finally:
            conn.close()

    def insert_detection_record(self, input_text, is_xss, xgboost_prob, bilstm_prob, transformer_prob, ensemble_prob):
        """插入检测记录"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                sql = """
                    INSERT INTO detection_records
                    (input_text, is_xss, xgboost_prob, bilstm_prob, transformer_prob, ensemble_prob)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (input_text, is_xss, xgboost_prob, bilstm_prob, transformer_prob, ensemble_prob))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"插入检测记录失败: {e}")
            return None
        finally:
            conn.close()

    def get_detection_history_with_pagination(self, page=1, page_size=10):
        """获取分页的检测历史记录"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) as total FROM detection_records")
                total = int(cursor.fetchone()['total'])

                offset = (page - 1) * page_size

                sql = """
                    SELECT id, input_text, is_xss, xgboost_prob, bilstm_prob, transformer_prob,
                           ensemble_prob, detection_time
                    FROM detection_records
                    ORDER BY detection_time DESC
                    LIMIT %s OFFSET %s
                """
                cursor.execute(sql, (page_size, offset))
                data = cursor.fetchall()

                for item in data:
                    for key, value in item.items():
                        if isinstance(value, Decimal):
                            item[key] = float(value)
                        elif isinstance(value, datetime.datetime):
                            item[key] = value.strftime('%Y-%m-%d %H:%M:%S')

                total_pages = (total + page_size - 1) // page_size

                return {
                    'data': data,
                    'pagination': {
                        'total': total,
                        'page': page,
                        'page_size': page_size,
                        'total_pages': total_pages,
                        'has_prev': page > 1,
                        'has_next': page < total_pages
                    }
                }
        except Exception as e:
            logger.error(f"获取分页检测历史失败: {e}")
            return {
                'data': [],
                'pagination': {
                    'total': 0,
                    'page': page,
                    'page_size': page_size,
                    'total_pages': 0,
                    'has_prev': False,
                    'has_next': False
                }
            }
        finally:
            conn.close()

    def get_statistics(self):
        """获取统计数据"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) as total FROM detection_records")
                total = int(cursor.fetchone()['total'])

                cursor.execute("SELECT COUNT(*) as xss_count FROM detection_records WHERE is_xss = 1")
                xss_count = int(cursor.fetchone()['xss_count'])

                normal_count = total - xss_count

                return {
                    'total': total,
                    'xss_count': xss_count,
                    'normal_count': normal_count
                }
        except Exception as e:
            logger.error(f"获取统计数据失败: {e}")
            return {'total': 0, 'xss_count': 0, 'normal_count': 0}
        finally:
            conn.close()

    def get_dashboard_data(self, days=7):
        """获取仪表盘数据"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                # 获取最近N天的检测趋势（按日期降序，最近的在前）
                cursor.execute("""
                    SELECT DATE(detection_time) as date,
                           COUNT(*) as total,
                           SUM(CASE WHEN is_xss = 1 THEN 1 ELSE 0 END) as xss_count
                    FROM detection_records
                    WHERE DATE(detection_time) >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
                    GROUP BY DATE(detection_time)
                    ORDER BY date DESC
                """, (days,))
                trend_data = cursor.fetchall()

                # 将日期对象转换为字符串，将Decimal转换为float
                for item in trend_data:
                    item['date'] = str(item['date'])
                    if 'total' in item:
                        item['total'] = int(item['total'])
                    if 'xss_count' in item and isinstance(item['xss_count'], Decimal):
                        item['xss_count'] = int(item['xss_count'])

                # 获取模型性能统计
                cursor.execute("""
                    SELECT
                        AVG(xgboost_prob) as avg_xgboost,
                        AVG(bilstm_prob) as avg_bilstm,
                        AVG(transformer_prob) as avg_transformer,
                        AVG(ensemble_prob) as avg_ensemble
                    FROM detection_records
                    WHERE DATE(detection_time) >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
                """, (days,))
                model_stats = cursor.fetchone()

                # 获取最近24小时检测统计
                cursor.execute("""
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN is_xss = 1 THEN 1 ELSE 0 END) as xss_count
                    FROM detection_records
                    WHERE detection_time >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
                """)
                hour_stats = cursor.fetchone()

                return {
                    'trend': trend_data,
                    'model_stats': {
                        'xgboost': float(model_stats['avg_xgboost'] or 0),
                        'bilstm': float(model_stats['avg_bilstm'] or 0),
                        'transformer': float(model_stats['avg_transformer'] or 0),
                        'ensemble': float(model_stats['avg_ensemble'] or 0)
                    },
                    'hour_stats': {
                        'total': int(hour_stats['total'] or 0),
                        'xss_count': int(hour_stats['xss_count'] or 0)
                    }
                }
        except Exception as e:
            logger.error(f"获取仪表盘数据失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'trend': [],
                'model_stats': {'xgboost': 0, 'bilstm': 0, 'transformer': 0, 'ensemble': 0},
                'hour_stats': {'total': 0, 'xss_count': 0}
            }
        finally:
            conn.close()


    def get_model_metrics(self):
        """获取各模型的评估指标"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                # 获取最新的模型指标
                cursor.execute("""
                    SELECT model_name, model_version, accuracy, precision_score,
                           recall_score, f1_score, auc_score, training_time
                    FROM model_info
                    ORDER BY training_time DESC
                """)
                models_data = cursor.fetchall()

                # 按模型名称分组，取最新版本
                model_metrics = {}
                for item in models_data:
                    model_name = item['model_name']
                    if model_name not in model_metrics:
                        model_metrics[model_name] = {
                            'model_name': model_name,
                            'model_version': item['model_version'],
                            'accuracy': float(item['accuracy'] or 0),
                            'precision_score': float(item['precision_score'] or 0),
                            'recall_score': float(item['recall_score'] or 0),
                            'f1_score': float(item['f1_score'] or 0),
                            'auc_score': float(item['auc_score'] or 0) if item['auc_score'] else None,
                            'training_time': item['training_time'].strftime('%Y-%m-%d %H:%M:%S')
                        }

                return list(model_metrics.values())
        except Exception as e:
            logger.error(f"获取模型指标失败: {e}")
            return []
        finally:
            conn.close()


    def get_training_history(self, model_name, limit=100):
        """获取模型训练历史"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT epoch, loss, accuracy, val_loss, val_accuracy, training_time
                    FROM training_history
                    WHERE model_name = %s
                    ORDER BY epoch ASC
                    LIMIT %s
                """, (model_name, limit))
                history = cursor.fetchall()

                return [{
                    'epoch': int(item['epoch']),
                    'loss': float(item['loss']) if item['loss'] else None,
                    'accuracy': float(item['accuracy']) if item['accuracy'] else None,
                    'val_loss': float(item['val_loss']) if item['val_loss'] else None,
                    'val_accuracy': float(item['val_accuracy']) if item['val_accuracy'] else None
                } for item in history]
        except Exception as e:
            logger.error(f"获取训练历史失败: {e}")
            return []
        finally:
            conn.close()

    def insert_sample_model_metrics(self, force=False):
        """插入示例模型指标数据"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                # 检查是否已有数据
                cursor.execute("SELECT COUNT(*) as count FROM model_info")
                count = cursor.fetchone()['count']
                if count > 0 and not force:
                    logger.info("模型指标数据已存在，跳过插入")
                    return True

                # 如果强制刷新，先清空旧数据
                if force:
                    cursor.execute("DELETE FROM model_info")
                    logger.info("清空旧模型指标数据")

                # 插入示例数据
                sample_data = [
                    ('xgboost', '1.0', 0.9234, 0.9156, 0.9345, 0.9250, 0.9687, 'models/xgboost_model.pkl'),
                    ('bilstm', '1.0', 0.9456, 0.9389, 0.9489, 0.9438, 0.9805, 'models/bilstm_model.h5'),
                    ('transformer', '1.0', 0.9567, 0.9512, 0.9601, 0.9556, 0.9878, 'models/transformer_model.h5'),
                    ('ensemble', '1.0', 0.9678, 0.9634, 0.9712, 0.9673, 0.9923, 'models/ensemble_model.json')
                ]

                for model_name, version, acc, prec, rec, f1, auc, path in sample_data:
                    cursor.execute("""
                        INSERT INTO model_info (model_name, model_version, accuracy, precision_score,
                                               recall_score, f1_score, auc_score, model_path)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (model_name, version, acc, prec, rec, f1, auc, path))

                conn.commit()
                logger.info("示例模型指标数据插入成功")
                return True
        except Exception as e:
            logger.error(f"插入示例模型指标失败: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()


    def insert_sample_training_history(self, force=False):
        """插入示例训练历史数据"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                # 检查是否已有数据
                cursor.execute("SELECT COUNT(*) as count FROM training_history")
                count = cursor.fetchone()['count']
                if count > 0 and not force:
                    logger.info("训练历史数据已存在，跳过插入")
                    return True

                # 如果强制刷新，先清空旧数据
                if force:
                    cursor.execute("DELETE FROM training_history")
                    logger.info("清空旧训练历史数据")

                # 为每个模型生成训练历史
                import random
                models = ['xgboost', 'bilstm', 'transformer', 'ensemble']

                for model_name in models:
                    # 模拟20个epoch的训练数据
                    for epoch in range(1, 21):
                        # 模拟loss逐渐下降，accuracy逐渐上升
                        initial_loss = 0.8 if model_name != 'ensemble' else 0.6
                        final_loss = 0.2 if model_name != 'ensemble' else 0.1
                        loss = initial_loss * (0.95 ** (epoch - 1))

                        initial_acc = 0.6 if model_name != 'ensemble' else 0.7
                        final_acc = 0.92 if model_name != 'ensemble' else 0.97
                        accuracy = initial_acc + (final_acc - initial_acc) * (epoch / 20)

                        # 添加一些随机波动
                        loss += random.uniform(-0.02, 0.02)
                        accuracy += random.uniform(-0.01, 0.01)
                        loss = max(0.05, min(0.99, loss))
                        accuracy = max(0.5, min(0.99, accuracy))

                        # 验证数据略好于训练数据
                        val_loss = loss * random.uniform(0.95, 1.1)
                        val_accuracy = accuracy * random.uniform(0.98, 1.02)
                        val_accuracy = min(0.99, val_accuracy)

                        cursor.execute("""
                            INSERT INTO training_history (model_name, epoch, loss, accuracy,
                                                         val_loss, val_accuracy)
                            VALUES (%s, %s, %s, %s, %s, %s)
                        """, (model_name, epoch, loss, accuracy, val_loss, val_accuracy))

                conn.commit()
                logger.info("示例训练历史数据插入成功")
                return True
        except Exception as e:
            logger.error(f"插入示例训练历史失败: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()



    def insert_model_metrics(self, model_name, model_version, metrics):
        """插入模型评估指标"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                sql = """
                    INSERT INTO model_info
                    (model_name, model_version, accuracy, precision_score, recall_score, f1_score, auc_score)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        accuracy = VALUES(accuracy),
                        precision_score = VALUES(precision_score),
                        recall_score = VALUES(recall_score),
                        f1_score = VALUES(f1_score),
                        auc_score = VALUES(auc_score)
                """
                cursor.execute(sql, (
                    model_name,
                    model_version,
                    metrics.get('accuracy'),
                    metrics.get('precision'),
                    metrics.get('recall'),
                    metrics.get('f1'),
                    metrics.get('auc')
                ))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"插入模型指标失败: {e}")
            return None
        finally:
            conn.close()

    def insert_training_history(self, model_name, epoch, history_data):
        """插入训练历史记录"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                sql = """
                    INSERT INTO training_history
                    (model_name, epoch, loss, accuracy, val_loss, val_accuracy)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (
                    model_name,
                    epoch,
                    history_data.get('loss'),
                    history_data.get('accuracy'),
                    history_data.get('val_loss'),
                    history_data.get('val_accuracy')
                ))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"插入训练历史失败: {e}")
            return None
        finally:
            conn.close()

    def close(self):
        """关闭数据库连接池"""
        if self.connection_pool:
            self.connection_pool.close()
            logger.info("数据库连接池已关闭")


# 创建数据库实例
db = Database()


def init_database():
    """初始化数据库"""
    try:
        db.create_database()
        db.create_tables()
        return True
    except Exception as e:
        logger.error(f"初始化数据库失败: {e}")
        return False
