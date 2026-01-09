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
                        training_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        model_path VARCHAR(255),
                        UNIQUE KEY unique_model (model_name, model_version)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """)
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
