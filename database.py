import pymysql
from pymysql.cursors import DictCursor
import logging
from decimal import Decimal
import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Database:
    """数据库连接管理类"""

    def __init__(self, host='localhost', user='root', password='123456', database='xss_detection'):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        self.connect()

    def ensure_connection(self):
        """确保数据库连接有效"""
        try:
            if self.connection is None or not self.connection.open:
                logger.info("数据库连接已断开，重新连接...")
                self.connect()
                self.connection.select_db(self.database)
        except Exception as e:
            logger.error(f"数据库连接检查失败: {e}")
            try:
                if self.connection and self.connection.open:
                    self.connection.close()
            except:
                pass
            self.connect()
            self.connection.select_db(self.database)

    def connect(self):
        """连接数据库"""
        try:
            self.connection = pymysql.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                charset='utf8mb4',
                cursorclass=DictCursor
            )
            logger.info("数据库连接成功")
            return True
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            return False
    
    def create_database(self):
        """创建数据库"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
                self.connection.commit()
                logger.info(f"数据库 {self.database} 创建成功或已存在")
            # 切换到指定数据库
            self.connection.select_db(self.database)
            return True
        except Exception as e:
            logger.error(f"创建数据库失败: {e}")
            return False
    
    def create_tables(self):
        """创建数据表"""
        try:
            with self.connection.cursor() as cursor:
                # XSS检测记录表
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
                
                # 训练数据集表
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
                
                # 模型信息表
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
                
                self.connection.commit()
                logger.info("数据表创建成功")
                return True
        except Exception as e:
            logger.error(f"创建数据表失败: {e}")
            return False
    
    def insert_detection_record(self, input_text, is_xss, xgboost_prob, bilstm_prob, transformer_prob, ensemble_prob):
        """插入检测记录"""
        try:
            self.ensure_connection()
            with self.connection.cursor() as cursor:
                sql = """
                    INSERT INTO detection_records
                    (input_text, is_xss, xgboost_prob, bilstm_prob, transformer_prob, ensemble_prob)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (input_text, is_xss, xgboost_prob, bilstm_prob, transformer_prob, ensemble_prob))
                self.connection.commit()
                return cursor.lastrowid
        except pymysql.MySQLError as e:
            logger.error(f"插入检测记录失败 (数据库错误): {e}")
            # 连接可能已断开，尝试重连并重试
            self.ensure_connection()
            try:
                with self.connection.cursor() as cursor:
                    cursor.execute(sql, (input_text, is_xss, xgboost_prob, bilstm_prob, transformer_prob, ensemble_prob))
                    self.connection.commit()
                    return cursor.lastrowid
            except:
                return None
        except Exception as e:
            logger.error(f"插入检测记录失败: {e}")
            return None
    
    def get_detection_history(self, limit=100):
        """获取检测历史记录"""
        try:
            self.ensure_connection()
            with self.connection.cursor() as cursor:
                sql = """
                    SELECT id, input_text, is_xss, xgboost_prob, bilstm_prob, transformer_prob,
                           ensemble_prob, detection_time
                    FROM detection_records
                    ORDER BY detection_time DESC
                    LIMIT %s
                """
                cursor.execute(sql, (limit,))
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"获取检测历史失败: {e}")
            return []
    
    def get_detection_history_with_pagination(self, page=1, page_size=10):
        """获取分页的检测历史记录"""
        try:
            self.ensure_connection()
            with self.connection.cursor() as cursor:
                # 获取总数
                cursor.execute("SELECT COUNT(*) as total FROM detection_records")
                total = cursor.fetchone()['total']

                # 计算偏移量
                offset = (page - 1) * page_size

                # 获取数据
                sql = """
                    SELECT id, input_text, is_xss, xgboost_prob, bilstm_prob, transformer_prob,
                           ensemble_prob, detection_time
                    FROM detection_records
                    ORDER BY detection_time DESC
                    LIMIT %s OFFSET %s
                """
                cursor.execute(sql, (page_size, offset))
                data = cursor.fetchall()

                # 转换Decimal和datetime为JSON可序列化的类型
                for item in data:
                    for key, value in item.items():
                        if isinstance(value, Decimal):
                            item[key] = float(value)
                        elif isinstance(value, datetime.datetime):
                            item[key] = value.strftime('%Y-%m-%d %H:%M:%S')

                # 计算总页数
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
    
    def get_statistics(self):
        """获取统计数据"""
        try:
            self.ensure_connection()
            with self.connection.cursor() as cursor:
                # 总检测次数
                cursor.execute("SELECT COUNT(*) as total FROM detection_records")
                total = int(cursor.fetchone()['total'])

                # XSS攻击次数
                cursor.execute("SELECT COUNT(*) as xss_count FROM detection_records WHERE is_xss = 1")
                xss_count = int(cursor.fetchone()['xss_count'])

                # 正常请求次数
                normal_count = total - xss_count

                return {
                    'total': total,
                    'xss_count': xss_count,
                    'normal_count': normal_count
                }
        except Exception as e:
            logger.error(f"获取统计数据失败: {e}")
            return {'total': 0, 'xss_count': 0, 'normal_count': 0}
    
    def close(self):
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
            logger.info("数据库连接已关闭")


# 创建数据库实例
db = Database()


def init_database():
    """初始化数据库"""
    if db.connect():
        db.create_database()
        db.create_tables()
        return True
    return False
