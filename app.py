from flask import Flask, render_template, request, jsonify
from database import init_database, db
from ensemble import init_detector, detect_xss
import logging
import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'


# 初始化应用
def initialize_app():
    """初始化应用"""
    logger.info("正在初始化应用...")

    # 初始化数据库
    if init_database():
        logger.info("数据库初始化成功")
    else:
        logger.warning("数据库初始化失败")

    # 初始化示例数据（只在数据为空时插入）
    try:
        db.insert_sample_model_metrics()
        db.insert_sample_training_history()
    except Exception as e:
        logger.warning(f"示例数据初始化失败: {e}")

    # 导入Kaggle数据集（只在没有数据时导入一次）
    try:
        csv_path = 'models/data/XSS_dataset.csv'
        db.import_dataset_from_csv(csv_path)
        # 根据实际数据集更新模型指标
        db.update_model_metrics_from_dataset()
    except Exception as e:
        logger.warning(f"导入Kaggle数据集失败: {e}")

    # 初始化检测器（如果模型已训练）
    try:
        init_detector()
        logger.info("模型加载成功")
    except Exception as e:
        logger.warning(f"模型加载失败: {e} (可能需要先训练模型)")

    logger.info("应用初始化完成")



@app.route('/')
def index():
    """首页"""
    return render_template('index.html')


@app.route('/test')
def test():
    """测试页面"""
    return render_template('test.html')


@app.route('/api/detect', methods=['POST'])
def detect():
    """XSS检测API"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': '请提供待检测的文本'}), 400
        
        text = data['text']
        
        if not text or not text.strip():
            return jsonify({'error': '文本不能为空'}), 400
        
        # 限制文本长度
        if len(text) > 5000:
            return jsonify({'error': '文本长度超过限制（最多5000字符）'}), 400
        
        # 执行检测
        result = detect_xss(text)
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 500
        
        # 保存检测记录到数据库
        try:
            record_id = db.insert_detection_record(
                input_text=text,
                is_xss=result['ensemble']['prediction'],
                xgboost_prob=result['xgboost']['probability'],
                bilstm_prob=result['bilstm']['probability'],
                transformer_prob=result['transformer']['probability'],
                ensemble_prob=result['ensemble']['probability']
            )
            logger.info(f"检测记录已保存，ID: {record_id}")
        except Exception as db_error:
            logger.error(f"保存检测记录失败: {db_error}")
            # 不影响检测结果的返回
        
        # 返回检测结果
        response = {
            'is_xss': result['is_xss'],
            'xgboost_prob': result['xgboost']['probability'],
            'bilstm_prob': result['bilstm']['probability'],
            'transformer_prob': result['transformer']['probability'],
            'ensemble_prob': result['ensemble']['probability'],
            'message': 'XSS攻击' if result['is_xss'] else '正常'
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"检测失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/statistics')
def statistics():
    """获取统计数据"""
    try:
        stats = db.get_statistics()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"获取统计数据失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch-detect', methods=['POST'])
def batch_detect():
    """批量检测API"""
    try:
        data = request.get_json()

        if not data or 'texts' not in data:
            return jsonify({'error': '请提供待检测的文本列表'}), 400

        texts = data['texts']

        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({'error': '文本列表不能为空'}), 400

        if len(texts) > 100:
            return jsonify({'error': '批量检测最多支持100条文本'}), 400

        # 限制每条文本长度
        for text in texts:
            if len(text) > 5000:
                return jsonify({'error': '单条文本长度超过限制（最多5000字符）'}), 400

        from ensemble import detector
        results = detector.detect_batch(texts)

        if not results:
            return jsonify({'error': '批量检测失败'}), 500

        # 保存检测记录
        saved_count = 0
        for result in results:
            try:
                db.insert_detection_record(
                    input_text=result['text'],
                    is_xss=result['is_xss'],
                    xgboost_prob=result['xgboost_prob'],
                    bilstm_prob=result['bilstm_prob'],
                    transformer_prob=result['transformer_prob'],
                    ensemble_prob=result['ensemble_prob']
                )
                saved_count += 1
            except Exception as db_error:
                logger.error(f"保存检测记录失败: {db_error}")

        return jsonify({
            'total': len(results),
            'xss_count': sum(1 for r in results if r['is_xss']),
            'normal_count': sum(1 for r in results if not r['is_xss']),
            'saved_count': saved_count,
            'results': results
        })

    except Exception as e:
        logger.error(f"批量检测失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/rules')
def get_rules():
    """获取检测规则配置"""
    try:
        rules = {
            'models': [
                {
                    'name': 'XGBoost',
                    'weight': 0.3,
                    'description': '梯度提升树模型，擅长特征提取',
                    'color': '#667eea'
                },
                {
                    'name': 'BiLSTM',
                    'weight': 0.35,
                    'description': '双向LSTM，擅长序列模式识别',
                    'color': '#764ba2'
                },
                {
                    'name': 'Transformer',
                    'weight': 0.35,
                    'description': '自注意力机制，擅长上下文理解',
                    'color': '#f093fb'
                }
            ],
            'threshold': 0.5,
            'method': 'weighted_average',
            'update_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        return jsonify(rules)
    except Exception as e:
        logger.error(f"获取规则配置失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/dashboard')
def dashboard():
    """获取仪表盘数据"""
    try:
        days = request.args.get('days', 7, type=int)
        data = db.get_dashboard_data(days=days)
        return jsonify(data)
    except Exception as e:
        logger.error(f"获取仪表盘数据失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/history')
def history():
    """获取检测历史"""
    try:
        page = request.args.get('page', 1, type=int)
        page_size = request.args.get('page_size', 10, type=int)
        
        history_data = db.get_detection_history_with_pagination(
            page=page,
            page_size=page_size
        )
        return jsonify(history_data)
    except Exception as e:
        logger.error(f"获取历史记录失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/detection-records/<int:record_id>', methods=['DELETE'])
def delete_detection_record(record_id):
    """删除单条检测记录"""
    try:
        success = db.delete_detection_record(record_id)
        if success:
            logger.info(f"已删除检测记录 ID: {record_id}")
            return jsonify({'success': True, 'message': '删除成功'})
        else:
            return jsonify({'success': False, 'error': '记录不存在'}), 404
    except Exception as e:
        logger.error(f"删除检测记录失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/detection-records/batch-delete', methods=['POST'])
def batch_delete_detection_records():
    """批量删除检测记录"""
    try:
        data = request.get_json()
        if not data or 'record_ids' not in data:
            return jsonify({'success': False, 'error': '请提供要删除的记录ID列表'}), 400
        
        record_ids = data['record_ids']
        if not isinstance(record_ids, list) or len(record_ids) == 0:
            return jsonify({'success': False, 'error': '记录ID列表不能为空'}), 400
        
        success = db.batch_delete_detection_records(record_ids)
        if success:
            logger.info(f"已批量删除 {len(record_ids)} 条检测记录")
            return jsonify({'success': True, 'message': f'成功删除 {len(record_ids)} 条记录'})
        else:
            return jsonify({'success': False, 'error': '批量删除失败'}), 500
    except Exception as e:
        logger.error(f"批量删除检测记录失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/health')
def health():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'message': 'XSS检测系统运行正常'
    })


@app.route('/api/model-metrics')
def model_metrics():
    """获取模型性能指标"""
    try:
        metrics = db.get_model_metrics()
        return jsonify({
            'success': True,
            'data': metrics
        })
    except Exception as e:
        logger.error(f"获取模型指标失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/confusion-matrix')
def confusion_matrix():
    """获取混淆矩阵数据"""
    try:
        conn = db.get_connection()
        try:
            with conn.cursor() as cursor:
                # 检查是否有检测记录
                cursor.execute("SELECT COUNT(*) as count FROM detection_records")
                result = cursor.fetchone()

                if result['count'] == 0:
                    # 没有检测记录，返回示例数据
                    return jsonify({
                        'success': True,
                        'data': {
                            'confusion_matrix': [[450, 50], [30, 470]],
                            'labels': {
                                'true_negative': 450,
                                'false_positive': 50,
                                'false_negative': 30,
                                'true_positive': 470
                            },
                            'is_sample': True
                        }
                    })

                # 计算混淆矩阵
                cursor.execute("""
                    SELECT
                        is_xss,
                        SUM(CASE WHEN ensemble_prob >= 0.5 THEN 1 ELSE 0 END) as predicted_xss,
                        SUM(CASE WHEN ensemble_prob < 0.5 THEN 1 ELSE 0 END) as predicted_normal
                    FROM detection_records
                    GROUP BY is_xss
                """)
                results = cursor.fetchall()

                # 构建混淆矩阵
                # TN: 预测为正常且实际为正常
                # FP: 预测为XSS但实际为正常
                # FN: 预测为正常但实际为XSS
                # TP: 预测为XSS且实际为XSS
                tn = fp = fn = tp = 0

                for row in results:
                    if row['is_xss'] == 0:  # 实际为正常
                        tn = int(row.get('predicted_normal', 0) or 0)
                        fp = int(row.get('predicted_xss', 0) or 0)
                    else:  # 实际为XSS
                        fn = int(row.get('predicted_normal', 0) or 0)
                        tp = int(row.get('predicted_xss', 0) or 0)

                return jsonify({
                    'success': True,
                    'data': {
                        'confusion_matrix': [[tn, fp], [fn, tp]],
                        'labels': {
                            'true_negative': tn,
                            'false_positive': fp,
                            'false_negative': fn,
                            'true_positive': tp
                        },
                        'is_sample': False
                    }
                })
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"获取混淆矩阵失败: {e}")
        import traceback
        traceback.print_exc()
        # 如果查询失败，返回示例数据
        return jsonify({
            'success': True,
            'data': {
                'confusion_matrix': [[450, 50], [30, 470]],
                'labels': {
                    'true_negative': 450,
                    'false_positive': 50,
                    'false_negative': 30,
                    'true_positive': 470
                },
                'is_sample': True
            }
        })


@app.route('/api/debug-model-info')
def debug_model_info():
    """调试：查看model_info表数据"""
    try:
        conn = db.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM model_info")
                data = cursor.fetchall()

                # 格式化时间
                for item in data:
                    for key, value in item.items():
                        if isinstance(value, datetime.datetime):
                            item[key] = value.strftime('%Y-%m-%d %H:%M:%S')

                return jsonify({
                    'success': True,
                    'count': len(data),
                    'data': data
                })
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"调试查询失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/dataset-stats')
def dataset_stats():
    """获取数据集统计信息"""
    try:
        stats = db.get_dataset_statistics()
        return jsonify({
            'success': True,
            'data': stats
        })
    except Exception as e:
        logger.error(f"获取数据集统计失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/training-history')
def training_history():
    """获取模型训练历史"""
    try:
        model_name = request.args.get('model_name', 'bilstm')
        limit = request.args.get('limit', 100, type=int)
        history = db.get_training_history(model_name, limit)
        return jsonify({
            'success': True,
            'data': history
        })
    except Exception as e:
        logger.error(f"获取训练历史失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/detect-detail', methods=['POST'])
def detect_detail():
    """详细检测分析"""
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({'error': '请提供待检测的文本'}), 400

        text = data['text']

        if not text or not text.strip():
            return jsonify({'error': '文本不能为空'}), 400

        # 执行检测
        result = detect_xss(text)

        if 'error' in result:
            return jsonify({'error': result['error']}), 500

        # 保存检测记录
        try:
            db.insert_detection_record(
                input_text=text,
                is_xss=result['ensemble']['prediction'],
                xgboost_prob=result['xgboost']['probability'],
                bilstm_prob=result['bilstm']['probability'],
                transformer_prob=result['transformer']['probability'],
                ensemble_prob=result['ensemble']['probability']
            )
        except Exception as db_error:
            logger.error(f"保存检测记录失败: {db_error}")

        # 计算风险等级
        ensemble_prob = result['ensemble']['probability']
        if ensemble_prob >= 0.8:
            risk_level = 'high'
            risk_label = '高风险'
        elif ensemble_prob >= 0.5:
            risk_level = 'medium'
            risk_label = '中风险'
        elif ensemble_prob >= 0.3:
            risk_level = 'low'
            risk_label = '低风险'
        else:
            risk_level = 'safe'
            risk_label = '安全'

        # 简单特征提取（高亮可能的危险字符）
        import re
        dangerous_patterns = [
            (r'<script[^>]*>', 'script标签'),
            (r'javascript:', 'javascript协议'),
            (r'on\w+\s*=', '事件处理器'),
            (r'alert\s*\(', 'alert函数'),
            (r'eval\s*\(', 'eval函数'),
            (r'document\.', 'document对象'),
            (r'window\.', 'window对象'),
            (r'<iframe[^>]*>', 'iframe标签'),
            (r'<embed[^>]*>', 'embed标签'),
            (r'<object[^>]*>', 'object标签'),
            (r'&lt;', 'HTML实体编码'),
            (r'&gt;', 'HTML实体编码'),
            (r'%3C', 'URL编码'),
            (r'%3E', 'URL编码'),
        ]

        highlighted_text = text
        matches = []
        for pattern, description in dangerous_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start, end = match.span()
                matches.append({
                    'start': start,
                    'end': end,
                    'text': text[start:end],
                    'type': description
                })

        response = {
            'is_xss': result['is_xss'],
            'risk_level': risk_level,
            'risk_label': risk_label,
            'xgboost_prob': result['xgboost']['probability'],
            'bilstm_prob': result['bilstm']['probability'],
            'transformer_prob': result['transformer']['probability'],
            'ensemble_prob': result['ensemble']['probability'],
            'highlighted_text': text,
            'matches': matches,
            'message': 'XSS攻击' if result['is_xss'] else '正常'
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"详细检测失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/analysis')
def analysis_page():
    """模型性能分析页面"""
    return render_template('analysis.html')


@app.route('/detail')
def detail_page():
    """检测结果详情页面"""
    return render_template('detail.html')


@app.route('/training')
def training_page():
    """模型训练页面"""
    return render_template('training.html')


@app.route('/train')
def train_page():
    """训练页面（保留兼容性）"""
    return render_template('train.html')


@app.errorhandler(404)
def not_found(error):
    """404错误处理"""
    return jsonify({'error': '接口不存在'}), 404


@app.errorhandler(500)
def internal_error(error):
    """500错误处理"""
    logger.error(f"内部错误: {error}")
    return jsonify({'error': '服务器内部错误'}), 500


if __name__ == '__main__':
    # 初始化应用
    initialize_app()

    # 启动Flask应用（关闭debug模式以避免重载问题）
    app.run(debug=False, host='0.0.0.0', port=5000)
