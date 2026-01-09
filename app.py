from flask import Flask, render_template, request, jsonify
from database import init_database, db
from ensemble import init_detector, detect_xss
import logging
import json

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


@app.route('/api/health')
def health():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'message': 'XSS检测系统运行正常'
    })


@app.route('/train')
def train_page():
    """训练页面"""
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
    
    # 启动Flask应用
    app.run(debug=True, host='0.0.0.0', port=5000)
