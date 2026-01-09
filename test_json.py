import json
from database import db, init_database
from decimal import Decimal
import datetime

# 初始化数据库
init_database()

# 查询历史
result = db.get_detection_history_with_pagination(page=1, page_size=2)

print("=== 测试JSON序列化 ===\n")

# 手动序列化（模拟Flask jsonify）
def custom_json_encoder(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, datetime.datetime):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

# 尝试序列化
try:
    # 逐个序列化每个字段
    serialized_data = []
    for item in result['data']:
        serialized_item = {
            'id': item['id'],
            'input_text': item['input_text'],
            'is_xss': item['is_xss'],
            'detection_time': custom_json_encoder(item['detection_time']),
            'ensemble_prob': custom_json_encoder(item['ensemble_prob'])
        }
        serialized_data.append(serialized_item)

    serialized_result = {
        'data': serialized_data,
        'pagination': result['pagination']
    }

    print("序列化成功:")
    print(json.dumps(serialized_result, indent=2, ensure_ascii=False))

except Exception as e:
    print(f"序列化失败: {e}")
    import traceback
    traceback.print_exc()
