from ensemble import detector
from database import db, init_database
import json

# 初始化
init_database()
detector.load_models()

# 测试检测
test_text = "hello"
print(f"检测文本: {test_text}")

result = detector.detect(test_text)
print(f"检测结果: {json.dumps(result, indent=2, ensure_ascii=False)}")

# 保存到数据库
print("\n保存到数据库...")
try:
    record_id = db.insert_detection_record(
        input_text=test_text,
        is_xss=result['ensemble']['prediction'],
        xgboost_prob=result['xgboost']['probability'],
        bilstm_prob=result['bilstm']['probability'],
        transformer_prob=result['transformer']['probability'],
        ensemble_prob=result['ensemble']['probability']
    )
    print(f"记录已保存，ID: {record_id}")
except Exception as e:
    print(f"保存失败: {e}")
    import traceback
    traceback.print_exc()

# 查询历史
print("\n查询历史记录...")
history = db.get_detection_history(limit=5)
print(f"找到 {len(history)} 条最新记录:")
for record in history:
    print(f"  - {record['input_text'][:30]} | XSS={record['is_xss']} | {record['detection_time']}")

# 查询统计
print("\n统计信息:")
stats = db.get_statistics()
print(json.dumps(stats, indent=2, ensure_ascii=False))
