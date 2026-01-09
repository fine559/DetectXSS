from database import db, init_database
from ensemble import detector

# 初始化数据库
init_database()

# 测试插入记录
print("测试插入检测记录...")
test_cases = [
    ("<script>alert('xss')</script>", True, 0.95, 0.92, 0.97, 0.95),
    ("Hello world", False, 0.05, 0.08, 0.03, 0.05),
    ("<img src=x onerror=alert(1)>", True, 0.93, 0.95, 0.98, 0.96),
]

for text, is_xss, xgb, bilstm, transformer, ensemble in test_cases:
    record_id = db.insert_detection_record(
        input_text=text,
        is_xss=1 if is_xss else 0,
        xgboost_prob=xgb,
        bilstm_prob=bilstm,
        transformer_prob=transformer,
        ensemble_prob=ensemble
    )
    print(f"插入记录: ID={record_id}, 文本={text[:30]}")

# 查询历史记录
print("\n查询检测历史...")
history = db.get_detection_history(limit=10)
print(f"找到 {len(history)} 条记录:")
for record in history:
    print(f"  - ID={record['id']}, 文本={record['input_text'][:30]}, XSS={record['is_xss']}, 概率={record['ensemble_prob']}")

# 查询统计
print("\n查询统计数据...")
stats = db.get_statistics()
print(f"总检测次数: {stats['total']}")
print(f"XSS攻击次数: {stats['xss_count']}")
print(f"正常请求次数: {stats['normal_count']}")

print("\n数据库测试完成!")
