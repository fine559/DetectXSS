import sys
sys.path.insert(0, '.')

from database import db, init_database
import json

# 初始化数据库
init_database()

print("=== 检查API返回数据格式 ===\n")

# 查询历史记录
result = db.get_detection_history_with_pagination(page=1, page_size=3)

print("返回的完整数据结构:")
print(json.dumps(result, indent=2, ensure_ascii=False))

print("\n第一条记录的详细信息:")
if result['data']:
    first_item = result['data'][0]
    print(json.dumps(first_item, indent=2, ensure_ascii=False))

print("\n检查每条记录的字段:")
for i, item in enumerate(result['data'][:5]):
    print(f"记录 {i+1}:")
    print(f"  - id: {item.get('id', 'MISSING')}")
    print(f"  - input_text: {item.get('input_text', 'MISSING')}")
    print(f"  - is_xss: {item.get('is_xss', 'MISSING')}")
    print(f"  - detection_time: {item.get('detection_time', 'MISSING')}")
    print(f"  - ensemble_prob: {item.get('ensemble_prob', 'MISSING')}")
