from database import db, init_database
import json

# 初始化数据库
init_database()

# 查询第一页（10条/页）
print("=== 测试分页功能 ===\n")

print("1. 查询第1页:")
result = db.get_detection_history_with_pagination(page=1, page_size=10)
print(f"   总记录数: {result['pagination']['total']}")
print(f"   当前页: {result['pagination']['page']}")
print(f"   每页数量: {result['pagination']['page_size']}")
print(f"   总页数: {result['pagination']['total_pages']}")
print(f"   记录数: {len(result['data'])}")

if result['data']:
    print("\n   前3条记录:")
    for item in result['data'][:3]:
        print(f"     - {item['input_text'][:30]} | XSS={item['is_xss']}")

# 如果总记录数超过10条，查询第2页
if result['pagination']['total_pages'] >= 2:
    print("\n2. 查询第2页:")
    result2 = db.get_detection_history_with_pagination(page=2, page_size=10)
    print(f"   记录数: {len(result2['data'])}")
    if result2['data']:
        print("\n   前3条记录:")
        for item in result2['data'][:3]:
            print(f"     - {item['input_text'][:30]} | XSS={item['is_xss']}")

print("\n=== 测试完成 ===")
