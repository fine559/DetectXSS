from database import db, init_database

# 初始化数据库
init_database()

print("=== 测试数据库连接稳定性 ===\n")

# 测试1: 查询统计
print("1. 第一次查询统计:")
stats1 = db.get_statistics()
print(f"   {stats1}")

# 测试2: 插入记录
print("\n2. 插入测试记录:")
record_id = db.insert_detection_record(
    input_text="test_connection_" + str(__import__('time').time()),
    is_xss=0,
    xgboost_prob=0.1,
    bilstm_prob=0.1,
    transformer_prob=0.1,
    ensemble_prob=0.1
)
print(f"   插入记录ID: {record_id}")

# 测试3: 再次查询统计
print("\n3. 第二次查询统计:")
stats2 = db.get_statistics()
print(f"   {stats2}")

# 测试4: 查询历史
print("\n4. 查询历史记录:")
history = db.get_detection_history_with_pagination(page=1, page_size=5)
print(f"   记录数: {len(history['data'])}")
print(f"   总数: {history['pagination']['total']}")

# 测试5: 再次查询统计
print("\n5. 第三次查询统计:")
stats3 = db.get_statistics()
print(f"   {stats3}")

# 测试6: 模拟多次查询
print("\n6. 模拟10次连续查询:")
for i in range(10):
    stats = db.get_statistics()
    print(f"   第{i+1}次: total={stats['total']}")

print("\n=== 测试完成 ===")
