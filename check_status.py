from database import db, init_database

# 初始化数据库连接
if init_database():
    print("数据库初始化成功")
else:
    print("数据库初始化失败")

# 查询当前记录
print("\n当前数据库中的记录:")
history = db.get_detection_history(limit=10)
print(f"找到 {len(history)} 条记录")
for record in history:
    print(f"  ID={record['id']}, 文本={record['input_text'][:30]}, XSS={record['is_xss']}, 时间={record['detection_time']}")

# 查询统计
print("\n当前统计:")
stats = db.get_statistics()
print(f"  总次数: {stats['total']}")
print(f"  XSS: {stats['xss_count']}")
print(f"  正常: {stats['normal_count']}")
