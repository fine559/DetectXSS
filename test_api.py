import requests
import json

# 测试API
base_url = "http://localhost:5000"

# 1. 测试检测接口
print("1. 测试检测接口...")
detect_data = {
    "text": "hello"
}
try:
    response = requests.post(f"{base_url}/api/detect", json=detect_data)
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
except Exception as e:
    print(f"请求失败: {e}")

# 2. 测试统计接口
print("\n2. 测试统计接口...")
try:
    response = requests.get(f"{base_url}/api/statistics")
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
except Exception as e:
    print(f"请求失败: {e}")

# 3. 测试历史接口
print("\n3. 测试历史接口...")
try:
    response = requests.get(f"{base_url}/api/history")
    print(f"状态码: {response.status_code}")
    result = response.json()
    print(f"记录数: {len(result)}")
    if result:
        print(f"第一条记录: {json.dumps(result[0], indent=2, ensure_ascii=False)}")
except Exception as e:
    print(f"请求失败: {e}")
