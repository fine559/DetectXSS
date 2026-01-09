print("Testing all imports...")
try:
    import numpy
    print(f"numpy: {numpy.__version__}")
except Exception as e:
    print(f"numpy error: {e}")

try:
    import pandas
    print(f"pandas: {pandas.__version__}")
except Exception as e:
    print(f"pandas error: {e}")

try:
    import sklearn
    print(f"sklearn: {sklearn.__version__}")
except Exception as e:
    print(f"sklearn error: {e}")

try:
    import xgboost
    print(f"xgboost: {xgboost.__version__}")
except Exception as e:
    print(f"xgboost error: {e}")

try:
    import tensorflow as tf
    print(f"tensorflow: {tf.__version__}")
except Exception as e:
    print(f"tensorflow error: {e}")

try:
    import keras
    print(f"keras: {keras.__version__}")
except Exception as e:
    print(f"keras error: {e}")

try:
    import flask
    print(f"flask: {flask.__version__}")
except Exception as e:
    print(f"flask error: {e}")

try:
    import pymysql
    print(f"pymysql: {pymysql.__version__}")
except Exception as e:
    print(f"pymysql error: {e}")

print("\nTesting custom modules...")
import sys
sys.path.insert(0, '.')
try:
    from data_processor import XSSDataProcessor
    print("data_processor: OK")
except Exception as e:
    print(f"data_processor error: {e}")
    import traceback
    traceback.print_exc()

try:
    from models import XGBoostModel, BiLSTMModel, TransformerModel
    print("models: OK")
except Exception as e:
    print(f"models error: {e}")
    import traceback
    traceback.print_exc()

print("\nAll imports completed!")
