print("Testing imports...")

try:
    from flask import Flask, render_template, request, jsonify
    print("✓ Flask imports OK")
except Exception as e:
    print(f"✗ Flask imports failed: {e}")

try:
    from flask.json import JSONEncoder
    print("✓ JSONEncoder import OK")
except Exception as e:
    print(f"✗ JSONEncoder import failed: {e}")

try:
    from database import init_database, db
    print("✓ Database imports OK")
except Exception as e:
    print(f"✗ Database imports failed: {e}")

try:
    from ensemble import init_detector, detect_xss
    print("✓ Ensemble imports OK")
except Exception as e:
    print(f"✗ Ensemble imports failed: {e}")

try:
    import logging
    print("✓ Logging import OK")
except Exception as e:
    print(f"✗ Logging import failed: {e}")

try:
    import datetime
    from decimal import Decimal
    print("✓ datetime/Decimal imports OK")
except Exception as e:
    print(f"✗ datetime/Decimal imports failed: {e}")

print("\nAll imports completed!")
