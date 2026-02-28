# =========================================================
# test_api.py — 本地 API 測試
# 先跑: uvicorn app:app --host 0.0.0.0 --port 8080
# 再跑: python test_api.py
# =========================================================
import requests
import numpy as np

BASE = "http://localhost:8080"
MODELS = ["lstm_multistep", "lstm_autoencoder", "lstm_seq2seq", "transformer"]

# 1. Health check
print("=" * 50)
print("1. Health Check")
r = requests.get(f"{BASE}/health")
print(r.json())

# 2. 列出可用模型
print("\n" + "=" * 50)
print("2. Available Models")
r = requests.get(f"{BASE}/models")
print(r.json())

# 3. 用真實 sample 測試四個模型
print("\n" + "=" * 50)
print("3. Predictions with real samples")

try:
    sample_single = np.load("saved_models/FD001_sample_single.npy")  # (1,32,24)
    sample_multi  = np.load("saved_models/FD001_sample_multi.npy")   # (1,32,24)
    print(f"   Loaded real samples: single={sample_single.shape}, multi={sample_multi.shape}")
except FileNotFoundError:
    print("   Real samples not found, using random data")
    sample_single = np.random.randn(1, 32, 24).astype("float32")
    sample_multi  = sample_single.copy()

for model_name in MODELS:
    payload = {
        "sequences": sample_single.tolist(),
        "model_name": model_name
    }
    r = requests.post(f"{BASE}/predict", json=payload)
    if r.status_code == 200:
        result = r.json()
        print(f"   ✅ {model_name:25s} → RUL = {result['predictions'][0]:.2f}")
    else:
        print(f"   ❌ {model_name:25s} → Error: {r.json()}")

# 4. Batch 測試 (5 筆)
print("\n" + "=" * 50)
print("4. Batch Prediction (5 samples) — Transformer")
batch = np.random.randn(5, 32, 24).tolist()
r = requests.post(f"{BASE}/predict", json={
    "sequences": batch,
    "model_name": "transformer"
})
result = r.json()
print(f"   Predictions: {[round(p, 2) for p in result['predictions']]}")
print(f"   Count: {result['count']}")

# 5. 錯誤處理測試
print("\n" + "=" * 50)
print("5. Error Handling Tests")

# 錯誤的 model name
r = requests.post(f"{BASE}/predict", json={
    "sequences": sample_single.tolist(),
    "model_name": "nonexistent_model"
})
print(f"   Wrong model name → status={r.status_code}, detail={r.json()['detail'][:60]}")

# 錯誤的 shape
wrong_shape = np.random.randn(1, 20, 24).tolist()  # seq_len=20 instead of 32
r = requests.post(f"{BASE}/predict", json={
    "sequences": wrong_shape,
    "model_name": "transformer"
})
print(f"   Wrong shape      → status={r.status_code}, detail={r.json()['detail'][:60]}")

print("\n✅ All tests completed")
