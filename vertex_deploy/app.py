# =========================================================
# app.py — FastAPI prediction server
# 支援四個模型: lstm_multistep, lstm_autoencoder,
#               lstm_seq2seq, transformer
#
# 本地啟動:
#   uvicorn app:app --host 0.0.0.0 --port 8080
# =========================================================
import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# ── PositionalEncoding (Transformer 需要) ─────────────────
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        pos    = np.arange(max_len)[:, np.newaxis]
        i      = np.arange(d_model)[np.newaxis, :]
        rates  = 1 / np.power(10000, (2*(i//2)) / np.float32(d_model))
        angles = pos * rates
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        self.pe = tf.cast(angles[np.newaxis, ...], tf.float32)

    def call(self, x):
        return x + self.pe[:, :tf.shape(x)[1], :]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"max_len": self.max_len, "d_model": self.d_model})
        return cfg

CUSTOM_OBJECTS = {"PositionalEncoding": PositionalEncoding}

# ── 載入四個模型 ──────────────────────────────────────────
MODEL_DIR  = os.environ.get("MODEL_DIR", "saved_models")
PRED_STEPS = 5   # seq2seq / multistep 的輸出步數

print(f"Loading models from: {MODEL_DIR}")

models = {
    "lstm_multistep":   tf.keras.models.load_model(
                            f"{MODEL_DIR}/FD001_lstm_multistep.keras"),
    "lstm_autoencoder": tf.keras.models.load_model(
                            f"{MODEL_DIR}/FD001_lstm_autoencoder.keras"),
    "lstm_seq2seq":     tf.keras.models.load_model(
                            f"{MODEL_DIR}/FD001_lstm_seq2seq.keras"),
    "transformer":      tf.keras.models.load_model(
                            f"{MODEL_DIR}/FD001_transformer.keras",
                            custom_objects=CUSTOM_OBJECTS),
}

for name, m in models.items():
    print(f"  ✅ {name:25s} input={m.input_shape}")

# ── FastAPI ───────────────────────────────────────────────
app = FastAPI(
    title="RUL Predictor — FD001",
    description="Remaining Useful Life prediction. 4 models available.",
    version="1.0.0"
)

# ── Schemas ───────────────────────────────────────────────
class PredictRequest(BaseModel):
    # sequences: (batch_size, seq_len=32, features=24)
    sequences: List[List[List[float]]]
    # model_name: one of lstm_multistep | lstm_autoencoder |
    #             lstm_seq2seq | transformer
    model_name: str = "transformer"

class PredictResponse(BaseModel):
    model_name: str
    # predictions: RUL 預測值，每筆輸入一個數字
    predictions: List[float]
    count: int

# ── Helper: normalize ────────────────────────────────────
def normalize(X: np.ndarray) -> np.ndarray:
    """Per-sequence standardization，與訓練時相同"""
    out = []
    for seq in X:
        mean = seq.mean(axis=0, keepdims=True)
        std  = seq.std(axis=0,  keepdims=True) + 1e-8
        out.append((seq - mean) / std)
    return np.array(out)

# ── Endpoints ────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(models.keys())}

@app.get("/models")
def list_models():
    return {"available_models": list(models.keys())}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # 驗證 model_name
    if request.model_name not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{request.model_name}'. "
                   f"Choose from: {list(models.keys())}"
        )

    # 轉成 numpy
    try:
        X = np.array(request.sequences, dtype="float32")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid input: {e}")

    # 驗證 shape
    if X.ndim != 3 or X.shape[1] != 32 or X.shape[2] != 24:
        raise HTTPException(
            status_code=422,
            detail=f"Expected shape (batch, 32, 24), got {X.shape}"
        )

    X_norm = normalize(X)
    model  = models[request.model_name]

    # ── 各模型的推論方式不同 ──────────────────────────────
    if request.model_name == "lstm_multistep":
        # output: (batch, pred_steps=5) → 取最後一步
        preds = model.predict(X_norm, verbose=0)
        preds = preds[:, -1]

    elif request.model_name == "lstm_autoencoder":
        # output: [reconstruction, prediction]
        # prediction shape: (batch, 1)
        out   = model.predict(X_norm, verbose=0)
        preds = out[1].ravel()

    elif request.model_name == "lstm_seq2seq":
        # 需要 decoder input (全零 teacher-forcing placeholder)
        batch = X_norm.shape[0]
        dec_input = np.zeros((batch, PRED_STEPS, 1), dtype="float32")
        preds = model.predict([X_norm, dec_input], verbose=0)
        # output: (batch, pred_steps, 1) → 取最後一步
        preds = preds[:, -1, 0]

    else:  # transformer
        # output: (batch, 1)
        preds = model.predict(X_norm, verbose=0).ravel()

    return PredictResponse(
        model_name=request.model_name,
        predictions=preds.tolist(),
        count=len(preds)
    )
