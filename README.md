## Engine-Remaining-Life-Prediction

### 🔧 Task Definition
This project focuses on Remaining Useful Life (RUL) prediction using the C-MAPSS turbofan engine datasets (FD001–FD004).
Given historical multivariate sensor readings, the goal is to predict the future degradation trajectory of RUL, formulated as a multi-step sequence prediction problem.
Past sensor window  →  Future RUL trajectory

Each input sample consists of a fixed-length sensor window (32 timesteps × 24 features), and the model outputs a 5-step ahead RUL sequence.

### 🧠 Models Evaluated
**🧱 Development Architectures：**
Four deep learning architectures were implemented and evaluated:
- LSTM Multi-step
- LSTM Autoencoder
- LSTM Seq2Seq
- Transformer

**📊 Evaluation Metrics**
To provide a comprehensive evaluation, the following metrics were used:
- MAE / Median AE
- RMSE
- R² Score
- Explained Variance
- sMAPE
- MAPE

All models were trained and evaluated consistently across FD001–FD004 datasets.

### 訓練結果
**1⃣ FD001**
**1⃣ FD001**
**1⃣ FD001**
**1⃣ FD001**

**📈 Results Summary**

### 專案檔案說明
- `Turbofan_RUL_Prediction.ipynb`：模型訓練 Notebook

Link to DataSet: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
