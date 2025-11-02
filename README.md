## Turbofan-Engine-Remaining-Life-Prediction

### 專案簡介
以 LSTM 與 Transformer 模型訓練 NASA CMAPSS 引擎資料集，預測引擎的剩餘壽命 (RUL)，用於設備健康管理與預測性維護。本專案使用 NASA 提供的 CMAPSS 資料集，建立時間序列預測模型，比較 LSTM 與 Transformer 架構在引擎退化預測上的表現差異。結果顯示 Transformer 在誤差率及 R² 皆顯著優於 LSTM。

### 技術架構
- **開發框架：** TensorFlow、Keras  
- **模型：** LSTM、Transformer  
- **評估指標：** RMSE、MAE、R²

### 專案檔案說明
- `Turbofan_RUL_Prediction.ipynb`：模型訓練 Notebook
