## Engine-Remaining-Life-Prediction

### 專案簡介
以 LSTM 與 Transformer 模型訓練 NASA CMAPSS 引擎資料集，預測引擎的剩餘壽命 (RUL)，用於設備健康管理與預測性維護。本專案使用 NASA 提供的 CMAPSS 資料集，建立時間序列預測模型，比較 LSTM 與 Transformer 架構在引擎退化預測上的表現差異。結果顯示 Transformer 在誤差率及 R² 皆顯著優於 LSTM。

### 技術架構
- **開發框架：** TensorFlow、Keras  
- **模型：** LSTM、Transformer  
- **評估指標：** RMSE、MAE、R²

### 訓練結果
<img width="450" height="138" alt="Screenshot 2026-02-07 at 01 21 41" src="https://github.com/user-attachments/assets/a7422958-81af-4d19-bd9f-f7196b4732c4" />

<img width="712" height="388" alt="Screenshot 2026-02-07 at 01 21 48" src="https://github.com/user-attachments/assets/457c4921-34b4-449c-bd6c-134b86500234" />
<img width="705" height="389" alt="Screenshot 2026-02-07 at 01 21 55" src="https://github.com/user-attachments/assets/1775352d-6f2b-4a5f-bb5c-aa744c940777" />
<img width="710" height="395" alt="Screenshot 2026-02-07 at 01 22 01" src="https://github.com/user-attachments/assets/63044347-0739-4a43-b1b5-23299d9e70f0" />
<img width="718" height="399" alt="Screenshot 2026-02-07 at 01 22 07" src="https://github.com/user-attachments/assets/9f5465c4-fc7e-4d67-889d-4bca87835b16" />

### 專案檔案說明
- `Turbofan_RUL_Prediction.ipynb`：模型訓練 Notebook

Link to DataSet: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
