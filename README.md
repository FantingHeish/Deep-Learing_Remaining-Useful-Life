## Engine-Remaining-Life-Prediction

### 🔧 Task Definition
This project focuses on Remaining Useful Life (RUL) prediction using the C-MAPSS turbofan engine datasets (FD001–FD004).
Given historical multivariate sensor readings, the goal is to predict the future degradation trajectory of RUL, formulated as a multi-step sequence prediction problem.
Past sensor window  →  Future RUL trajectory

Each input sample consists of a fixed-length sensor window (32 timesteps × 24 features), and the model outputs a 5-step ahead RUL sequence.

### 🧠 Models Evaluated
**💡 Development Architectures：**
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
#### 1⃣ FD001**
**✨📈 FD001 Results Summary**
<img width="907" height="295" alt="FD001_summery" src="https://github.com/user-attachments/assets/10fea008-6180-457f-945b-190a298542d1" />

<img width="1230" height="477" alt="FD001_Loss_MAE" src="https://github.com/user-attachments/assets/af7c32dd-f83c-4c7e-ab28-0fc3db45dba4" />
<img width="1230" height="427" alt="FD001_LSTM_Mulit" src="https://github.com/user-attachments/assets/96d8fa9a-0fe2-4f89-a0b4-47c7cadf4b7e" />
<img width="1230" height="427" alt="FD001_LSTM_Autoencoder" src="https://github.com/user-attachments/assets/dd2fe0ea-bfc0-4e37-bdce-8bb7a0227a4f" />
<img width="1230" height="427" alt="FD001_LSTM_Seq" src="https://github.com/user-attachments/assets/f582b9a2-e32a-4605-89b2-4c6759f2b896" />
<img width="1230" height="427" alt="FD001_Transfrom" src="https://github.com/user-attachments/assets/0ac47011-31f8-4e78-8933-e6ae6405cdd2" />

#### **2⃣ FD002**
**✨📈 FD002 Results Summary**
<img width="907" height="295" alt="FD002_summery" src="https://github.com/user-attachments/assets/481170e4-2fcf-4063-aca3-db92edbe2f12" />

<img width="1225" height="480" alt="FD002_Loss_MAE" src="https://github.com/user-attachments/assets/ebcbca37-e05a-4fed-b124-9565434a1d7c" />
<img width="1225" height="426" alt="FD002_LSTM_Multi" src="https://github.com/user-attachments/assets/92cdf3cb-7ec7-4bb1-a38b-7ec0d76fb341" />
<img width="1225" height="426" alt="FD002_LSTM_Autoencoder" src="https://github.com/user-attachments/assets/dc5213a8-f12c-4838-8353-42a2cf487643" />
<img width="1225" height="426" alt="FD002_LSTM_Seq" src="https://github.com/user-attachments/assets/51f05ebf-ab41-4853-a010-faf7a56376f8" />
<img width="1225" height="426" alt="FD002_Transfrom" src="https://github.com/user-attachments/assets/cf8182e4-8cd0-492e-a513-c9a14629588f" />

#### **3⃣ FD003**
**✨📈 FD003 Results Summary**
<img width="907" height="295" alt="FD003_summery" src="https://github.com/user-attachments/assets/98142c66-1dd1-433e-ac72-ac3f69dd1df5" />

<img width="1225" height="478" alt="FD003_Loss_MAE" src="https://github.com/user-attachments/assets/689ea943-fd47-4dac-98a4-bf9ed57914fd" />
<img width="1225" height="433" alt="FD003_LSTM_Multi" src="https://github.com/user-attachments/assets/90c27c85-4e73-481c-af76-2e5a22302570" />
<img width="1225" height="424" alt="FD003_LSTM_Autoencoder" src="https://github.com/user-attachments/assets/53c6dfc5-1769-4d3b-b24a-6d0a6d9b5987" />
<img width="1225" height="432" alt="FD003_LSTM_Seq" src="https://github.com/user-attachments/assets/d4277b56-454f-4857-9b2f-44cbe3841cd6" />
<img width="1225" height="429" alt="FD003_Transfrom" src="https://github.com/user-attachments/assets/cd986dc9-cd5c-44e1-85f5-e12ea553dbfd" />

#### **4⃣ FD004**
**✨📈 FD004 Results Summary**
<img width="907" height="295" alt="FD004_summery" src="https://github.com/user-attachments/assets/f48be03e-0727-4931-8e6b-ff97acc90a0b" />

<img width="1225" height="480" alt="FD004_Loss_MAE" src="https://github.com/user-attachments/assets/a90d0042-7cfc-44b2-8c6c-1fed1d18d00d" />
<img width="1225" height="429" alt="FD004_LSTM_Multi" src="https://github.com/user-attachments/assets/89a4c9b7-c8e0-455d-9e8d-9cdc4ed0d34e" />
<img width="1225" height="430" alt="FD004_LSTM_Autoencoder" src="https://github.com/user-attachments/assets/ae3a6666-824c-476c-87c4-266b490ec827" />
<img width="1225" height="431" alt="FD004_LSTM_Seq" src="https://github.com/user-attachments/assets/7d69e646-ff69-4bd0-8588-38adff26116a" />
<img width="1225" height="428" alt="FD004_Transfrom" src="https://github.com/user-attachments/assets/ab97a511-db52-4975-b555-524f5c7836c3" />



### 專案檔案說明
- `Turbofan_RUL_Prediction.ipynb`：模型訓練 Notebook

Link to DataSet: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
