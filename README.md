## Deep Learing Modle_Remaining Useful Life

### 🚀 Project Overview
This project focuses on predicting the degradation process and Remaining Useful Life (RUL) of turbofan jet engines using the NASA C-MAPSS dataset. Four deep learning architectures are trained, compared, and deployed as a production REST API on GCP Vertex AI.
Given historical multivariate sensor readings formulated as a multi-step sequence prediction problem, the models forecast RUL across multiple time horizons. Explore different architectures including LSTM, Seq2Seq, Autoencoder, and Transformer to forecast RUL across multiple sensors.

### 🎯 Task Definition
1. Transform historical sensor readings into sequence inputs to predict future RUL trajectories
2. Support single-step and multi-step forecasting, comparing different models' ability to capture degradation trends
3. Evaluate model accuracy and reliability using multiple comprehensive metrics
4. Deploy all models as a scalable REST API endpoint on GCP Vertex AI

### 🗂 Data Processing
1. Compute RUL for each sensor  
2. Split dataset into 80% training and 20% testing sets  
3. Convert time-series data into input sequences using a sliding window approach  
4. Normalize each sequence independently 

### 🧠 Model Architectures
Four deep learning architectures implemented and evaluated:
| Model | Architecture | Prediction Type |
|---|---|---|
| LSTM Multi-step | 2-layer LSTM + BatchNorm + Dense | Multi-step |
| LSTM Autoencoder | Encoder-Decoder LSTM + prediction head | Single-step |
| LSTM Seq2Seq |Encoder-Decoder with teacher forcing | Multi-step (5 steps) |
| Transformer | Multi-head attention + positional encoding | Single-step |

### 📊 Evaluation Metrics：
To provide a comprehensive evaluation, the following metrics were used
1. MAE / Median AE — average and median absolute prediction error
2. RMSE — root mean squared error, sensitive to large errors
3. R² Score — proportion of variance explained
4. Explained Variance — similar to R² but robust to systematic bias
5. sMAPE — symmetric MAPE, avoids extreme values near RUL=0
6. MAPE — standard percentage error (filtered for RUL > 10)
All models were trained and evaluated consistently across FD001–FD004 datasets.

### **🚢 Deployment**
All four models are containerized and deployed as a **single REST API** on **GCP Vertex AI**.

#### Deployment Pipeline
1. Train models and save as `.keras` files
2. Wrap with **FastAPI** as a REST API supporting all 4 models
3. Containerize with **Docker** to ensure environment consistency
4. Push image to **GCP Container Registry**
5. Deploy to **GCP Vertex AI** as a managed endpoint

#### Architecture
```
FastAPI server (app.py)
    └── 4 models loaded at startup
            ├── FD001_lstm_multistep.keras
            ├── FD001_lstm_autoencoder.keras
            ├── FD001_lstm_seq2seq.keras
            └── FD001_transformer.keras

Docker container → GCP Container Registry → Vertex AI Managed Endpoint
```

#### Local Setup
```bash
cd vertex_deploy
pip install fastapi uvicorn tensorflow
uvicorn app:app --host 0.0.0.0 --port 8080
```

#### API Usage
```python
import requests

payload = {
    "sequences": [[[...]]],  # shape: (batch_size, 32, 24)
    "model_name": "transformer"  # lstm_multistep | lstm_autoencoder | lstm_seq2seq | transformer
}
r = requests.post("http://localhost:8080/predict", json=payload)
# {"model_name": "transformer", "predictions": [45.23], "count": 1}
```

#### GCP Deployment
```bash
cd vertex_deploy
chmod +x deploy_to_vertex.sh
./deploy_to_vertex.sh   # builds image → pushes to GCR → uploads model → creates endpoint
```


### **🛠 Tech Stack**
**Modeling:** Python, TensorFlow/Keras, NumPy, Scikit-learn
**Serving:** FastAPI, Uvicorn
**Infrastructure:** Docker, GCP Vertex AI, GCP Container Registry
**Dataset:** NASA C-MAPSS(https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)

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

