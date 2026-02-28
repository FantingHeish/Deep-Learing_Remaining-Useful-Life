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
| LSTM Seq2Seq |Encoder-Decoder with teacher forcing | Multi-step |
| Transformer | Multi-head attention + positional encoding | Single-step |

### 📊 Evaluation Metrics
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
2. Wrap with **FastAPI** as a **REST API** supporting all 4 models
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
- **Modeling:** Python, TensorFlow/Keras, NumPy, Scikit-learn
- **Serving:** FastAPI, Uvicorn
- **Infrastructure:** Docker, GCP Vertex AI, GCP Container Registry
- **Dataset:** NASA C-MAPSS(https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)

### ✨✨Result 

#### 1⃣ FD001**
**✨📈 FD001 Results Summary**
<img width="1123" height="341" alt="FD001_summery" src="https://github.com/user-attachments/assets/ccd4ac00-5692-4ce9-b61f-544ffd5100e1" />
<img width="1221" height="465" alt="FD001_tain_his" src="https://github.com/user-attachments/assets/fe09833d-51a7-49ea-9fa9-03ac02ec2ace" />
**✨ LSTM Multi-step**
<img width="1221" height="438" alt="FD001_multi" src="https://github.com/user-attachments/assets/e6e8f509-ff93-40d8-82ec-65e4ad1d14d7" />
**✨ LSTM Autoencoder**
<img width="1221" height="422" alt="FD001_autoencoder" src="https://github.com/user-attachments/assets/1f858c72-3f1e-4429-8b87-42d8c44f0f2a" />
**✨ LSTM Seq2Seq**
<img width="1221" height="438" alt="FD001_Seq2seq" src="https://github.com/user-attachments/assets/af38a816-366c-438d-bb2b-58087f957d7c" />
**✨ Transformer**
<img width="1221" height="440" alt="FD001_transf" src="https://github.com/user-attachments/assets/6b2fd466-b975-4742-ac6d-96bea15bcf90" />


#### **2⃣ FD002**
**✨📈 FD002 Results Summary**
<img width="1123" height="341" alt="FD002_summery" src="https://github.com/user-attachments/assets/57918dd0-2c32-4f2c-8156-f472909c49e4" />
<img width="1221" height="467" alt="FD002_train_his" src="https://github.com/user-attachments/assets/5d434e31-85ee-4fbb-bae6-716eb8600cbb" />
**✨ LSTM Multi-step**
<img width="1221" height="426" alt="FD002_multi" src="https://github.com/user-attachments/assets/88b771be-41b1-44a8-a858-36269f8a8e9f" />
**✨ LSTM Autoencoder**
<img width="1221" height="426" alt="FD002_autoencoder" src="https://github.com/user-attachments/assets/7880204a-6dff-4610-befc-2c5b7e6e0c83" />
**✨ LSTM Seq2Seq**
<img width="1221" height="430" alt="FD002_Seq2seq" src="https://github.com/user-attachments/assets/b6771068-3c29-45a0-b55e-192372e8f2d1" />
**✨ Transformer**
<img width="1221" height="426" alt="FD002_transf" src="https://github.com/user-attachments/assets/f46b6d73-2405-490b-ad7f-93f6cbd0601c" />


#### **3⃣ FD003**
> **✨📈 FD003 Results Summary**
<img width="1123" height="341" alt="FD003_summery" src="https://github.com/user-attachments/assets/f350dcd4-9e3b-41ba-9a88-b232dd42de0b" />
<img width="1221" height="474" alt="FD003_train_his" src="https://github.com/user-attachments/assets/a31cc31a-3635-4a60-b683-09df428fc256" />
**✨ LSTM Multi-step**
<img width="1221" height="428" alt="FD003_multi" src="https://github.com/user-attachments/assets/35004385-c1a0-43d0-bc95-e1f4d2aaf22b" />
**✨ LSTM Autoencoder**
<img width="1221" height="427" alt="FD003_autoencoder" src="https://github.com/user-attachments/assets/c8e03e1d-f301-404d-8e89-edcc025661fc" />
**✨ LSTM Seq2Seq**
<img width="1221" height="428" alt="FD003_Seq2seq" src="https://github.com/user-attachments/assets/6796d7d1-6749-450f-be19-57a97d5c654a" />
**✨ Transformer**
<img width="1221" height="431" alt="FD003_transf" src="https://github.com/user-attachments/assets/f036d281-930d-49f6-9660-fc3235f70ae4" />


#### **4⃣ FD004**
**✨📈 FD004 Results Summary**
<img width="1123" height="341" alt="FD004_summery" src="https://github.com/user-attachments/assets/5fd6b3e4-0565-4a94-8659-82733ae0c3c0" />
<img width="1221" height="477" alt="FD004_train_his" src="https://github.com/user-attachments/assets/c30a7fbc-ddd3-478b-938b-ae16c52f45c5" />
**✨ LSTM Multi-step**
<img width="1221" height="426" alt="FD004_multi" src="https://github.com/user-attachments/assets/9cd83b84-7695-4513-911f-dd314e31ca0a" />
**✨ LSTM Autoencoder**
<img width="1221" height="426" alt="FD004_antoencoder" src="https://github.com/user-attachments/assets/3167f092-8cb7-4350-950b-45ea4f2b2920" />
**✨ LSTM Seq2Seq**
<img width="1221" height="426" alt="FD004_Seq2seq" src="https://github.com/user-attachments/assets/e2fa7239-0728-4e50-a680-ee956995b4c0" />
**✨ Transformer**
<img width="1221" height="426" alt="FD004_transf" src="https://github.com/user-attachments/assets/1f9c2732-750d-4845-b165-9fb19088a739" />

