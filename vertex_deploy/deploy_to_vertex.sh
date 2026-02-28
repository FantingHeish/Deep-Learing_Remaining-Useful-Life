# =========================================================
# deploy_to_vertex.sh — 部署到 GCP Vertex AI
#
# 使用方式:
#   chmod +x deploy_to_vertex.sh
#   ./deploy_to_vertex.sh
# =========================================================
# ────────────────────────────────────────
PROJECT_ID="agile-entry-454207-t3"   # GCP Project ID
REGION="us-central1"
IMAGE_NAME="rul-predictor"
IMAGE_TAG="v1"
# ─────────────────────────────────────────────────────────

IMAGE_URI="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "=================================================="
echo "RUL Predictor — Vertex AI Deployment"
echo "  Project : ${PROJECT_ID}"
echo "  Region  : ${REGION}"
echo "  Image   : ${IMAGE_URI}"
echo "=================================================="
echo ""

# ── Step 1: 登入 & 設定 GCP ──────────────────────────────
echo "[1/5] 登入 GCP..."
gcloud auth login
gcloud config set project ${PROJECT_ID}

echo "開啟必要 API..."
gcloud services enable aiplatform.googleapis.com
gcloud services enable containerregistry.googleapis.com
echo "✅ API 已開啟"

# ── Step 2: Build Docker image ───────────────────────────
echo ""
echo "[2/5] Build Docker image..."
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
echo "✅ Docker image built"

# 本地快速驗證
echo "本地驗證 container..."
docker run -d --name rul-test -p 8080:8080 ${IMAGE_NAME}:${IMAGE_TAG}
sleep 12
HEALTH=$(curl -s http://localhost:8080/health)
echo "   Health response: ${HEALTH}"
docker stop rul-test && docker rm rul-test
echo "✅ 本地驗證完成"

# ── Step 3: Push image 到 GCP ────────────────────────────
echo ""
echo "[3/5] Push image 到 GCP Container Registry..."
gcloud auth configure-docker --quiet
docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${IMAGE_URI}
docker push ${IMAGE_URI}
echo "✅ Image pushed: ${IMAGE_URI}"

# ── Step 4: Upload model 到 Vertex AI Model Registry ─────
echo ""
echo "[4/5] Upload model 到 Vertex AI..."
MODEL_RESOURCE=$(gcloud ai models upload \
  --region=${REGION} \
  --display-name="${IMAGE_NAME}" \
  --container-image-uri=${IMAGE_URI} \
  --container-ports=8080 \
  --container-health-route=/health \
  --container-predict-route=/predict \
  --format="value(model)")

MODEL_ID=$(echo ${MODEL_RESOURCE} | sed 's/.*models\///' | sed 's/@.*//')
echo "✅ Model uploaded, ID: ${MODEL_ID}"

# ── Step 5: 建立 Endpoint 並部署 ─────────────────────────
echo ""
echo "[5/5] 建立 Endpoint 並部署 (約需 5-10 分鐘)..."

ENDPOINT_RESOURCE=$(gcloud ai endpoints create \
  --region=${REGION} \
  --display-name="${IMAGE_NAME}-endpoint" \
  --format="value(name)")

ENDPOINT_ID=$(echo ${ENDPOINT_RESOURCE} | sed 's/.*endpoints\///')
echo "   Endpoint ID: ${ENDPOINT_ID}"

gcloud ai endpoints deploy-model ${ENDPOINT_ID} \
  --region=${REGION} \
  --model=${MODEL_ID} \
  --display-name="${IMAGE_NAME}-v1" \
  --machine-type=n1-standard-2 \
  --min-replica-count=1 \
  --max-replica-count=2

# ── 產生測試 payload ──────────────────────────────────────
python3 -c "
import json, numpy as np
payload = {
    'sequences': np.random.randn(2, 32, 24).tolist(),
    'model_name': 'transformer'
}
with open('test_payload.json', 'w') as f:
    json.dump(payload, f)
print('test_payload.json 已產生')
"

# ── 完成 ──────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "✅ 部署完成！"
echo ""
echo "Endpoint ID : ${ENDPOINT_ID}"
echo "Model ID    : ${MODEL_ID}"
echo ""
echo "測試部署 (在 GCP Console 或 terminal):"
echo ""
echo "  gcloud ai endpoints predict ${ENDPOINT_ID} \\"
echo "    --region=${REGION} \\"
echo "    --json-request=test_payload.json"
echo ""
echo "⚠️  請記下 Endpoint ID 和 Model ID，cleanup 時需要"
echo "=================================================="
