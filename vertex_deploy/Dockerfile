FROM tensorflow/tensorflow:2.15.0

WORKDIR /app

COPY wheels/ ./wheels/
COPY requirements_light.txt .
RUN pip install --no-cache-dir ./wheels/*.whl && \
    pip install --no-cache-dir fastapi==0.109.0 uvicorn==0.27.0 pydantic==2.5.3

COPY saved_models/ ./saved_models/
COPY app.py .

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
