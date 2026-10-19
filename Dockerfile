FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY recsys ./recsys
ENV ARTIFACTS=/app/artifacts
EXPOSE 8080
# Mount trained artifacts at /app/artifacts (dataset.pkl, two_tower.pt, item_vecs.npy, ranker_lgbm.txt, serving.pkl)
CMD ["uvicorn", "recsys.serve:app", "--host", "0.0.0.0", "--port", "8080"]
