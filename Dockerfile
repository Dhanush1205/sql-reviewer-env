FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# HuggingFace Spaces uses port 7860
EXPOSE 7860

# Environment variables (API_BASE_URL and MODEL_NAME have defaults in inference.py)
ENV PORT=7860

CMD ["python", "app.py"]
