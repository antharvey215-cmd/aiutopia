# Use a slim Python image
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# system deps for some Python packages (e.g., asyncpg)
RUN apt-get update && apt-get install -y build-essential gcc libpq-dev curl && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Expose port
EXPOSE 8000

# Start the server
CMD ["uvicorn", "aiutopia_backend:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
