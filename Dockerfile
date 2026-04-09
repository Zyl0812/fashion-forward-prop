FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml ./
COPY src ./src
COPY assets ./assets
COPY artifacts ./artifacts
COPY app.py ./

ENV PYTHONPATH=/app/src

CMD ["python", "app.py"]
