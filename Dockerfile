# Serves the Streamlit app. Ollama runs on the host machine, so the container
# points at it via host.docker.internal.
#
#   docker build -t text2sql .
#   docker run -p 8501:8501 -v ./data:/app/data text2sql

FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY app.py config.yaml ./
COPY eval ./eval

RUN useradd --create-home appuser
USER appuser

ENV OLLAMA_HOST=http://host.docker.internal:11434
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.headless=true"]
