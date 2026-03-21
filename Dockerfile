# DIP-UP – Docker image (multi-platform)
# Usage:
#   docker compose build && docker compose up
#   Open http://localhost:7860

FROM python:3.10-slim

LABEL maintainer="Hongfu Sun <hongfu.sun@uq.edu.au>"
LABEL description="DIP-UP – web interface"

ARG TARGETARCH
ARG TORCH_VARIANT=cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN if [ "$TARGETARCH" = "arm64" ]; then \
        pip install --no-cache-dir "torch>=2.1.0"; \
    else \
        pip install --no-cache-dir "torch>=2.1.0" \
            --index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}"; \
    fi

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /DIP-UP
COPY . .

EXPOSE 7860
CMD ["python", "app.py", "--server-name", "0.0.0.0", "--server-port", "7860"]
