FROM swarmresearch-task-base:py312
WORKDIR /benchmark

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libcairo2 \
    libgdk-pixbuf-2.0-0 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x evaluate.sh

ENTRYPOINT ["./evaluate.sh"]
