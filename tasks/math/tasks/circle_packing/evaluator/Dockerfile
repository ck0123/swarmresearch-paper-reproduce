FROM swarmresearch-task-base:py312
WORKDIR /benchmark

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY evaluator.py .
COPY evaluate.sh .
RUN chmod +x evaluate.sh

ENTRYPOINT ["./evaluate.sh"]
