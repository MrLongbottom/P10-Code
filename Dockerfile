# syntax=docker/dockerfile:1
FROM python:3.9-slim
WORKDIR .
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENV PYTHONPATH "${PYTHONPATH}:/model/"
CMD ["python", "./model/pachinko_gibbs_lda.py"]