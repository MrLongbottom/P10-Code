# syntax=docker/dockerfile:1
FROM python:3.9
WORKDIR .
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "./model/pachinko_gibbs_lda.py"]