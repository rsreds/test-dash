FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir \
    numpy \
    pandas \
    plotly \
    dash \
    TheOptimizer\
    dash-bootstrap-components

EXPOSE 8080

CMD ["python", "app.py"]
