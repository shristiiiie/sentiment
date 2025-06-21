FROM python:3.9-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip && pip install -r requirements.txt

CMD ["python", "src/main.py"]  # Replace with correct script name
