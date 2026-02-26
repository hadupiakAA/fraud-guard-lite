FROM python:3.11-slim

# Робоча директорія всередині контейнера
WORKDIR /app

# Копіюємо requirements окремо (кешування)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Копіюємо весь код
COPY . .

# Відкриваємо порт
EXPOSE 8000

# Запуск FastAPI
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]