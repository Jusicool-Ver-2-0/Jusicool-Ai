FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml poetry.lock /app/

RUN pip install --upgrade pip

RUN pip install poetry && poetry install --no-root

COPY . /app

EXPOSE 3000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3000"]