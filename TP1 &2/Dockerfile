# Utiliser une image de base Python légère
FROM python:3.9-slim

# Variables d'environnement utiles
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

WORKDIR /app

# Copier requirements.txt depuis la racine
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le contenu de code/ vers /app (app.py, mnist_model.h5, etc.)
COPY codesource/ .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--workers", "2", "--threads", "4"]
