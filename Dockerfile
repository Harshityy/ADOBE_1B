FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Add this line to download punkt
RUN python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"


CMD ["python", "main.py"]
