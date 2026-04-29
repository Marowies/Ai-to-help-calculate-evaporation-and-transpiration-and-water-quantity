FROM python:3.10

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Hugging Face Spaces uses port 7860 by default
ENV PORT=7860
EXPOSE 7860

# Run the application
CMD ["python", "api_v2.py"]
