FROM python:3.12-slim

# Step 2: Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8181

# Step 3: Set working directory
WORKDIR /app

# Step 4: Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the FastAPI app files into the container
COPY . .

# Step 6: Expose the application port
EXPOSE $PORT

# Step 7: Command to run the application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8181"]
