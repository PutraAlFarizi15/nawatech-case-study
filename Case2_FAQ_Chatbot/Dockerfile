# Define Base Image
FROM python:3.12-slim

# Set Working Directory
WORKDIR /app

# Copy and Install Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Application Code and Data
COPY . .

# Expose Port
EXPOSE 8501

# Run the Application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]