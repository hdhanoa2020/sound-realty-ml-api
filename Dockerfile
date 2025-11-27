# Dockerfile
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy all project files into the container
COPY . /app

# Install Python dependencies
# (you can replace this with: RUN pip install -r requirements.txt  if you have that file)
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" pandas scikit-learn

# Expose port 80 inside the container
EXPOSE 80

# Start the FastAPI app with Uvicorn
# "app:app" means: module "app.py", variable "app" (your FastAPI instance)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
