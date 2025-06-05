# Use a Python base image that already includes Python and pip
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install Python dependencies
RUN python3 -m ensurepip --upgrade && \
    pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

# Copy the rest of your application code into the container
COPY . /app/

# Expose the port on which FastAPI will run (default is 8000)
EXPOSE 8000

# Set the environment variable for the port (optional, but good practice)
ENV PORT 8000

# Start the FastAPI application using uvicorn
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
