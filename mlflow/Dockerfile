FROM python:3.11

# Install MLflow
RUN pip install mlflow==2.5.0 && pip install psycopg2-binary

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app

COPY . /app

CMD ["mlflow", "server", "--host", "admin.entropy.nz"]
