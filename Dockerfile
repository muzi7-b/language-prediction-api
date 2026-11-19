FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code and model.
COPY app/ ./app/
COPY model/ ./model/

# Inside the container the server binds all interfaces — this is correct, because
# the container's network IS the stack boundary. Exposure is controlled by how you
# PUBLISH the port, not by this bind. Publish to loopback on the host:
#     docker run -p 127.0.0.1:8000:8000 language-prediction-api
# (A bare `-p 8000:8000` would publish on all host interfaces — avoid that unless
# you intend the service to be reachable from other machines.)
ENV LANGID_HOST=0.0.0.0
ENV LANGID_PORT=8000
EXPOSE 8000

CMD ["python", "-m", "app.main"]
