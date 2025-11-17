# Imagen base oficial de Python
FROM python:3.12.11

WORKDIR /app

# Install curl + uv
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv

# First copy manifest files
COPY pyproject.toml ./

# Install dependencies from pyproject
RUN uv pip install . --system

# Copy project source code (keeping the structure)
COPY src/ /app/

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]