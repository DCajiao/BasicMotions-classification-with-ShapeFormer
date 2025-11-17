# Imagen base oficial de Python
FROM python:3.12.11

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source code (keeping the structure)
COPY ./src/ ./

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

# # UV INSTALLATION
# RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
#     && mv /root/.local/bin/uv /usr/local/bin/uv \
#     && mv /root/.local/bin/uvx /usr/local/bin/uvx

# # First copy manifest files
# COPY pyproject.toml ./



# Install dependencies from pyproject
# RUN uv pip install -r pyproject.toml --system
# RUN uv pip compile pyproject.toml -o requirements.txt \
#     && uv pip install --system -r requirements.txt

# Copy project source code (keeping the structure)
# COPY ./src/ ./

