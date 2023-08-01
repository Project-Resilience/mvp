FROM python:3.10-slim

# Define environment variables
ENV APPS_HOME=/usr/local/cognizant \
    ELUC_APP_HOME=/usr/local/cognizant/eluc \
    GDAL_VERSION=3.7.1 \
    PYTHONPATH=/usr/local/cognizant/eluc

# Debian basics and cleaning up in one RUN statement to reduce image size
RUN apt-get update -y && \
    apt-get install --no-install-recommends curl git gcc g++ libgdal-dev -y && \
    rm -rf /var/lib/apt/lists/* 

# Set work directory
WORKDIR ${ELUC_APP_HOME}

# Token to download leaf-util
ARG leaf_private_source_credentials
ENV LEAF_PRIVATE_SOURCE_CREDENTIALS=${leaf_private_source_credentials}

# Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source files over
COPY . .

# Expose Flask (Dash) port
EXPOSE 4057

# Run main UI
ENTRYPOINT ["python", "-m", "app.app"]