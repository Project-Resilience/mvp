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

# Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source files over
COPY . .

# Python setup script - downloads data and processes it
RUN python -m app.process_data

# Expose Flask (Dash) port
EXPOSE 4057

# Run main UI
ENTRYPOINT ["gunicorn", "-b", "0.0.0.0:4057", "--timeout", "45", "app.app:server"]