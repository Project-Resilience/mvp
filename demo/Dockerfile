FROM python:3.10-slim

# Debian basics
RUN apt-get update -y
RUN apt-get install curl git -y
RUN apt-get install gcc -y
RUN apt-get install g++ -y
RUN apt-get install libgdal-dev -y

# Where apps will live
ENV APPS_HOME /usr/local/cognizant

# Where this particular app will live
ENV ELUC_APP_HOME ${APPS_HOME}/eluc

# Dependencies
RUN pip install --no-cache-dir --upgrade pip
COPY ./requirements.txt ${ELUC_APP_HOME}/requirements.txt
# This is needed because of some weird GDAL error?
ENV GDAL_VERSION 3.7.1
# Token to download leaf-util
ARG leaf_private_source_credentials
ENV LEAF_PRIVATE_SOURCE_CREDENTIALS=${leaf_private_source_credentials}
RUN pip install --no-cache-dir -r ${ELUC_APP_HOME}/requirements.txt

# Copy source files over
COPY . ${ELUC_APP_HOME}
WORKDIR ${ELUC_APP_HOME}
ENV PYTHONPATH=${ELUC_APP_HOME}


