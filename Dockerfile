FROM ubuntu:20.04

# Install Python 3.9 and pip
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.9 python3.9-distutils
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
RUN apt-get install -y python3-pip

# Rest of your Dockerfile...
COPY csv_results /app/csv_results
COPY requirements.txt /app/requirements.txt
WORKDIR /app

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install requirements
RUN pip3 install -r requirements.txt

# Add any other necessary commands here