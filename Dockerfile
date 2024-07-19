FROM ubuntu:20.04

# Copy the entire csv_results directory
COPY csv_results /app/csv_results

# Copy other necessary files
COPY requirements.txt /app/requirements.txt

# Set the working directory
WORKDIR /app

# Install pip and upgrade it
RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install --upgrade pip

# Install requirements
RUN pip3 install -r requirements.txt

# Add any other necessary commands here