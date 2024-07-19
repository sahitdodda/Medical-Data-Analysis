FROM ubuntu:20.04

# Install necessary dependencies (adjust as needed)
RUN apt-get update && apt-get install -y python3 python3-pip

# Set the working directory in the container
WORKDIR /app

# Copy the entire csv_results directory
COPY csv_results ./csv_results

# Copy the main Python file
COPY vitalsP_graphing.py .

# Copy requirements file
COPY requirements.txt .

# Install requirements
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Expose the port Streamlit will run on
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "vitalsP_graphing.py", "--server.port=8501", "--server.address=0.0.0.0"]