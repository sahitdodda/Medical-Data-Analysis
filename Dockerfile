FROM ubuntu:20.04

FROM python:3.9

# Copy the entire csv_results directory
COPY csv_results /app/csv_results

# Copy other necessary files
COPY requirements.txt /app/requirements.txt
COPY Procfile /app/Procfile

# Set the working directory
WORKDIR /app

# Install pip and upgrade it
RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install --upgrade pip

# Install requirements
RUN pip3 install -r requirements.txt

# Expose the port your application will run on
EXPOSE 8000

# Install Procfile dependencies (if any)
RUN apt-get install -y procps

# Run the command from the Procfile
CMD ["streamlit", "run", "csv_results/final_graph_streamlit_stuff.py", "--server.port=8000"]