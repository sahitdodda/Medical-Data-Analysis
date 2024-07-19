FROM ubuntu:20.04

COPY ./path/to/vitalsP_graphing.py csv_results/vitalsP_graphing.py

RUN pip install --upgrade pip

RUN pip install --no-cacher-dir -r requirements.txt

RUN pip install streamlit

RUN pip install --upgrade pip setuptools