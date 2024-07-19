FROM ubuntu:20.04

COPY final_graph_streamlit_stuff.py /var/home/sahitdodda/hospit/Medical-Data-Analysis/csv_results/final_graph_streamlit_stuff.py

RUN pip install --upgrade pip

RUN pip install --no-cacher-dir -r requirements.txt

RUN pip install streamlit

RUN pip install --upgrade pip setuptools