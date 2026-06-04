#!/bin/bash
python main.py &
exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0