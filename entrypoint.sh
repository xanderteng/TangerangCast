#!/bin/sh
# Start the background data fetcher scheduler
python src/api_fetcher.py &

# Start Streamlit in the foreground, using exec to replace this shell process (so it receives OS signals properly)
exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0
