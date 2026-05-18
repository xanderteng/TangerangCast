import streamlit as st

st.set_page_config(
    page_title="Tangerang Rain Prediction",
    page_icon="🌧️",
    layout="wide"
)

def main():
    st.title("Tangerang Rain Prediction Dashboard 🌧️")
    
    st.markdown("""
    Welcome to the **Tangerang Rain Prediction Dashboard**!
    
    This application leverages machine learning (XGBoost) to provide predictions for rainfall in the Tangerang area.
    
    ### Navigation
    Please use the sidebar to navigate to the specific modules:
    - **Live Map (`live_map.py`)**: View the current and future predictions on a geospatial map.
    - **Data Insight (`data_insight.py`)**: Explore the underlying historical data and model performance metrics.
    """)

if __name__ == "__main__":
    main()
