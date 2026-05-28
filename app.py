import streamlit as st


def show_home():
    st.title("Tangerang Rain Prediction Dashboard 🌧️")

    st.markdown("""
    Welcome to the **Tangerang Rain Prediction Dashboard**!
    
    This application leverages machine learning (XGBoost) to provide predictions for rainfall in the Tangerang area.
    
    ### Navigation
    Please use the sidebar to navigate to the specific modules:
    - **Live Map**: View the current and future predictions on a geospatial map.
    - **Data Insight**: Explore the underlying historical data and model performance metrics.
    """)


# Define the navigation structure with capitalized titles and icons
home_page = st.Page(show_home, title="App", icon="🏠", default=True)
live_map_page = st.Page("pages/live_map.py", title="Live Map", icon="🗺️")
data_insight_page = st.Page("pages/data_insight.py", title="Data Insight", icon="📊")

st.set_page_config(page_title="Tangerang Rain Prediction", page_icon="🌧️", layout="wide")

# Initialize navigation
pg = st.navigation([home_page, live_map_page, data_insight_page])
pg.run()
