import streamlit as st


def show_home():
    st.title("TangerangCast: Rain Prediction Dashboard 🌧️")

    st.markdown("""
    Welcome to **TangerangCast**!
    
    This application leverages an advanced Stacking Ensemble machine learning architecture to provide highly accurate predictions for rainfall across the Tangerang area.
    
    ### Navigation
    Please use the sidebar to navigate to the specific modules:
    - **Live Map**: View current and future predictions on an interactive geospatial grid.
    - **Data Insight**: Explore underlying historical data, model performance metrics, and feature importance.
    """)


# Define the navigation structure with capitalized titles and icons
home_page = st.Page(show_home, title="App", icon="🏠", default=True)
live_map_page = st.Page("pages/live_map.py", title="Live Map", icon="🗺️")
data_insight_page = st.Page("pages/data_insight.py", title="Data Insight", icon="📊")

st.set_page_config(page_title="TangerangCast", page_icon="🌧️", layout="wide")

# Initialize navigation
pg = st.navigation([home_page, live_map_page, data_insight_page])
pg.run()
