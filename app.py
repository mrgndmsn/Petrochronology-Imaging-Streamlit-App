"""Application entry point and ordered sidebar navigation."""

import streamlit as st
from core.navigation import pages

st.set_page_config(page_title="Home and Import", page_icon="🗺️", layout="wide")
st.navigation(pages()).run()
