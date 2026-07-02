"""US market scanner page."""
import streamlit as st
from ui_common import render_market

st.set_page_config(page_title="Stock Scanner — US", page_icon="🇺🇸", layout="wide")
render_market("US")
