"""Home page — Indian market scanner. US market is under 'US Market' in the sidebar."""
import streamlit as st
from ui_common import render_market

st.set_page_config(page_title="Stock Scanner — India", page_icon="🇮🇳", layout="wide")
render_market("IN")
