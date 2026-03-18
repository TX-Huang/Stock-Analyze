import streamlit as st
import pandas as pd


def custom_metric(label, value, delta=None):
    delta_str = ""
    if delta:
        delta_str = f" {delta}"
    st.markdown(f"**{label}**: {value} {delta_str}")


def highlight_ret(val):
    color = ''
    if pd.isna(val):
        return ''
    if isinstance(val, (int, float)):
        # In TW, Red (#ef4444) is positive, Green (#22c55e) is negative
        color = 'color: #ef4444' if val > 0 else 'color: #22c55e'
    return color
