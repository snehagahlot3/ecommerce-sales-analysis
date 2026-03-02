import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="E-commerce Sales Analysis", layout="wide")

st.title("📊 E-commerce Sales Analysis Dashboard")

# Load data
orders = pd.read_csv("List of Orders.csv")
details = pd.read_csv("Order Details.csv")
targets = pd.read_csv("Sales target.csv")

st.subheader("Dataset Preview")
st.dataframe(orders.head())

st.subheader("Monthly Sales Trend")
fig, ax = plt.subplots()
orders.groupby("Month")["Sales"].sum().plot(ax=ax)
st.pyplot(fig)
