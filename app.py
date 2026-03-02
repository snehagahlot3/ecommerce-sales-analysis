import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="E-commerce Sales Analysis",
    layout="wide"
)

st.title("📊 E-commerce Sales Analysis Dashboard")

# ----------------------------
# Load data
# ----------------------------
@st.cache_data
def load_data():
    orders = pd.read_csv("List of Orders.csv")
    order_details = pd.read_csv("Order Details.csv")
    sales_target = pd.read_csv("Sales target.csv")
    return orders, order_details, sales_target

orders, order_details, sales_target = load_data()

# ----------------------------
# Data preprocessing
# ----------------------------
# Convert Order Date to datetime (DD-MM-YYYY format)
orders["Order Date"] = pd.to_datetime(
    orders["Order Date"],
    format="%d-%m-%Y",
    errors="coerce"
)

# Create Month column
orders["Month"] = orders["Order Date"].dt.to_period("M").astype(str)

# Merge orders with order details
data = pd.merge(
    orders,
    order_details,
    on="Order ID",
    how="inner"
)

# ----------------------------
# Sidebar filters
# ----------------------------
st.sidebar.header("🔎 Filters")

state_filter = st.sidebar.multiselect(
    "Select State(s):",
    options=sorted(data["State"].unique()),
    default=sorted(data["State"].unique())
)

filtered_data = data[data["State"].isin(state_filter)]

# KPIs

total_sales = filtered_data["Sales"].sum()
total_profit = filtered_data["Profit"].sum()
total_orders = filtered_data["Order ID"].nunique()

col1, col2, col3 = st.columns(3)

col1.metric("💰 Total Sales", f"₹{total_sales:,.0f}")
col2.metric("📈 Total Profit", f"₹{total_profit:,.0f}")
col3.metric("🧾 Total Orders", total_orders)

st.markdown("---")

# Monthly Sales Trend

st.subheader("📅 Monthly Sales Trend")

monthly_sales = (
    filtered_data
    .groupby("Month")["Sales"]
    .sum()
    .reset_index()
)

fig1, ax1 = plt.subplots(figsize=(10, 4))
sns.lineplot(data=monthly_sales, x="Month", y="Sales", marker="o", ax=ax1)
ax1.set_xlabel("Month")
ax1.set_ylabel("Sales")
ax1.set_title("Monthly Sales Trend")
plt.xticks(rotation=45)

st.pyplot(fig1)
# Sales by State
st.subheader("
