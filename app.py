import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(page_title="E-commerce Sales Analysis", layout="wide")
st.title("📊 E-commerce Sales Analysis Dashboard")

# --------------------------------------------------
# Load data
# --------------------------------------------------
@st.cache_data
def load_data():
    orders = pd.read_csv("List of Orders.csv")
    details = pd.read_csv("Order Details.csv")
    targets = pd.read_csv("Sales target.csv")
    return orders, details, targets

orders, details, targets = load_data()

# --------------------------------------------------
# Preprocessing
# --------------------------------------------------
orders["Order Date"] = pd.to_datetime(
    orders["Order Date"], format="%d-%m-%Y", errors="coerce"
)

orders["Month"] = orders["Order Date"].dt.to_period("M").astype(str)

data = pd.merge(orders, details, on="Order ID", how="inner")

# Standardize column name
data = data.rename(columns={"Amount": "Sales"})

# --------------------------------------------------
# Sidebar filters
# --------------------------------------------------
st.sidebar.header("🔎 Filters")

states = st.sidebar.multiselect(
    "Select State(s)",
    options=sorted(data["State"].unique()),
    default=sorted(data["State"].unique())
)

filtered_data = data[data["State"].isin(states)]

# --------------------------------------------------
# KPIs
# --------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("💰 Total Sales", f"₹{filtered_data['Sales'].sum():,.0f}")
col2.metric("📈 Total Profit", f"₹{filtered_data['Profit'].sum():,.0f}")
col3.metric("🧾 Total Orders", filtered_data["Order ID"].nunique())

st.markdown("---")

# --------------------------------------------------
# Monthly Sales Trend
# --------------------------------------------------
st.subheader("📅 Monthly Sales Trend")

monthly_sales = filtered_data.groupby("Month")["Sales"].sum().reset_index()

fig, ax = plt.subplots()
sns.lineplot(data=monthly_sales, x="Month", y="Sales", marker="o", ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# --------------------------------------------------
# Top States by Sales
# --------------------------------------------------
st.subheader("🗺️ Top 10 States by Sales")

state_sales = (
    filtered_data.groupby("State")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

fig, ax = plt.subplots()
state_sales.plot(kind="bar", ax=ax)
st.pyplot(fig)

# --------------------------------------------------
# Sales by Category
# --------------------------------------------------
st.subheader("📦 Sales by Category")

category_sales = filtered_data.groupby("Category")["Sales"].sum()

fig, ax = plt.subplots()
category_sales.plot(kind="bar", ax=ax)
st.pyplot(fig)

# --------------------------------------------------
# Profit by Category
# --------------------------------------------------
st.subheader("📊 Profit by Category")

profit_category = filtered_data.groupby("Category")["Profit"].sum().reset_index()

fig, ax = plt.subplots()
sns.barplot(data=profit_category, x="Category", y="Profit", ax=ax)
st.pyplot(fig)

# --------------------------------------------------
# Customer Behavior
# --------------------------------------------------
st.subheader("🧍 Customer Purchase Behavior")

customer_behavior = filtered_data.groupby("Quantity")["Sales"].sum().reset_index()

fig, ax = plt.subplots()
sns.barplot(data=customer_behavior, x="Quantity", y="Sales", ax=ax)
st.pyplot(fig)

# --------------------------------------------------
# Sales Heatmap (Category vs Month)
# --------------------------------------------------
st.subheader("🔥 Sales Heatmap (Category × Month)")

sales_heatmap = filtered_data.pivot_table(
    index="Category", columns="Month", values="Sales", aggfunc="sum"
)

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(sales_heatmap, cmap="YlOrRd", linewidths=0.5, ax=ax)
st.pyplot(fig)

# --------------------------------------------------
# Segment Category Heatmap (State vs Category)
# --------------------------------------------------
st.subheader("🧩 Segment Category Heatmap (State × Category)")

segment_heatmap = filtered_data.pivot_table(
    index="State", columns="Category", values="Sales", aggfunc="sum"
)

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(segment_heatmap, cmap="Blues", linewidths=0.5, ax=ax)
st.pyplot(fig)

# --------------------------------------------------
# Correlation Heatmap
# --------------------------------------------------
st.subheader("📈 Correlation Heatmap")

corr = filtered_data[["Sales", "Profit", "Quantity"]].corr()

fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# --------------------------------------------------
# Raw Data
# --------------------------------------------------
with st.expander("📄 View Raw Data"):
    st.dataframe(filtered_data.head(50))
