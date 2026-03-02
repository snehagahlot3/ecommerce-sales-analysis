import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Sales Analysis",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Dark Theme CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; color: #e0e0e0; }
    .main .block-container { padding: 2rem 2rem 2rem 2rem; }
    .metric-card {
        background: #1a1d27;
        border: 1px solid #2d3040;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value { font-size: 28px; font-weight: bold; color: #00d4ff; }
    .metric-label { font-size: 13px; color: #888; margin-top: 4px; }
    .section-header {
        font-size: 22px;
        font-weight: bold;
        color: #ffffff;
        border-left: 4px solid #00d4ff;
        padding-left: 12px;
        margin: 20px 0 15px 0;
    }
    div[data-testid="stSidebar"] { background-color: #1a1d27; }
    .stSelectbox label, .stMultiSelect label { color: #e0e0e0 !important; }
    h1, h2, h3 { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

# ── Plot Style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor':   '#1a1d27',
    'axes.labelcolor':  '#e0e0e0',
    'axes.titlecolor':  '#ffffff',
    'axes.titlesize':   13,
    'axes.titleweight': 'bold',
    'xtick.color':      '#aaaaaa',
    'ytick.color':      '#aaaaaa',
    'text.color':       '#e0e0e0',
    'grid.color':       '#2d3040',
    'legend.facecolor': '#1a1d27',
    'legend.edgecolor': '#444',
})
ACCENT = ['#00d4ff','#ff6b6b','#ffd166','#06d6a0','#a78bfa','#f97316','#ec4899','#84cc16']
MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        orders  = pd.read_csv('List of Orders.csv')
        details = pd.read_csv('Order Details.csv')
        df = orders.merge(details, on='Order ID')
        df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
        df['order_date']  = df['Order Date']
        df['order_id']    = df['Order ID']
        df['customer_id'] = df['CustomerName']
        df['category']    = df['Category']
        df['region']      = df['State']
        df['revenue']     = df['Amount']
        df['quantity']    = df['Quantity']
        df['unit_price']  = df['Amount']
        df['discount']    = 0.0
        df['segment']     = 'Consumer'
        df['month']       = df['Order Date'].dt.month
        df['month_name']  = df['Order Date'].dt.strftime('%b')
        df['week']        = df['Order Date'].dt.isocalendar().week.astype(int)
        df['dow']         = df['Order Date'].dt.dayofweek
        df['dow_name']    = df['Order Date'].dt.strftime('%a')
        df['quarter']     = df['Order Date'].dt.quarter
        return df, True
    except:
        # Synthetic fallback
        np.random.seed(42)
        n = 1500
        categories   = ['Electronics','Apparel','Home & Garden','Sports','Beauty','Books','Toys','Food']
        cat_weights  = [0.22,0.20,0.14,0.12,0.10,0.08,0.08,0.06]
        regions      = ['Madhya Pradesh','Maharashtra','Uttar Pradesh','Delhi','Karnataka']
        dates        = pd.date_range('2018-01-01','2020-12-31',periods=n)
        dates        = dates[np.random.choice(len(dates),n,replace=True)]
        df = pd.DataFrame({
            'order_date' : dates,
            'order_id'   : ['ORD-'+str(i).zfill(5) for i in range(n)],
            'customer_id': np.random.choice(['CUST-'+str(i).zfill(4) for i in range(400)],n),
            'category'   : np.random.choice(categories,n,p=cat_weights),
            'region'     : np.random.choice(regions,n),
            'segment'    : 'Consumer',
            'quantity'   : np.random.randint(1,10,n),
            'unit_price' : np.round(np.random.lognormal(3.5,0.9,n),2),
            'discount'   : 0.0,
        })
        df['revenue']     = np.round(df['quantity']*df['unit_price'],2)
        df['month']       = df['order_date'].dt.month
        df['month_name']  = df['order_date'].dt.strftime('%b')
        df['week']        = df['order_date'].dt.isocalendar().week.astype(int)
        df['dow']         = df['order_date'].dt.dayofweek
        df['dow_name']    = df['order_date'].dt.strftime('%a')
        df['quarter']     = df['order_date'].dt.quarter
        return df, False

df, real_data = load_data()

# ── RFM ───────────────────────────────────────────────────────────────────────
@st.cache_data
def compute_rfm(df):
    snapshot = df['order_date'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('customer_id').agg(
        recency  =('order_date',  lambda x: (snapshot-x.max()).days),
        frequency=('order_id',    'count'),
        monetary =('revenue',     'sum')
    ).reset_index()
    rfm['R_score'] = pd.qcut(rfm['recency'],  5, labels=[5,4,3,2,1]).astype(int)
    rfm['F_score'] = pd.qcut(rfm['frequency'].rank(method='first'),5,labels=[1,2,3,4,5]).astype(int)
    rfm['M_score'] = pd.qcut(rfm['monetary'], 5, labels=[1,2,3,4,5]).astype(int)
    rfm['RFM_total'] = rfm['R_score']+rfm['F_score']+rfm['M_score']
    def seg(s):
        if s>=13: return 'Champions'
        elif s>=10: return 'Loyal Customers'
        elif s>=7:  return 'Potential Loyalists'
        elif s>=5:  return 'At Risk'
        else: return 'Lost'
    rfm['segment'] = rfm['RFM_total'].apply(seg)
    return rfm

rfm = compute_rfm(df)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.shields.io/badge/🛒_E--Commerce-Analysis-00d4ff?style=for-the-badge", width=250)
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Filters")

all_categories = sorted(df['category'].unique().tolist())
all_regions    = sorted(df['region'].unique().tolist())

selected_cats    = st.sidebar.multiselect("📦 Category", all_categories, default=all_categories)
selected_regions = st.sidebar.multiselect("🌍 Region",   all_regions,    default=all_regions)

years = sorted(df['order_date'].dt.year.unique().tolist())
selected_years = st.sidebar.multiselect("📅 Year", years, default=years)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Overview",
    "📈 Monthly Trends",
    "🏆 Best Categories",
    "🌡️ Heatmaps",
    "👤 Customer Behavior",
    "💰 Revenue Contribution"
])
st.sidebar.markdown("---")
if real_data:
    st.sidebar.success("✅ Using real Kaggle data")
else:
    st.sidebar.warning("⚠️ Using synthetic demo data")

# ── Filter Data ───────────────────────────────────────────────────────────────
filtered = df[
    (df['category'].isin(selected_cats)) &
    (df['region'].isin(selected_regions)) &
    (df['order_date'].dt.year.isin(selected_years))
]

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("<h1 style='text-align:center'>🛒 E-Commerce Sales Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#888'>Exploratory Data Analysis | Python · Pandas · Seaborn · Streamlit</p>", unsafe_allow_html=True)
    st.markdown("---")

    # KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    total_rev   = filtered['revenue'].sum()
    total_orders= len(filtered)
    aov         = filtered['revenue'].mean()
    unique_cust = filtered['customer_id'].nunique()
    top_cat     = filtered.groupby('category')['revenue'].sum().idxmax()

    for col, val, label in zip(
        [col1,col2,col3,col4,col5],
        [f"₹{total_rev:,.0f}", f"{total_orders:,}", f"₹{aov:,.0f}", f"{unique_cust:,}", top_cat],
        ["💰 Total Revenue","📦 Total Orders","🧾 Avg Order Value","👥 Unique Customers","🏆 Top Category"]
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='section-header'>📊 Quick Overview</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # Revenue by category
    with col1:
        cat_rev = filtered.groupby('category')['revenue'].sum().sort_values()
        fig, ax = plt.subplots(figsize=(7,4))
        ax.barh(cat_rev.index, cat_rev.values/1000, color=ACCENT[:len(cat_rev)], edgecolor='none')
        ax.set_title('Revenue by Category (₹ 000s)')
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f'₹{x:.0f}K'))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Revenue by region
    with col2:
        reg_rev = filtered.groupby('region')['revenue'].sum().sort_values(ascending=False).head(8)
        fig, ax = plt.subplots(figsize=(7,4))
        ax.bar(reg_rev.index, reg_rev.values/1000, color=ACCENT[:len(reg_rev)], edgecolor='none', width=0.6)
        ax.set_title('Top Regions by Revenue (₹ 000s)')
        ax.set_xticklabels(reg_rev.index, rotation=35, ha='right')
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f'₹{x:.0f}K'))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MONTHLY TRENDS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Monthly Trends":
    st.markdown("<div class='section-header'>📈 Monthly Sales Trends</div>", unsafe_allow_html=True)

    monthly = (filtered.groupby(['month','month_name'])
               .agg(revenue=('revenue','sum'), orders=('order_id','count'))
               .reset_index().sort_values('month'))
    monthly['revenue_k']  = monthly['revenue']/1000
    monthly['mom_growth'] = monthly['revenue'].pct_change()*100
    monthly['cum_rev']    = monthly['revenue'].cumsum()/1000

    fig, axes = plt.subplots(2,2,figsize=(14,9))
    fig.patch.set_facecolor('#0f1117')

    # Revenue
    ax = axes[0,0]
    ax.plot(range(len(monthly)), monthly['revenue_k'], marker='o', lw=2.5, color=ACCENT[0], markersize=8, markerfacecolor='white', zorder=5)
    ax.fill_between(range(len(monthly)), monthly['revenue_k'], alpha=0.15, color=ACCENT[0])
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(monthly['month_name'])
    ax.set_title('Monthly Revenue (₹ 000s)')
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f'₹{x:.0f}K'))

    # Orders
    ax = axes[0,1]
    ax.plot(range(len(monthly)), monthly['orders'], marker='s', lw=2.5, color=ACCENT[1], markersize=8, markerfacecolor='white', zorder=5)
    ax.fill_between(range(len(monthly)), monthly['orders'], alpha=0.15, color=ACCENT[1])
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(monthly['month_name'])
    ax.set_title('Monthly Order Volume')

    # MoM Growth
    ax = axes[1,0]
    colors_g = [ACCENT[3] if v>=0 else ACCENT[1] for v in monthly['mom_growth'].fillna(0)]
    ax.bar(monthly['month_name'], monthly['mom_growth'].fillna(0), color=colors_g, edgecolor='none', width=0.6)
    ax.axhline(0, color='#555', lw=1)
    ax.set_title('Month-over-Month Growth (%)')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    # Cumulative
    ax = axes[1,1]
    ax.plot(range(len(monthly)), monthly['cum_rev'], marker='D', lw=2.5, color=ACCENT[2], markersize=8, markerfacecolor='white', zorder=5)
    ax.fill_between(range(len(monthly)), monthly['cum_rev'], alpha=0.15, color=ACCENT[2])
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(monthly['month_name'])
    ax.set_title('Cumulative Revenue (₹ 000s)')
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f'₹{x:.0f}K'))

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.markdown("**📋 Monthly Data Table**")
    st.dataframe(monthly[['month_name','revenue','orders','mom_growth']].rename(columns={
        'month_name':'Month','revenue':'Revenue (₹)','orders':'Orders','mom_growth':'MoM Growth %'
    }).set_index('Month'), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: BEST CATEGORIES
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Best Categories":
    st.markdown("<div class='section-header'>🏆 Best-Selling Categories</div>", unsafe_allow_html=True)

    cat_stats = (filtered.groupby('category')
                 .agg(revenue=('revenue','sum'), orders=('order_id','count'), units=('quantity','sum'))
                 .reset_index().sort_values('revenue', ascending=False))
    cat_stats['revenue_share'] = (cat_stats['revenue']/cat_stats['revenue'].sum()*100).round(1)
    cat_stats['revenue_k']     = cat_stats['revenue']/1000

    fig, axes = plt.subplots(1,3,figsize=(16,5))
    fig.patch.set_facecolor('#0f1117')

    # Revenue
    ax = axes[0]
    sorted_c = cat_stats.sort_values('revenue')
    ax.barh(sorted_c['category'], sorted_c['revenue_k'], color=ACCENT[:len(cat_stats)], edgecolor='none', height=0.6)
    ax.set_title('Revenue by Category (₹ 000s)')
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f'₹{x:.0f}K'))

    # Units
    ax = axes[1]
    sorted_u = cat_stats.sort_values('units')
    ax.barh(sorted_u['category'], sorted_u['units'], color=ACCENT[:len(cat_stats)], edgecolor='none', height=0.6)
    ax.set_title('Units Sold')

    # Share
    ax = axes[2]
    ax.bar(cat_stats['category'], cat_stats['revenue_share'], color=ACCENT[:len(cat_stats)], edgecolor='none', width=0.6)
    ax.set_title('Revenue Share (%)')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_xticklabels(cat_stats['category'], rotation=35, ha='right')
    for bar, val in zip(ax.patches, cat_stats['revenue_share']):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f'{val}%', ha='center', fontsize=9, color='#ccc')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.dataframe(cat_stats[['category','revenue','units','orders','revenue_share']].rename(columns={
        'category':'Category','revenue':'Revenue (₹)','units':'Units','orders':'Orders','revenue_share':'Share %'
    }).set_index('Category'), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: HEATMAPS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🌡️ Heatmaps":
    st.markdown("<div class='section-header'>🌡️ Sales Heatmaps</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        pivot1 = filtered.pivot_table(values='revenue',index='category',columns='month',aggfunc='sum').fillna(0)
        pivot1.columns = MONTHS[:len(pivot1.columns)]
        fig, ax = plt.subplots(figsize=(8,5))
        fig.patch.set_facecolor('#0f1117')
        sns.heatmap(pivot1/1000, ax=ax, cmap='YlOrRd', annot=True, fmt='.0f',
                    linewidths=0.4, linecolor='#0f1117',
                    cbar_kws={'label':'Revenue (₹ 000s)','shrink':0.85}, annot_kws={'size':8})
        ax.set_title('Revenue: Category × Month')
        ax.set_xlabel('Month')
        ax.set_ylabel('Category')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        day_order = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        pivot2 = filtered.pivot_table(values='order_id',index='dow_name',columns='month',aggfunc='count').reindex(day_order).fillna(0)
        pivot2.columns = MONTHS[:len(pivot2.columns)]
        fig, ax = plt.subplots(figsize=(8,5))
        fig.patch.set_facecolor('#0f1117')
        sns.heatmap(pivot2, ax=ax, cmap='Blues', annot=True, fmt='.0f',
                    linewidths=0.4, linecolor='#0f1117',
                    cbar_kws={'label':'Order Count','shrink':0.85}, annot_kws={'size':8})
        ax.set_title('Orders: Day-of-Week × Month')
        ax.set_xlabel('Month')
        ax.set_ylabel('Day')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown("<div class='section-header'>🔗 Feature Correlation Heatmap</div>", unsafe_allow_html=True)
    corr_cols = ['quantity','unit_price','discount','revenue','month','dow']
    available = [c for c in corr_cols if c in filtered.columns]
    corr = filtered[available].corr()
    fig, ax = plt.subplots(figsize=(8,6))
    fig.patch.set_facecolor('#0f1117')
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, ax=ax, mask=mask, cmap='coolwarm', vmin=-1, vmax=1,
                annot=True, fmt='.2f', annot_kws={'size':11},
                linewidths=0.5, linecolor='#1a1d27',
                cbar_kws={'label':'Pearson Correlation','shrink':0.8})
    ax.set_title('Feature Correlation Matrix')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: CUSTOMER BEHAVIOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "👤 Customer Behavior":
    st.markdown("<div class='section-header'>👤 Customer Behavior & RFM Analysis</div>", unsafe_allow_html=True)

    seg_order  = ['Champions','Loyal Customers','Potential Loyalists','At Risk','Lost']
    seg_counts = rfm['segment'].value_counts().reindex(seg_order).fillna(0)
    seg_revenue= rfm.groupby('segment')['monetary'].sum().reindex(seg_order).fillna(0)

    # KPIs
    champions = rfm[rfm['segment']=='Champions']
    col1,col2,col3,col4 = st.columns(4)
    for col, val, label in zip(
        [col1,col2,col3,col4],
        [len(rfm), len(champions), f"₹{champions['monetary'].sum():,.0f}", f"{rfm['frequency'].mean():.1f}"],
        ["👥 Total Customers","👑 Champions","💰 Champion Revenue","📦 Avg Orders/Customer"]
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    fig, axes = plt.subplots(1,3,figsize=(16,5))
    fig.patch.set_facecolor('#0f1117')

    # Segment count
    ax = axes[0]
    bars = ax.bar(seg_counts.index, seg_counts.values, color=ACCENT[:5], edgecolor='none', width=0.6)
    ax.set_title('Customers by RFM Segment')
    ax.set_xticklabels(seg_counts.index, rotation=25, ha='right')
    for bar, val in zip(bars, seg_counts.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, str(int(val)), ha='center', fontsize=10, color='#ccc')

    # Revenue by segment
    ax = axes[1]
    bars = ax.bar(seg_revenue.index, seg_revenue.values/1000, color=ACCENT[:5], edgecolor='none', width=0.6)
    ax.set_title('Revenue by Segment (₹ 000s)')
    ax.set_xticklabels(seg_revenue.index, rotation=25, ha='right')
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f'₹{x:.0f}K'))

    # RFM scatter
    ax = axes[2]
    colors_map = {'Champions':ACCENT[0],'Loyal Customers':ACCENT[3],'Potential Loyalists':ACCENT[2],'At Risk':ACCENT[1],'Lost':'#888'}
    for seg in seg_order:
        subset = rfm[rfm['segment']==seg]
        ax.scatter(subset['frequency'], subset['monetary']/1000, label=seg,
                   color=colors_map.get(seg,'#888'), alpha=0.7, s=40)
    ax.set_title('Frequency vs Monetary Value')
    ax.set_xlabel('Purchase Frequency')
    ax.set_ylabel('Revenue (₹ 000s)')
    ax.legend(fontsize=7)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.markdown("**📋 RFM Segment Summary**")
    summary = rfm.groupby('segment')[['recency','frequency','monetary']].mean().round(1).reindex(seg_order)
    st.dataframe(summary.rename(columns={'recency':'Avg Recency (days)','frequency':'Avg Frequency','monetary':'Avg Revenue (₹)'}),
                 use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: REVENUE CONTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Revenue Contribution":
    st.markdown("<div class='section-header'>💰 Revenue Contribution Analysis</div>", unsafe_allow_html=True)

    fig, axes = plt.subplots(2,2,figsize=(14,10))
    fig.patch.set_facecolor('#0f1117')

    # Region
    ax = axes[0,0]
    reg_rev = filtered.groupby('region')['revenue'].sum().sort_values(ascending=False).head(10)
    bars = ax.bar(reg_rev.index, reg_rev.values/1000, color=ACCENT[:len(reg_rev)], edgecolor='none', width=0.6)
    ax.set_title('Revenue by Region (₹ 000s)')
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f'₹{x:.0f}K'))
    ax.set_xticklabels(reg_rev.index, rotation=35, ha='right')

    # Stacked category × region
    ax = axes[0,1]
    top_regions = filtered.groupby('region')['revenue'].sum().nlargest(5).index
    reg_cat = filtered[filtered['region'].isin(top_regions)].pivot_table(
        values='revenue', index='region', columns='category', aggfunc='sum').fillna(0)/1000
    bottom = np.zeros(len(reg_cat))
    for i, cat in enumerate(reg_cat.columns):
        ax.bar(reg_cat.index, reg_cat[cat].values, bottom=bottom, label=cat, color=ACCENT[i%len(ACCENT)], edgecolor='none')
        bottom += reg_cat[cat].values
    ax.set_title('Stacked Revenue: Top Regions × Category')
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f'₹{x:.0f}K'))
    ax.set_xticklabels(reg_cat.index, rotation=25, ha='right')
    ax.legend(loc='upper right', fontsize=7, ncol=2)

    # Quarterly by top categories
    ax = axes[1,0]
    top4 = filtered.groupby('category')['revenue'].sum().nlargest(4).index.tolist()
    q_cat = filtered[filtered['category'].isin(top4)].pivot_table(
        values='revenue', index='quarter', columns='category', aggfunc='sum').fillna(0)/1000
    for i, cat in enumerate(top4):
        if cat in q_cat.columns:
            ax.plot(q_cat.index, q_cat[cat], marker='o', lw=2.5, label=cat, color=ACCENT[i], markersize=8, markerfacecolor='white')
    ax.set_title('Quarterly Revenue — Top 4 Categories')
    ax.set_xlabel('Quarter')
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f'₹{x:.0f}K'))
    ax.set_xticks([1,2,3,4])
    ax.set_xticklabels(['Q1','Q2','Q3','Q4'])
    ax.legend(fontsize=9)

    # Category revenue share bar
    ax = axes[1,1]
    cat_share = filtered.groupby('category')['revenue'].sum().sort_values(ascending=False)
    cat_pct   = (cat_share/cat_share.sum()*100).round(1)
    bars = ax.barh(cat_share.index[::-1], cat_pct.values[::-1], color=ACCENT[:len(cat_share)], edgecolor='none', height=0.6)
    ax.set_title('Category Revenue Share (%)')
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    for bar, val in zip(bars, cat_pct.values[::-1]):
        ax.text(val+0.3, bar.get_y()+bar.get_height()/2, f'{val}%', va='center', fontsize=9, color='#ccc')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Executive Summary
    st.markdown("---")
    st.markdown("<div class='section-header'>📋 Executive Summary</div>", unsafe_allow_html=True)
    champions = rfm[rfm['segment']=='Champions']
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | 💰 Total Revenue | ₹{filtered['revenue'].sum():,.0f} |
        | 📦 Total Orders | {len(filtered):,} |
        | 🧾 Avg Order Value | ₹{filtered['revenue'].mean():,.2f} |
        | 👥 Unique Customers | {filtered['customer_id'].nunique():,} |
        """)
    with col2:
        top_cat    = filtered.groupby('category')['revenue'].sum().idxmax()
        top_region = filtered.groupby('region')['revenue'].sum().idxmax()
        best_month = filtered.groupby('month_name')['revenue'].sum().idxmax()
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | 🏆 Top Category | {top_cat} |
        | 🌍 Top Region | {top_region} |
        | 📅 Best Month | {best_month} |
        | 👑 Champion Customers | {len(champions)} ({len(champions)/max(len(rfm),1)*100:.1f}%) |
        """)
