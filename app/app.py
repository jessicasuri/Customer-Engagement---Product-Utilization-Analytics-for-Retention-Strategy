import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import os
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Customer Engagement & Retention Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: 700; color: #1a3a5c;
        border-bottom: 3px solid #1a3a5c; padding-bottom: 0.4rem; margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.3rem; font-weight: 600; color: #1a3a5c;
        margin-top: 1.5rem; margin-bottom: 0.5rem;
    }
    .kpi-card {
        background: #f0f4fa; border-left: 5px solid #1a3a5c;
        border-radius: 8px; padding: 1rem 1.2rem; margin-bottom: 0.5rem;
    }
    .kpi-value { font-size: 1.8rem; font-weight: 700; color: #1a3a5c; }
    .kpi-label { font-size: 0.85rem; color: #555; margin-top: 0.2rem; }
    .kpi-formula { font-size: 0.78rem; color: #888; margin-top: 0.15rem; font-style: italic; }
    .insight-box {
        background: #fffbea; border-left: 4px solid #f0a500;
        border-radius: 6px; padding: 0.8rem 1rem; margin: 0.6rem 0;
        font-size: 0.92rem; color: #333;
    }
    .ml-box {
        background: #f0f7ff; border-left: 4px solid #2980b9;
        border-radius: 6px; padding: 0.8rem 1rem; margin: 0.6rem 0;
        font-size: 0.92rem; color: #333;
    }
</style>
""", unsafe_allow_html=True)


# ── Data Loading & Feature Engineering ──────────────────────────────────────
@st.cache_data
def load_data():
    # Works for: local run, app/ subfolder, and Streamlit Cloud
    possible_paths = [
        "European_Bank.csv",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "European_Bank.csv"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "European_Bank.csv"),
    ]
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break
    if df is None:
        st.error("❌ European_Bank.csv not found. Please place it in the project root directory.")
        st.stop()

    med_bal = df['Balance'].median()

    def segment(row):
        active = row['IsActiveMember'] == 1
        multi  = row['NumOfProducts'] >= 2
        hi_bal = row['Balance'] > med_bal
        if active and multi:        return 'Loyal Core'
        elif active and not multi:  return 'Cross-sell Target'
        elif not active and hi_bal: return 'At-Risk Premium'
        else:                       return 'Silent Churner'

    df['Segment'] = df.apply(segment, axis=1)

    df['RSI'] = (
        df['IsActiveMember'] * 30 +
        df['NumOfProducts'].clip(upper=3) / 3 * 30 +
        df['HasCrCard'] * 10 +
        (df['Tenure'] / df['Tenure'].max()) * 20 +
        ((df['Balance'] > med_bal).astype(int)) * 10
    ).round(1)

    return df


@st.cache_data
def train_ml_model(df):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import LabelEncoder

    df_ml = df.copy()
    le = LabelEncoder()
    df_ml['Geography_enc'] = le.fit_transform(df_ml['Geography'])
    df_ml['Gender_enc']    = le.fit_transform(df_ml['Gender'])

    features = ['CreditScore','Geography_enc','Gender_enc','Age','Tenure',
                'Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','RSI']
    X = df_ml[features]
    y = df_ml['Exited']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    return model, acc, report, importance_df, features


df_full = load_data()

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🔎 Filters")
geo_opts   = ['All'] + sorted(df_full['Geography'].unique().tolist())
geo        = st.sidebar.selectbox("Geography", geo_opts)
engagement = st.sidebar.radio("Engagement Status", ['All', 'Active (1)', 'Inactive (0)'])
prod_range = st.sidebar.slider("Number of Products", 1, 4, (1, 4))
bal_max    = int(df_full['Balance'].max())
bal_range  = st.sidebar.slider("Balance Range (€)", 0, bal_max, (0, bal_max), step=1000)
sal_max    = int(df_full['EstimatedSalary'].max())
sal_range  = st.sidebar.slider("Salary Range (€)", 0, sal_max, (0, sal_max), step=5000)

df = df_full.copy()
if geo != 'All':              df = df[df['Geography'] == geo]
if engagement == 'Active (1)':   df = df[df['IsActiveMember'] == 1]
elif engagement == 'Inactive (0)': df = df[df['IsActiveMember'] == 0]
df = df[df['NumOfProducts'].between(prod_range[0], prod_range[1])]
df = df[df['Balance'].between(bal_range[0], bal_range[1])]
df = df[df['EstimatedSalary'].between(sal_range[0], sal_range[1])]

st.sidebar.markdown(f"**{len(df):,} customers** selected")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Export Filtered Data")
csv_export = df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="⬇️ Download CSV",
    data=csv_export,
    file_name="filtered_customers.csv",
    mime="text/csv"
)

# ── KPI Calculations ─────────────────────────────────────────────────────────
churn_rate     = df['Exited'].mean() * 100 if len(df) else 0
active_churn   = df[df['IsActiveMember']==1]['Exited'].mean() * 100 if len(df[df['IsActiveMember']==1]) else 0
inactive_churn = df[df['IsActiveMember']==0]['Exited'].mean() * 100 if len(df[df['IsActiveMember']==0]) else 0
ERR            = (inactive_churn / active_churn) if active_churn > 0 else 0
prod_churn     = df.groupby('NumOfProducts')['Exited'].mean()
PDI            = (1 - prod_churn.get(2, prod_churn.mean())) * 100
med_bal        = df_full['Balance'].median()
hi_bal_in      = df[(df['Balance'] > med_bal) & (df['IsActiveMember']==0)]
HBDR           = hi_bal_in['Exited'].mean() * 100 if len(hi_bal_in) else 0
cc_churn       = df[df['HasCrCard']==1]['Exited'].mean() * 100 if len(df[df['HasCrCard']==1]) else 0
nocc_churn     = df[df['HasCrCard']==0]['Exited'].mean() * 100 if len(df[df['HasCrCard']==0]) else 0
CSS            = nocc_churn - cc_churn
avg_RSI        = df['RSI'].mean() if len(df) else 0

seg_colors = {
    'Loyal Core':        '#27ae60',
    'Cross-sell Target': '#2980b9',
    'At-Risk Premium':   '#e67e22',
    'Silent Churner':    '#c0392b'
}

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🏦 Customer Engagement & Retention Analytics</div>', unsafe_allow_html=True)
st.markdown("**European Central Bank | Behavioral Churn Analysis Dashboard**")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Engagement Overview",
    "📦 Product Utilization",
    "⚠️ At-Risk Customers",
    "🏆 Retention Scoring",
    "🤖 ML Churn Prediction"
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — ENGAGEMENT OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{churn_rate:.1f}%</div><div class="kpi-label">Overall Churn Rate</div><div class="kpi-formula">Exited=1 ÷ Total Customers</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{ERR:.2f}x</div><div class="kpi-label">Engagement Retention Ratio</div><div class="kpi-formula">Inactive Churn % ÷ Active Churn %</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{PDI:.1f}%</div><div class="kpi-label">Product Depth Index</div><div class="kpi-formula">1 − Churn Rate (2-product customers)</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{HBDR:.1f}%</div><div class="kpi-label">High-Balance Disengagement Rate</div><div class="kpi-formula">Churn % | Inactive & Balance > Median</div></div>', unsafe_allow_html=True)
    with c5:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{avg_RSI:.1f}</div><div class="kpi-label">Avg Relationship Strength Index</div><div class="kpi-formula">Active(30)+Products(30)+Card(10)+Tenure(20)+Balance(10)</div></div>', unsafe_allow_html=True)

    with st.expander("ℹ️ KPI Calculation Methodology"):
        st.markdown("""
| KPI | Formula | What it measures |
|---|---|---|
| **Engagement Retention Ratio (ERR)** | Inactive Churn Rate ÷ Active Churn Rate | How many times more likely inactive members are to churn vs active |
| **Product Depth Index (PDI)** | (1 − Churn Rate of 2-product customers) × 100 | Retention strength at the optimal product depth (2 products) |
| **High-Balance Disengagement Rate (HBDR)** | Churn % among customers where Balance > Median AND IsActiveMember = 0 | Silent churn risk among premium but disengaged customers |
| **Credit Card Stickiness Score (CSS)** | Churn % (no card) − Churn % (has card) | Marginal retention benefit of credit card ownership |
| **Relationship Strength Index (RSI)** | IsActive×30 + (Products/3)×30 + HasCard×10 + (Tenure/MaxTenure)×20 + HighBalance×10 | Composite loyalty score per customer (0–100) |
        """)

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-header">Churn Rate by Engagement Status</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(['Active Members','Inactive Members'], [active_churn, inactive_churn],
                      color=['#27ae60','#c0392b'], width=0.5, edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars, [active_churn, inactive_churn]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f'{val:.1f}%', ha='center', fontweight='bold', fontsize=12)
        ax.set_ylabel('Churn Rate (%)', fontsize=11)
        ax.set_title(f'Engagement Retention Ratio: {ERR:.2f}x', fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(active_churn, inactive_churn) * 1.3)
        ax.spines[['top','right']].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3, linestyle='--'); ax.set_axisbelow(True)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown(f'<div class="insight-box">💡 Inactive members churn at <b>{inactive_churn:.1f}%</b> vs <b>{active_churn:.1f}%</b> for active — <b>{ERR:.2f}x higher risk</b>.</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="section-header">Customer Engagement Segments</div>', unsafe_allow_html=True)
        seg_counts = df['Segment'].value_counts()
        seg_churn  = df.groupby('Segment')['Exited'].mean() * 100
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        segs = seg_counts.index.tolist()
        wedges, texts, autotexts = ax2.pie(
            seg_counts.values, labels=segs,
            colors=[seg_colors.get(s,'#888') for s in segs],
            autopct='%1.1f%%', pctdistance=0.75, startangle=140,
            wedgeprops={'edgecolor':'white','linewidth':2}
        )
        for at in autotexts: at.set_fontsize(10); at.set_fontweight('bold'); at.set_color('white')
        for t in texts: t.set_fontsize(9)
        ax2.set_title('Segment Distribution', fontsize=12, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig2); plt.close()
        st.markdown("**Segment Churn Rates:**")
        for seg in ['Loyal Core','Cross-sell Target','At-Risk Premium','Silent Churner']:
            rate = seg_churn.get(seg, 0); cnt = seg_counts.get(seg, 0)
            color = '#27ae60' if rate < 15 else ('#e67e22' if rate < 25 else '#c0392b')
            st.markdown(f"• **{seg}** — <span style='color:{color};font-weight:600'>{rate:.1f}% churn</span> ({cnt:,} customers)", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Churn by Geography & Gender</div>', unsafe_allow_html=True)
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        geo_churn = df.groupby('Geography')['Exited'].mean() * 100
        fig3, ax3 = plt.subplots(figsize=(5, 3.5))
        bars3 = ax3.barh(geo_churn.index, geo_churn.values,
                         color=['#1a3a5c','#2980b9','#85c1e9'][:len(geo_churn)], edgecolor='white')
        for bar, val in zip(bars3, geo_churn.values):
            ax3.text(val+0.3, bar.get_y()+bar.get_height()/2, f'{val:.1f}%', va='center', fontweight='bold', fontsize=11)
        ax3.set_xlabel('Churn Rate (%)'); ax3.set_title('Churn Rate by Country', fontsize=11, fontweight='bold')
        ax3.spines[['top','right']].set_visible(False); ax3.set_xlim(0, geo_churn.max()*1.3)
        plt.tight_layout(); st.pyplot(fig3); plt.close()
    with col_g2:
        gen_churn = df.groupby('Gender')['Exited'].mean() * 100
        fig4, ax4 = plt.subplots(figsize=(5, 3.5))
        bars4 = ax4.bar(gen_churn.index, gen_churn.values, color=['#9b59b6','#e91e8c'], width=0.4, edgecolor='white')
        for bar, val in zip(bars4, gen_churn.values):
            ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f'{val:.1f}%', ha='center', fontweight='bold', fontsize=12)
        ax4.set_ylabel('Churn Rate (%)'); ax4.set_title('Churn Rate by Gender', fontsize=11, fontweight='bold')
        ax4.spines[['top','right']].set_visible(False); ax4.set_ylim(0, gen_churn.max()*1.3)
        ax4.yaxis.grid(True, alpha=0.3, linestyle='--'); ax4.set_axisbelow(True)
        plt.tight_layout(); st.pyplot(fig4); plt.close()


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — PRODUCT UTILIZATION
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Product Depth vs Churn Rate</div>', unsafe_allow_html=True)
    prod_stats = df.groupby('NumOfProducts').agg(
        Churn_Rate=('Exited','mean'), Customer_Count=('Exited','count')).reset_index()
    prod_stats['Churn_Rate'] *= 100

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        p_colors = ['#27ae60' if x <= 2 else '#c0392b' for x in prod_stats['NumOfProducts']]
        bars5 = ax5.bar([f'{p} Product{"s" if p>1 else ""}' for p in prod_stats['NumOfProducts']],
                        prod_stats['Churn_Rate'], color=p_colors, edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars5, prod_stats['Churn_Rate']):
            ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{val:.1f}%', ha='center', fontweight='bold', fontsize=11)
        ax5.set_ylabel('Churn Rate (%)'); ax5.set_title('Churn Rate by Number of Products', fontsize=12, fontweight='bold')
        ax5.spines[['top','right']].set_visible(False); ax5.yaxis.grid(True, alpha=0.3, linestyle='--'); ax5.set_axisbelow(True)
        ax5.legend(handles=[mpatches.Patch(color='#27ae60',label='Low Risk'), mpatches.Patch(color='#c0392b',label='High Risk')], fontsize=9)
        plt.tight_layout(); st.pyplot(fig5); plt.close()

    with col_p2:
        fig6, ax6 = plt.subplots(figsize=(6, 4))
        ax6b = ax6.twinx()
        ax6.bar(prod_stats['NumOfProducts'].astype(str), prod_stats['Customer_Count'], color='#2980b9', alpha=0.7, label='Customer Count')
        ax6b.plot(prod_stats['NumOfProducts'].astype(str), prod_stats['Churn_Rate'], 'ro-', linewidth=2.5, markersize=8, label='Churn Rate %')
        ax6.set_ylabel('Number of Customers', color='#2980b9'); ax6b.set_ylabel('Churn Rate (%)', color='red')
        ax6.set_title('Customer Count vs Churn Rate', fontsize=12, fontweight='bold')
        ax6.spines[['top']].set_visible(False)
        lines1, labels1 = ax6.get_legend_handles_labels(); lines2, labels2 = ax6b.get_legend_handles_labels()
        ax6.legend(lines1+lines2, labels1+labels2, fontsize=9, loc='upper left')
        plt.tight_layout(); st.pyplot(fig6); plt.close()

    st.markdown("---")
    col_p3, col_p4 = st.columns(2)
    with col_p3:
        st.markdown('<div class="section-header">Single vs Multi-Product Retention</div>', unsafe_allow_html=True)
        df['Product_Group'] = df['NumOfProducts'].apply(lambda x: 'Single Product' if x==1 else ('Multi Product (2)' if x==2 else 'Over-Subscribed (3+)'))
        pg_churn = df.groupby('Product_Group')['Exited'].mean() * 100
        pg_map   = {'Single Product':'#e67e22','Multi Product (2)':'#27ae60','Over-Subscribed (3+)':'#c0392b'}
        fig7, ax7 = plt.subplots(figsize=(6, 4))
        bars7 = ax7.bar(list(pg_map.keys()), [pg_churn.get(k,0) for k in pg_map], color=list(pg_map.values()), edgecolor='white', linewidth=1.5)
        for bar, k in zip(bars7, pg_map):
            ax7.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f'{pg_churn.get(k,0):.1f}%', ha='center', fontweight='bold', fontsize=11)
        ax7.set_ylabel('Churn Rate (%)'); ax7.set_title('Single vs Multi-Product Churn', fontsize=11, fontweight='bold')
        ax7.spines[['top','right']].set_visible(False); ax7.yaxis.grid(True, alpha=0.3, linestyle='--'); ax7.set_axisbelow(True)
        plt.xticks(fontsize=9); plt.tight_layout(); st.pyplot(fig7); plt.close()
        st.markdown(f'<div class="insight-box">💡 2-product customers churn at only <b>{pg_churn.get("Multi Product (2)",0):.1f}%</b> vs <b>{pg_churn.get("Single Product",0):.1f}%</b> for single-product — the optimal retention sweet spot.</div>', unsafe_allow_html=True)

    with col_p4:
        st.markdown('<div class="section-header">Product Mix by Geography</div>', unsafe_allow_html=True)
        geo_prod     = df.groupby(['Geography','NumOfProducts']).size().unstack(fill_value=0)
        geo_prod_pct = geo_prod.div(geo_prod.sum(axis=1), axis=0) * 100
        fig8, ax8 = plt.subplots(figsize=(6, 4))
        geo_prod_pct.plot(kind='bar', ax=ax8, color=['#1a3a5c','#2980b9','#85c1e9','#d6eaf8'][:len(geo_prod_pct.columns)],
                          edgecolor='white', linewidth=0.8)
        ax8.set_ylabel('% of Customers'); ax8.set_title('Product Distribution by Country', fontsize=11, fontweight='bold')
        ax8.spines[['top','right']].set_visible(False); ax8.legend(title='# Products', fontsize=8)
        ax8.yaxis.grid(True, alpha=0.3, linestyle='--'); ax8.set_axisbelow(True)
        plt.xticks(rotation=0, fontsize=10); plt.tight_layout(); st.pyplot(fig8); plt.close()

    st.markdown("---")
    st.markdown('<div class="section-header">Product Utilization Summary Table</div>', unsafe_allow_html=True)
    summary = df.groupby('NumOfProducts').agg(
        Total_Customers=('Exited','count'), Churned=('Exited','sum'),
        Retained=('Exited', lambda x: (x==0).sum()),
        Churn_Rate_Pct=('Exited','mean'), Avg_Balance=('Balance','mean'), Avg_Age=('Age','mean')
    ).reset_index()
    summary['Churn_Rate_Pct'] = (summary['Churn_Rate_Pct']*100).round(1).astype(str) + '%'
    summary['Avg_Balance'] = summary['Avg_Balance'].round(0).astype(int)
    summary['Avg_Age'] = summary['Avg_Age'].round(1)
    st.dataframe(summary, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — AT-RISK CUSTOMERS
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">⚠️ High-Value Disengaged Customer Detector</div>', unsafe_allow_html=True)
    at_risk = df[(df['IsActiveMember']==0) & (df['Balance']>med_bal) & (df['Exited']==0)].copy()

    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1: st.metric("At-Risk Premium Customers", f"{len(at_risk):,}", help="Inactive + High Balance + Not yet churned")
    with col_r2: st.metric("Avg Balance at Risk", f"€{at_risk['Balance'].mean():,.0f}" if len(at_risk) else "€0")
    with col_r3: st.metric("Potential Revenue at Risk", f"€{at_risk['Balance'].sum()/1e6:.1f}M" if len(at_risk) else "€0M")

    st.markdown("---")
    col_ar1, col_ar2 = st.columns(2)
    with col_ar1:
        st.markdown('<div class="section-header">Balance Distribution: Active vs Inactive</div>', unsafe_allow_html=True)
        fig9, ax9 = plt.subplots(figsize=(6, 4))
        ax9.hist(df[df['IsActiveMember']==1]['Balance'], bins=40, alpha=0.6, color='#27ae60', label='Active', density=True)
        ax9.hist(df[df['IsActiveMember']==0]['Balance'], bins=40, alpha=0.6, color='#c0392b', label='Inactive', density=True)
        ax9.axvline(med_bal, color='navy', linestyle='--', linewidth=1.5, label=f'Median €{med_bal:,.0f}')
        ax9.set_xlabel('Account Balance (€)'); ax9.set_ylabel('Density')
        ax9.set_title('Balance Distribution by Activity Status', fontsize=11, fontweight='bold')
        ax9.legend(fontsize=9); ax9.spines[['top','right']].set_visible(False)
        plt.tight_layout(); st.pyplot(fig9); plt.close()

    with col_ar2:
        st.markdown('<div class="section-header">Churn Rate: Balance × Engagement</div>', unsafe_allow_html=True)
        df['Bal_Group']    = pd.cut(df['Balance'], bins=[0, med_bal, df['Balance'].max()+1], labels=['Low Balance','High Balance'])
        df['Active_Label'] = df['IsActiveMember'].map({1:'Active',0:'Inactive'})
        heat_data  = df.groupby(['Active_Label','Bal_Group'], observed=True)['Exited'].mean() * 100
        heat_pivot = heat_data.unstack()
        fig10, ax10 = plt.subplots(figsize=(5, 3))
        im = ax10.imshow(heat_pivot.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=50)
        ax10.set_xticks(range(len(heat_pivot.columns))); ax10.set_xticklabels(heat_pivot.columns, fontsize=10)
        ax10.set_yticks(range(len(heat_pivot.index)));   ax10.set_yticklabels(heat_pivot.index, fontsize=10)
        for i in range(heat_pivot.shape[0]):
            for j in range(heat_pivot.shape[1]):
                val = heat_pivot.values[i,j]
                ax10.text(j, i, f'{val:.1f}%', ha='center', va='center', fontsize=13, fontweight='bold',
                          color='white' if val > 30 else 'black')
        plt.colorbar(im, ax=ax10, label='Churn Rate (%)', shrink=0.8)
        ax10.set_title('Churn Heatmap: Engagement × Balance', fontsize=11, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig10); plt.close()

    st.markdown("---")
    st.markdown('<div class="section-header">Top At-Risk Premium Customers</div>', unsafe_allow_html=True)
    at_risk_display = at_risk[['CustomerId','Surname','Geography','Age','Balance','NumOfProducts','Tenure','RSI']].copy()
    at_risk_display = at_risk_display.sort_values('Balance', ascending=False).head(20)
    at_risk_display['Balance'] = at_risk_display['Balance'].apply(lambda x: f'€{x:,.0f}')
    at_risk_display['RSI']     = at_risk_display['RSI'].apply(lambda x: f'{x:.1f}/100')
    at_risk_display.columns    = ['Customer ID','Surname','Country','Age','Balance','Products','Tenure (yrs)','RSI Score']
    st.dataframe(at_risk_display, use_container_width=True, hide_index=True)

    at_risk_csv = at_risk.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Export At-Risk Customers CSV", at_risk_csv, "at_risk_customers.csv", "text/csv")
    st.markdown(f'<div class="insight-box">⚠️ <b>{len(at_risk):,} customers</b> are inactive with high balances but have not yet churned. Immediate re-engagement recommended.</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — RETENTION SCORING
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">🏆 Relationship Strength Index (RSI)</div>', unsafe_allow_html=True)
    st.markdown("**RSI Formula:** IsActiveMember×30 + (NumOfProducts/3)×30 + HasCrCard×10 + (Tenure/MaxTenure)×20 + HighBalance×10 = **100 pts max**")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        retained_rsi = df[df['Exited']==0]['RSI']; churned_rsi = df[df['Exited']==1]['RSI']
        fig11, ax11 = plt.subplots(figsize=(6, 4))
        ax11.hist(retained_rsi, bins=30, alpha=0.65, color='#27ae60', label=f'Retained (n={len(retained_rsi):,})', density=True)
        ax11.hist(churned_rsi,  bins=30, alpha=0.65, color='#c0392b', label=f'Churned (n={len(churned_rsi):,})',  density=True)
        ax11.axvline(retained_rsi.mean(), color='#1e8449', linestyle='--', linewidth=2, label=f'Retained Avg: {retained_rsi.mean():.1f}')
        ax11.axvline(churned_rsi.mean(),  color='#922b21', linestyle='--', linewidth=2, label=f'Churned Avg: {churned_rsi.mean():.1f}')
        ax11.set_xlabel('RSI Score'); ax11.set_ylabel('Density')
        ax11.set_title('RSI Distribution: Retained vs Churned', fontsize=11, fontweight='bold')
        ax11.legend(fontsize=8); ax11.spines[['top','right']].set_visible(False)
        plt.tight_layout(); st.pyplot(fig11); plt.close()

    with col_s2:
        df['RSI_Tier'] = pd.cut(df['RSI'], bins=[0,30,50,70,100],
                                 labels=['Very Low (0-30)','Low (31-50)','Medium (51-70)','High (71-100)'])
        rsi_churn = df.groupby('RSI_Tier', observed=True)['Exited'].agg(['mean','count']).reset_index()
        rsi_churn['mean'] *= 100
        fig12, ax12 = plt.subplots(figsize=(6, 4))
        bars12 = ax12.bar(rsi_churn['RSI_Tier'], rsi_churn['mean'],
                          color=['#c0392b','#e67e22','#f1c40f','#27ae60'], edgecolor='white', linewidth=1.5)
        for bar, row in zip(bars12, rsi_churn.itertuples()):
            label = str(round(row.mean,1)) + '%\n(n=' + f'{row.count:,}' + ')'
            ax12.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, label, ha='center', fontsize=9, fontweight='bold')
        ax12.set_ylabel('Churn Rate (%)'); ax12.set_title('Churn Rate by RSI Tier', fontsize=11, fontweight='bold')
        ax12.spines[['top','right']].set_visible(False); ax12.yaxis.grid(True, alpha=0.3, linestyle='--'); ax12.set_axisbelow(True)
        plt.xticks(fontsize=8); plt.tight_layout(); st.pyplot(fig12); plt.close()

    st.markdown("---")
    col_s3, col_s4 = st.columns(2)
    with col_s3:
        st.markdown('<div class="section-header">RSI vs Churn Rate by Segment</div>', unsafe_allow_html=True)
        seg_rsi        = df.groupby('Segment')['RSI'].mean().sort_values()
        seg_churn_rate = df.groupby('Segment')['Exited'].mean() * 100
        fig13, ax13 = plt.subplots(figsize=(6, 4))
        for seg in seg_rsi.index:
            ax13.scatter(seg_rsi[seg], seg_churn_rate[seg], c=seg_colors.get(seg,'#888'), s=300, zorder=5, edgecolors='white', linewidth=2)
            ax13.annotate(seg, (seg_rsi[seg], seg_churn_rate[seg]), textcoords='offset points', xytext=(8,4), fontsize=9)
        ax13.set_xlabel('Average RSI Score'); ax13.set_ylabel('Churn Rate (%)')
        ax13.set_title('RSI vs Churn Rate by Segment', fontsize=11, fontweight='bold')
        ax13.spines[['top','right']].set_visible(False); ax13.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout(); st.pyplot(fig13); plt.close()

    with col_s4:
        st.markdown('<div class="section-header">Retention Recommendations</div>', unsafe_allow_html=True)
        for seg, rec in {
            "🟢 Loyal Core":        "Reward & deepen. Offer premium cards, investment products, loyalty bonuses.",
            "🔵 Cross-sell Target": "Bundle offers. Push 2nd product. 27.7% → 7.6% churn drop possible.",
            "🟠 At-Risk Premium":   "Urgent re-engagement. Personalized outreach, RM calls, exclusive offers.",
            "🔴 Silent Churner":    "Win-back campaigns. Lower-barrier entry products, digital nudges."
        }.items():
            st.markdown(f"**{seg}**")
            st.markdown(f"<div style='background:#f8f9fa;border-left:3px solid #1a3a5c;padding:0.5rem 0.8rem;border-radius:4px;margin-bottom:0.8rem;font-size:0.9rem'>{rec}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">KPI Summary Table</div>', unsafe_allow_html=True)
    kpi_df = pd.DataFrame({
        'KPI':            ['Engagement Retention Ratio','Product Depth Index','High-Balance Disengagement Rate','Credit Card Stickiness Score','Avg RSI'],
        'Formula':        ['Inactive Churn ÷ Active Churn','1 − Churn Rate(2 products)','Churn % | Inactive & High Balance','Churn%(no card) − Churn%(has card)','Weighted composite score /100'],
        'Value':          [f'{ERR:.2f}x', f'{PDI:.1f}%', f'{HBDR:.1f}%', f'{CSS:.1f}pp', f'{avg_RSI:.1f}/100'],
        'Interpretation': ['Inactive members churn ~2x more','Retention rate of 2-product customers','Churn risk among premium inactive','Marginal retention from credit card','Average loyalty score across customers'],
        'Status':         ['⚠️ High Gap' if ERR>1.5 else '✅ Healthy', '✅ Strong' if PDI>85 else '⚠️ Moderate',
                           '🔴 Critical' if HBDR>25 else '⚠️ Elevated', '✅ Positive' if CSS>0 else '⚠️ Negative',
                           '✅ Good' if avg_RSI>50 else '⚠️ Low']
    })
    st.dataframe(kpi_df, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — ML CHURN PREDICTION
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">🤖 Machine Learning Churn Prediction</div>', unsafe_allow_html=True)
    st.markdown("Random Forest classifier trained on all 10,000 customers to predict churn probability and identify the most important churn drivers.")

    with st.spinner("Training Random Forest model..."):
        model, acc, report, importance_df, features = train_ml_model(df_full)

    # Model metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.markdown(f'<div class="kpi-card"><div class="kpi-value">{acc*100:.1f}%</div><div class="kpi-label">Model Accuracy</div><div class="kpi-formula">Correct predictions on test set (20%)</div></div>', unsafe_allow_html=True)
    with m2: st.markdown(f'<div class="kpi-card"><div class="kpi-value">{report["1"]["precision"]*100:.1f}%</div><div class="kpi-label">Churn Precision</div><div class="kpi-formula">Of predicted churners, % actually churned</div></div>', unsafe_allow_html=True)
    with m3: st.markdown(f'<div class="kpi-card"><div class="kpi-value">{report["1"]["recall"]*100:.1f}%</div><div class="kpi-label">Churn Recall</div><div class="kpi-formula">Of actual churners, % correctly identified</div></div>', unsafe_allow_html=True)
    with m4: st.markdown(f'<div class="kpi-card"><div class="kpi-value">{report["1"]["f1-score"]*100:.1f}%</div><div class="kpi-label">F1 Score</div><div class="kpi-formula">Harmonic mean of precision & recall</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    col_ml1, col_ml2 = st.columns(2)

    with col_ml1:
        st.markdown('<div class="section-header">Feature Importance — What Drives Churn?</div>', unsafe_allow_html=True)
        fig14, ax14 = plt.subplots(figsize=(6, 5))
        colors14 = ['#c0392b' if i < 3 else '#2980b9' if i < 6 else '#85c1e9' for i in range(len(importance_df))]
        bars14 = ax14.barh(importance_df['Feature'][::-1], importance_df['Importance'][::-1],
                           color=colors14[::-1], edgecolor='white')
        for bar, val in zip(bars14, importance_df['Importance'][::-1]):
            ax14.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
                      f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
        ax14.set_xlabel('Feature Importance Score')
        ax14.set_title('Random Forest Feature Importance', fontsize=12, fontweight='bold')
        ax14.spines[['top','right']].set_visible(False)
        ax14.set_xlim(0, importance_df['Importance'].max() * 1.2)
        plt.tight_layout(); st.pyplot(fig14); plt.close()
        st.markdown(f'<div class="ml-box">🔍 Top churn driver: <b>{importance_df.iloc[0]["Feature"]}</b> (importance: {importance_df.iloc[0]["Importance"]:.3f}). This confirms the analytical findings from the EDA.</div>', unsafe_allow_html=True)

    with col_ml2:
        st.markdown('<div class="section-header">Churn Probability by Segment</div>', unsafe_allow_html=True)
        from sklearn.preprocessing import LabelEncoder
        df_pred = df_full.copy()
        le = LabelEncoder()
        df_pred['Geography_enc'] = le.fit_transform(df_pred['Geography'])
        df_pred['Gender_enc']    = le.fit_transform(df_pred['Gender'])
        X_all  = df_pred[features]
        probs  = model.predict_proba(X_all)[:, 1]
        df_pred['Churn_Probability'] = probs

        seg_prob = df_pred.groupby('Segment')['Churn_Probability'].mean() * 100
        seg_prob = seg_prob.sort_values(ascending=False)

        fig15, ax15 = plt.subplots(figsize=(6, 4))
        bar_colors = [seg_colors.get(s,'#888') for s in seg_prob.index]
        bars15 = ax15.barh(seg_prob.index, seg_prob.values, color=bar_colors, edgecolor='white')
        for bar, val in zip(bars15, seg_prob.values):
            ax15.text(val+0.3, bar.get_y()+bar.get_height()/2, f'{val:.1f}%', va='center', fontweight='bold', fontsize=11)
        ax15.set_xlabel('Avg Predicted Churn Probability (%)')
        ax15.set_title('ML-Predicted Churn Probability by Segment', fontsize=11, fontweight='bold')
        ax15.spines[['top','right']].set_visible(False); ax15.set_xlim(0, seg_prob.max()*1.25)
        plt.tight_layout(); st.pyplot(fig15); plt.close()

    st.markdown("---")
    st.markdown('<div class="section-header">🔮 Individual Customer Churn Predictor</div>', unsafe_allow_html=True)
    st.markdown("Enter a customer's details to get their predicted churn probability.")

    with st.form("predict_form"):
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            p_age        = st.slider("Age", 18, 92, 40)
            p_tenure     = st.slider("Tenure (years)", 0, 10, 5)
            p_products   = st.slider("Number of Products", 1, 4, 1)
        with pc2:
            p_balance    = st.number_input("Balance (€)", 0, 250000, 50000, step=5000)
            p_salary     = st.number_input("Estimated Salary (€)", 10000, 200000, 80000, step=5000)
            p_credit     = st.slider("Credit Score", 300, 850, 650)
        with pc3:
            p_geo        = st.selectbox("Geography", ['France','Germany','Spain'])
            p_gender     = st.selectbox("Gender", ['Male','Female'])
            p_active     = st.selectbox("Is Active Member?", ['Yes','No'])
            p_card       = st.selectbox("Has Credit Card?", ['Yes','No'])

        submitted = st.form_submit_button("🔮 Predict Churn Risk", use_container_width=True)

    if submitted:
        geo_map    = {'France':0,'Germany':1,'Spain':2}
        gender_map = {'Female':0,'Male':1}
        p_rsi = (
            (1 if p_active=='Yes' else 0) * 30 +
            min(p_products, 3) / 3 * 30 +
            (1 if p_card=='Yes' else 0) * 10 +
            (p_tenure / 10) * 20 +
            (1 if p_balance > med_bal else 0) * 10
        )
        input_data = pd.DataFrame([[
            p_credit, geo_map[p_geo], gender_map[p_gender], p_age, p_tenure,
            p_balance, p_products, 1 if p_card=='Yes' else 0,
            1 if p_active=='Yes' else 0, p_salary, p_rsi
        ]], columns=features)

        churn_prob = model.predict_proba(input_data)[0][1] * 100
        churn_pred = model.predict(input_data)[0]

        res1, res2, res3 = st.columns(3)
        with res1:
            color = '#c0392b' if churn_prob > 50 else ('#e67e22' if churn_prob > 30 else '#27ae60')
            st.markdown(f"<div style='background:{color}15;border-left:5px solid {color};border-radius:8px;padding:1rem;text-align:center'><div style='font-size:2.5rem;font-weight:700;color:{color}'>{churn_prob:.1f}%</div><div style='color:#555;font-size:0.9rem'>Churn Probability</div></div>", unsafe_allow_html=True)
        with res2:
            label = "🔴 HIGH RISK" if churn_prob > 50 else ("🟠 MEDIUM RISK" if churn_prob > 30 else "🟢 LOW RISK")
            st.markdown(f"<div style='background:#f8f9fa;border-radius:8px;padding:1rem;text-align:center'><div style='font-size:1.3rem;font-weight:700'>{label}</div><div style='color:#555;font-size:0.9rem;margin-top:0.3rem'>Risk Category</div></div>", unsafe_allow_html=True)
        with res3:
            st.markdown(f"<div style='background:#f0f4fa;border-radius:8px;padding:1rem;text-align:center'><div style='font-size:1.8rem;font-weight:700;color:#1a3a5c'>{p_rsi:.1f}/100</div><div style='color:#555;font-size:0.9rem'>Relationship Strength Index</div></div>", unsafe_allow_html=True)

        if churn_prob > 50:
            st.markdown('<div class="insight-box">⚠️ <b>High churn risk detected.</b> This customer should be flagged for immediate re-engagement. Consider personalized offers or relationship manager outreach.</div>', unsafe_allow_html=True)
        elif churn_prob > 30:
            st.markdown('<div class="insight-box">🟠 <b>Moderate churn risk.</b> Monitor this customer closely. A product bundle offer or loyalty incentive could reduce churn risk.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="insight-box">✅ <b>Low churn risk.</b> This customer shows strong retention signals. Focus on deepening the relationship further.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Model Information</div>', unsafe_allow_html=True)
    with st.expander("ℹ️ About the ML Model"):
        st.markdown(f"""
**Algorithm:** Random Forest Classifier (100 trees)

**Training Setup:**
- Train/Test Split: 80% / 20% (stratified)
- Features used: {len(features)} variables
- Target: Exited (0 = Retained, 1 = Churned)

**Feature Engineering:**
- Geography and Gender label-encoded
- RSI (Relationship Strength Index) added as engineered feature
- No scaling required for Random Forest

**Performance:**
- Accuracy: {acc*100:.1f}%
- Precision (churn): {report['1']['precision']*100:.1f}%
- Recall (churn): {report['1']['recall']*100:.1f}%
- F1 Score: {report['1']['f1-score']*100:.1f}%

**Why Random Forest?** Handles mixed data types, robust to outliers, provides feature importance scores, and does not require feature scaling — ideal for this customer dataset.
        """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#888;font-size:0.85rem'>"
    "Customer Engagement & Retention Analytics | European Central Bank | "
    f"Dataset: {len(df_full):,} customers | Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)
