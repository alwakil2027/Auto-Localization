# auto_industry_financial_dashboard.py
# ---------------------------------------------------------
# 🚗 Dashboard تفاعلي لتوطين صناعة السيارات في مصر
# KPIs Cards • سيناريوهات • حساسية سعر الصرف • تصدير Excel/CSV
# ---------------------------------------------------------

import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import numpy_financial as npf

# ---------- إعداد الصفحة ----------
st.set_page_config(
    page_title="Financial Dashboard - Auto Egypt",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗  توطين صناعة السيارات في مصر")
st.caption("EMS Experts | سيناريوهات (محافظ/أساسي/طموح) + حساسية سعر الصرف + KPIs + تصدير التقارير")

# ---------- عناصر التحكم الجانبية ----------
with st.sidebar:
    st.header("⚙️ إعدادات عامة")
    years = list(range(2025, 2036))  # 11 سنة
    base_revenue = st.number_input("الإيراد الابتدائي (USD)", min_value=1_000_000, value=1_000_000_000, step=50_000_000)
    fx_rate = st.slider("سعر الصرف (EGP / USD)", 30, 150, 50, step=5)
    discount_rate = st.slider("معدل الخصم (Discount Rate)", 0.05, 0.25, 0.10, step=0.01)
    investment_usd = st.number_input("الاستثمار المبدئي (USD)", min_value=100_000_000, value=1_500_000_000, step=100_000_000)

    st.divider()
    st.subheader("معلمات السيناريوهات")
    colA, colB, colC = st.columns(3)
    with colA:
        g1 = st.slider("نمو محافظ", 0.00, 0.20, 0.05, step=0.01)
        m1 = st.slider("هامش محافظ", 0.00, 0.30, 0.08, step=0.01)
    with colB:
        g2 = st.slider("نمو أساسي", 0.00, 0.25, 0.10, step=0.01)
        m2 = st.slider("هامش أساسي", 0.00, 0.35, 0.12, step=0.01)
    with colC:
        g3 = st.slider("نمو طموح", 0.00, 0.35, 0.15, step=0.01)
        m3 = st.slider("هامش طموح", 0.00, 0.50, 0.18, step=0.01)

    st.divider()
    st.subheader("تحليل الحساسية")
    sens_growth_min, sens_growth_max = st.slider("مدى معدل النمو للحساسية", 0.00, 0.30, (0.03, 0.15), step=0.01)
    sens_fx_min, sens_fx_max = st.slider("مدى سعر الصرف للحساسية", 20, 180, (30, 120), step=5)

# ---------- تعريف السيناريوهات ----------
scenarios = {
    "محافظ": {"growth": g1, "margin": m1},
    "أساسي": {"growth": g2, "margin": m2},
    "طموح": {"growth": g3, "margin": m3},
}

# ---------- توليد البيانات ----------
records = []
for scen, params in scenarios.items():
    revenue = base_revenue
    for year in years:
        revenue *= (1 + params["growth"])
        profit_usd = revenue * params["margin"]
        profit_egp = profit_usd * fx_rate
        records.append({
            "السيناريو": scen,
            "السنة": year,
            "الإيرادات (مليار $)": revenue / 1e9,
            "الأرباح (مليار $)": profit_usd / 1e9,
            "الأرباح (مليار جنيه)": profit_egp / 1e9,
        })

df = pd.DataFrame(records)

# ---------- دوال مساعدة KPIs ----------
def compute_kpis(df_all: pd.DataFrame, scen: str, investment: float, dr: float):
    """ يحسب NPV/IRR/ROI/Payback بناءً على أرباح بالدولار """
    subset = df_all[df_all["السيناريو"] == scen].copy()
    cashflows = [-investment] + list((subset["الأرباح (مليار $)"] * 1e9).values)
    npv = float(npf.npv(dr, cashflows))
    irr = float(npf.irr(cashflows)) if any(v > 0 for v in cashflows[1:]) else np.nan
    roi = ((sum(cashflows[1:]) - abs(cashflows[0])) / abs(cashflows[0])) if investment != 0 else np.nan

    cum = np.cumsum(cashflows)
    payback_year = next((i for i, v in enumerate(cum) if v >= 0), None)
    return npv, irr, roi, payback_year

kpi_rows = []
for scen in scenarios:
    npv, irr, roi, payback = compute_kpis(df, scen, investment_usd, discount_rate)
    kpi_rows.append({
        "السيناريو": scen,
        "NPV (مليون $)": npv / 1e6,
        "IRR (%)": (irr * 100) if not np.isnan(irr) else np.nan,
        "ROI (%)": roi * 100 if roi is not None else np.nan,
        "Payback (سنوات)": payback
    })
kpi_df = pd.DataFrame(kpi_rows).round(2)

# ---------- كروت KPIs (Executive Cards) ----------
col1, col2, col3 = st.columns(3)
def fmt_money_b(b):
    try:
        return f"{b:,.2f}"
    except Exception:
        return "-"

with col1:
    total_rev_egp = df.groupby("السيناريو")["الأرباح (مليار جنيه)"].sum().reindex(["محافظ","أساسي","طموح"])
    st.metric("إجمالي الأرباح بالجنيه — محافظ", fmt_money_b(total_rev_egp.get("محافظ", 0)))
with col2:
    st.metric("إجمالي الأرباح بالجنيه — أساسي", fmt_money_b(total_rev_egp.get("أساسي", 0)))
with col3:
    st.metric("إجمالي الأرباح بالجنيه — طموح", fmt_money_b(total_rev_egp.get("طموح", 0)))

col4, col5, col6 = st.columns(3)
with col4:
    st.metric("NPV (مليون $) — أساسي", fmt_money_b(kpi_df.set_index("السيناريو").loc["أساسي","NPV (مليون $)"]))
with col5:
    st.metric("IRR (%) — أساسي", fmt_money_b(kpi_df.set_index("السيناريو").loc["أساسي","IRR (%)"]))
with col6:
    st.metric("Payback (سنوات) — أساسي", str(int(kpi_df.set_index("السيناريو").loc["أساسي","Payback (سنوات)"])) if not pd.isna(kpi_df.set_index("السيناريو").loc["أساسي","Payback (سنوات)"]) else "—")

st.divider()

# ---------- رسوم تفاعلية ----------
st.subheader("📈 تطور الإيرادات (مليار دولار)")
fig1 = px.line(df, x="السنة", y="الإيرادات (مليار $)", color="السيناريو", markers=True)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("💰 الأرباح بالجنيه المصري (مليار)")
fig2 = px.line(df, x="السنة", y="الأرباح (مليار جنيه)", color="السيناريو", markers=True)
st.plotly_chart(fig2, use_container_width=True)

# ---------- جدول ملخص ----------
st.subheader("📊 جدول البيانات")
st.dataframe(df.round(3), use_container_width=True)

# ---------- تحليل الحساسية (Heatmap) ----------
st.subheader("🔍 Heatmap — تأثير سعر الصرف + معدل النمو على الأرباح (سنة النهاية)")
growth_grid = np.linspace(sens_growth_min, sens_growth_max, 7)
fx_grid = np.arange(sens_fx_min, sens_fx_max + 1, 10)

sens_rows = []
# نستخدم هامش السيناريو "أساسي" كمقياس قياسي
base_margin = scenarios["أساسي"]["margin"]
for g in growth_grid:
    for fx in fx_grid:
        terminal_revenue = base_revenue * ((1 + g) ** (len(years)))  # سنة النهاية
        terminal_profit = terminal_revenue * base_margin
        terminal_profit_egp = terminal_profit * fx
        sens_rows.append({
            "معدل النمو": f"{g*100:.0f}%",
            "سعر الصرف": fx,
            "الأرباح (مليار جنيه)": terminal_profit_egp / 1e9
        })
sens_df = pd.DataFrame(sens_rows)

fig3 = px.density_heatmap(
    sens_df, x="سعر الصرف", y="معدل النمو", z="الأرباح (مليار جنيه)",
    color_continuous_scale="Viridis"
)
st.plotly_chart(fig3, use_container_width=True)

# ---------- جداول KPIs ----------
st.subheader("🏁 KPIs الملخصة")
st.dataframe(kpi_df, use_container_width=True)

st.divider()

# ---------- ملاحظة ----------
st.caption(
    "ملاحظة: القيم تقديرية لغرض النمذجة—عدّل المعلمات من الشريط الجانبي لتخصيص النتائج، "
    "وأصدر التقارير مباشرة من الأزرار أعلاه."
)
