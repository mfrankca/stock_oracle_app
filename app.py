
# Streamlit Ai Finance App – Watchlist, Reports & Options Analysis
# Full single-file app (fixed)

import math
from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# =============================
# App config
# =============================
st.set_page_config(page_title="AI MonFin – Watchlist · Reports · Options", page_icon="📈", layout="wide")
RISK_FREE_DEFAULT = 0.045

# =============================
# Helpers (cached where sensible)
# =============================
@st.cache_data(ttl=120)
def fetch_price_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker.upper().strip()).history(period=period, interval=interval, auto_adjust=False)
        if df is None:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60)
def fetch_live_price(ticker: str) -> float:
    try:
        t = yf.Ticker(ticker.upper().strip())
        fi = getattr(t, "fast_info", None)
        if fi and getattr(fi, "last_price", None) is not None:
            return float(fi.last_price)
        intr = t.history(period="1d", interval="1m", auto_adjust=False)
        if not intr.empty:
            return float(intr["Close"].iloc[-1])
    except Exception:
        pass
    return float("nan")

@st.cache_data(ttl=300)
def get_option_expiries(ticker: str) -> list[str]:
    try:
        exps = yf.Ticker(ticker.upper().strip()).options or []
        return list(exps)
    except Exception:
        return []

@st.cache_data(ttl=120)
def get_option_chain(ticker: str, expiry: str) -> dict:
    """Return a pickle‑serializable option chain: dict with DataFrames for calls/puts."""
    try:
        oc = yf.Ticker(ticker.upper().strip()).option_chain(expiry)
        calls = oc.calls.reset_index(drop=True) if hasattr(oc, "calls") else pd.DataFrame()
        puts  = oc.puts.reset_index(drop=True)  if hasattr(oc, "puts")  else pd.DataFrame()
        # Ensure numeric types
        for df in (calls, puts):
            for col in ["strike","lastPrice","bid","ask","volume","openInterest","impliedVolatility"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
        return {"calls": calls, "puts": puts}
    except Exception:
        return {"calls": pd.DataFrame(), "puts": pd.DataFrame()}

# Exports
def make_excel(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name[:31], index=False)
    return buf.getvalue()

def make_pdf_report(title: str, lines: list[str]) -> bytes | None:
    try:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import inch
        buf = BytesIO()
        c = canvas.Canvas(buf, pagesize=LETTER)
        width, height = LETTER
        y = height - 1*inch
        c.setFont("Helvetica-Bold", 14)
        c.drawString(1*inch, y, title)
        y -= 0.4*inch
        c.setFont("Helvetica", 11)
        for line in lines:
            if y < 1*inch:
                c.showPage(); y = height - 1*inch; c.setFont("Helvetica", 11)
            c.drawString(1*inch, y, line)
            y -= 0.25*inch
        c.showPage(); c.save()
        pdf = buf.getvalue(); buf.close()
        return pdf
    except Exception:
        return None

# Simple 12‑mo prediction (demo)
def ai_predict_next_year(ticker: str):
    hist = fetch_price_history(ticker, period="5y", interval="1mo")
    if hist.empty:
        return None, None, None
    df = hist.dropna(subset=["Close"]).copy().reset_index()
    df["t"] = np.arange(len(df))
    # linear trend on monthly closes
    coeffs = np.polyfit(df["t"], df["Close"], 1)
    pred = float(np.polyval(coeffs, len(df) + 12))
    current = float(df["Close"].iloc[-1])
    series = df.set_index("Date")["Close"]
    return pred, current, series

# Weekly watchlist suggestion (momentum-ish)
def suggest_weekly_watchlist() -> pd.DataFrame:
    base = ["SPY","QQQ","DIA","IWM","XLF","XLV","XLE","XLK","SMH","ARKK",
            "AAPL","MSFT","NVDA","AMZN","META","TSLA","GOOGL","JPM","XOM","UNH"]
    rows = []
    for t in base:
        h = fetch_price_history(t, period="5d", interval="1d")
        if h.empty or len(h) < 2:
            continue
        change = (h["Close"].iloc[-1] / h["Close"].iloc[0]) - 1
        rows.append({"Ticker": t, "5d %": change})
    df = pd.DataFrame(rows)
    return df.sort_values("5d %", ascending=False).reset_index(drop=True) if not df.empty else df

# Helper function for RSI calculation
def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# =============================
# Sidebar / Nav
# =============================
with st.sidebar:
    st.title("📈 AI Finance")
    page = st.radio(
        "Navigate",
        [
            "Watchlist",
            "Portfolio Analysis",
            "Options Analysis",
            "Options Strategy Builder",
            "Weekly Watchlist",
            "AI Next Year Prediction",
            "Intrinsic Value",
            "Daily Scanner",
            "Sector Tracker",
            "Pattern Scanner",
            "Options Flow",
            "Settings",
        ],
        index=0,
        key="nav_page",
    )
    st.markdown("---")
    rf = st.number_input(
        "Risk‑free (annual %)", 
        0.0, 15.0, RISK_FREE_DEFAULT*100, 0.1, 
        key="rf_input",
        help="Risk-free rate (typically Treasury yield) used in calculations. Affects discount rates and option pricing."
    )
    st.session_state["risk_free"] = rf/100.0

# =============================
# Watchlist
# =============================
if page == "Watchlist":
    st.header("Watchlist")
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = ["AAPL","MSFT","NVDA"]

    c1, c2 = st.columns([3,2])
    with c1:
        add_t = st.text_input("Add ticker (e.g., AAPL)", key="watch_add")
        cols = st.columns(3)
        if cols[0].button("➕ Add", key="watch_add_btn") and add_t:
            t = add_t.upper().strip()
            if t and t not in st.session_state.watchlist:
                st.session_state.watchlist.append(t)
        rm_t = cols[1].selectbox("Remove", options=["(select)"] + st.session_state.watchlist, key="watch_rm_sel")
        if cols[2].button("🗑️ Remove", key="watch_rm_btn") and rm_t != "(select)":
            st.session_state.watchlist = [x for x in st.session_state.watchlist if x != rm_t]
    with c2:
        st.download_button("⬇️ Download watchlist", data=pd.Series(st.session_state.watchlist, name="ticker").to_csv(index=False), file_name="watchlist.csv", key="watch_dl")
        up = st.file_uploader("Upload tickers CSV", type=["csv"], key="watch_up")
        if up is not None:
            try:
                ticks = pd.read_csv(up).iloc[:,0].dropna().astype(str).str.upper().str.strip().tolist()
                st.session_state.watchlist = sorted(list(set(ticks)))
                st.success(f"Loaded {len(ticks)} tickers.")
            except Exception as e:
                st.error(f"CSV parse failed: {e}")

    st.markdown("---")
    if not st.session_state.watchlist:
        st.info("Add tickers to view quotes and charts.")
    else:
        period = st.selectbox("History period", ["1mo","3mo","6mo","1y","2y","5y","10y","max"], index=3, key="watch_period")
        interval = st.selectbox("Interval", ["1d","1wk","1mo"], index=0, key="watch_interval")
        # quotes table
        rows = []
        ticker_descriptions = {
            # Major ETFs
            "SPY": "SPDR S&P 500 ETF - Tracks the S&P 500 index (500 largest US companies)",
            "QQQ": "Invesco QQQ Trust - Tracks NASDAQ-100 (top 100 non-financial NASDAQ stocks)",
            "DIA": "SPDR Dow Jones Industrial Average ETF - Tracks the Dow Jones Industrial Average",
            "IWM": "iShares Russell 2000 ETF - Tracks small-cap US stocks",
            "XLK": "Technology Select Sector SPDR Fund - Technology sector ETF",
            "XLF": "Financial Select Sector SPDR Fund - Financial sector ETF",
            "XLV": "Health Care Select Sector SPDR Fund - Healthcare sector ETF",
            "XLE": "Energy Select Sector SPDR Fund - Energy sector ETF",
            "XLI": "Industrial Select Sector SPDR Fund - Industrial sector ETF",
            "XLP": "Consumer Staples Select Sector SPDR Fund - Consumer staples sector ETF",
            "XLU": "Utilities Select Sector SPDR Fund - Utilities sector ETF",
            "XLB": "Materials Select Sector SPDR Fund - Materials sector ETF",
            "XLRE": "Real Estate Select Sector SPDR Fund - Real estate sector ETF",
            "XLC": "Communication Services Select Sector SPDR Fund - Communication services sector ETF",
            "SMH": "VanEck Vectors Semiconductor ETF - Semiconductor industry ETF",
            "ARKK": "ARK Innovation ETF - Disruptive innovation companies",
            
            # Major Tech Companies
            "AAPL": "Apple Inc. - Consumer electronics, software, and services",
            "MSFT": "Microsoft Corporation - Software, cloud computing, and technology",
            "GOOGL": "Alphabet Inc. (Google) - Internet services, advertising, and technology",
            "AMZN": "Amazon.com Inc. - E-commerce, cloud computing, and digital services",
            "NVDA": "NVIDIA Corporation - Graphics processing units and AI computing",
            "META": "Meta Platforms Inc. - Social media and digital advertising",
            "TSLA": "Tesla Inc. - Electric vehicles, energy storage, and solar panels",
            "NFLX": "Netflix Inc. - Streaming entertainment and content production",
            "ADBE": "Adobe Inc. - Creative software and digital media solutions",
            "CRM": "Salesforce Inc. - Customer relationship management software",
            "PYPL": "PayPal Holdings Inc. - Digital payments and financial services",
            "INTC": "Intel Corporation - Semiconductor manufacturing and computing",
            "AMD": "Advanced Micro Devices Inc. - Semiconductor and computing technology",
            "ORCL": "Oracle Corporation - Database software and cloud services",
            "CSCO": "Cisco Systems Inc. - Networking hardware and software",
            
            # Growth Companies
            "ZM": "Zoom Video Communications Inc. - Video conferencing and communication",
            "SHOP": "Shopify Inc. - E-commerce platform and business solutions",
            "SQ": "Block Inc. (Square) - Financial services and mobile payments",
            "ROKU": "Roku Inc. - Streaming platform and smart TV operating system",
            "CRWD": "CrowdStrike Holdings Inc. - Cybersecurity and endpoint protection",
            "OKTA": "Okta Inc. - Identity and access management software",
            "DOCU": "DocuSign Inc. - Electronic signature and document management",
            
            # Financial & Industrial
            "BRK-B": "Berkshire Hathaway Inc. - Conglomerate with diverse business holdings",
            "UNH": "UnitedHealth Group Inc. - Healthcare insurance and services",
            "JNJ": "Johnson & Johnson - Healthcare products and pharmaceuticals",
            "JPM": "JPMorgan Chase & Co. - Banking and financial services",
            "V": "Visa Inc. - Payment processing and financial services",
            "PG": "Procter & Gamble Co. - Consumer goods and household products",
            "HD": "Home Depot Inc. - Home improvement retail",
            "MA": "Mastercard Inc. - Payment processing and financial services",
            "DIS": "Walt Disney Co. - Entertainment, media, and theme parks",
            "XOM": "Exxon Mobil Corporation - Oil and gas exploration and production"
        }
        
        for t in st.session_state.watchlist:
            h3 = fetch_price_history(t, period="3mo", interval="1d")
            if h3.empty:
                rows.append({
                    "Ticker": t, 
                    "Description": ticker_descriptions.get(t, 'N/A'),
                    "Last": np.nan, 
                    "1D %": np.nan, 
                    "YTD %": np.nan
                })
                continue
            last = float(h3["Close"].iloc[-1])
            day = h3["Close"].pct_change().iloc[-1] if len(h3) > 1 else np.nan
            ytd = np.nan
            ytd_hist = fetch_price_history(t, period="ytd", interval="1d")
            if not ytd_hist.empty:
                ytd = (ytd_hist["Close"].pct_change().fillna(0) + 1).prod() - 1
            rows.append({
                "Ticker": t, 
                "Description": ticker_descriptions.get(t, 'N/A'),
                "Last": last, 
                "1D %": day, 
                "YTD %": ytd
            })
        
        df_watchlist = pd.DataFrame(rows)
        st.dataframe(
            df_watchlist,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"),
                "Description": st.column_config.TextColumn("Description", width="large"),
                "Last": st.column_config.NumberColumn("Last Price", format="%.2f"),
                "1D %": st.column_config.NumberColumn("1D %", format="%.2f%%"),
                "YTD %": st.column_config.NumberColumn("YTD %", format="%.2f%%"),
            }
        )

        sel = st.selectbox("Chart ticker", st.session_state.watchlist, key="watch_chart_sel")
        hist = fetch_price_history(sel, period=period, interval=interval)
        if hist.empty:
            st.warning("No data for selected ticker.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name=sel))
            fig.update_layout(height=460, title=f"{sel} – {period}/{interval}")
            st.plotly_chart(fig, use_container_width=True)

# =============================
# Portfolio Analysis
# =============================
elif page == "Portfolio Analysis":
    st.header("📊 Portfolio Analysis & Performance")
    st.caption("Comprehensive performance metrics, risk analysis, and portfolio insights")
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = ["AAPL","MSFT","NVDA"]

    def compute_perf_metrics(df: pd.DataFrame) -> dict:
        if df.empty or "Close" not in df:
            return {}
        px = df["Close"].dropna()
        if px.size < 2:
            return {}
        rets = px.pct_change().dropna()
        if rets.empty:
            return {}
        ann = 252
        ann_ret = (1 + rets.mean()) ** ann - 1
        ann_vol = rets.std() * np.sqrt(ann)
        sharpe = (ann_ret - st.session_state.get("risk_free", RISK_FREE_DEFAULT)) / ann_vol if ann_vol > 0 else np.nan
        mdd = ((px / px.cummax()) - 1).min()
        ytd = np.nan
        ytd_px = px[px.index.year == datetime.today().year]
        if not ytd_px.empty:
            ytd = (ytd_px.pct_change().fillna(0) + 1).prod() - 1
        return {"last_price": float(px.iloc[-1]), "ytd_return": float(ytd), "ann_return": float(ann_ret), "ann_vol": float(ann_vol), "sharpe": float(sharpe), "max_drawdown": float(mdd)}

    # Analysis options
    col1, col2 = st.columns([2, 1])
    with col1:
        tickers = st.multiselect(
            "Select tickers to analyze", 
            options=st.session_state.watchlist, 
            default=st.session_state.watchlist[:3], 
            key="rpt_ticks",
            help="Choose stocks/ETFs to include in your portfolio analysis"
        )
    with col2:
        period = st.selectbox(
            "Analysis Period", 
            ["6mo","1y","2y","3y","5y"], 
            index=1, 
            key="rpt_period",
            help="Time frame for performance analysis"
        )

    if st.button("📊 Generate Portfolio Analysis", key="rpt_btn"):
        if not tickers:
            st.warning("Select at least one ticker.")
        else:
            recs = []
            for t in tickers:
                h = fetch_price_history(t, period=period, interval="1d")
                m = compute_perf_metrics(h)
                if m:
                    m["Ticker"] = t
                    recs.append(m)
            if not recs:
                st.error("No valid data for the selected tickers.")
            else:
                rpt = pd.DataFrame(recs).set_index("Ticker")
                
                # Add ticker descriptions
                ticker_descriptions = {
                    # Major ETFs
                    "SPY": "SPDR S&P 500 ETF - Tracks the S&P 500 index (500 largest US companies)",
                    "QQQ": "Invesco QQQ Trust - Tracks NASDAQ-100 (top 100 non-financial NASDAQ stocks)",
                    "DIA": "SPDR Dow Jones Industrial Average ETF - Tracks the Dow Jones Industrial Average",
                    "IWM": "iShares Russell 2000 ETF - Tracks small-cap US stocks",
                    "XLK": "Technology Select Sector SPDR Fund - Technology sector ETF",
                    "XLF": "Financial Select Sector SPDR Fund - Financial sector ETF",
                    "XLV": "Health Care Select Sector SPDR Fund - Healthcare sector ETF",
                    "XLE": "Energy Select Sector SPDR Fund - Energy sector ETF",
                    "XLI": "Industrial Select Sector SPDR Fund - Industrial sector ETF",
                    "XLP": "Consumer Staples Select Sector SPDR Fund - Consumer staples sector ETF",
                    "XLU": "Utilities Select Sector SPDR Fund - Utilities sector ETF",
                    "XLB": "Materials Select Sector SPDR Fund - Materials sector ETF",
                    "XLRE": "Real Estate Select Sector SPDR Fund - Real estate sector ETF",
                    "XLC": "Communication Services Select Sector SPDR Fund - Communication services sector ETF",
                    "SMH": "VanEck Vectors Semiconductor ETF - Semiconductor industry ETF",
                    "ARKK": "ARK Innovation ETF - Disruptive innovation companies",
                    
                    # Major Tech Companies
                    "AAPL": "Apple Inc. - Consumer electronics, software, and services",
                    "MSFT": "Microsoft Corporation - Software, cloud computing, and technology",
                    "GOOGL": "Alphabet Inc. (Google) - Internet services, advertising, and technology",
                    "AMZN": "Amazon.com Inc. - E-commerce, cloud computing, and digital services",
                    "NVDA": "NVIDIA Corporation - Graphics processing units and AI computing",
                    "META": "Meta Platforms Inc. - Social media and digital advertising",
                    "TSLA": "Tesla Inc. - Electric vehicles, energy storage, and solar panels",
                    "NFLX": "Netflix Inc. - Streaming entertainment and content production",
                    "ADBE": "Adobe Inc. - Creative software and digital media solutions",
                    "CRM": "Salesforce Inc. - Customer relationship management software",
                    "PYPL": "PayPal Holdings Inc. - Digital payments and financial services",
                    "INTC": "Intel Corporation - Semiconductor manufacturing and computing",
                    "AMD": "Advanced Micro Devices Inc. - Semiconductor and computing technology",
                    "ORCL": "Oracle Corporation - Database software and cloud services",
                    "CSCO": "Cisco Systems Inc. - Networking hardware and software",
                    
                    # Growth Companies
                    "ZM": "Zoom Video Communications Inc. - Video conferencing and communication",
                    "SHOP": "Shopify Inc. - E-commerce platform and business solutions",
                    "SQ": "Block Inc. (Square) - Financial services and mobile payments",
                    "ROKU": "Roku Inc. - Streaming platform and smart TV operating system",
                    "CRWD": "CrowdStrike Holdings Inc. - Cybersecurity and endpoint protection",
                    "OKTA": "Okta Inc. - Identity and access management software",
                    "DOCU": "DocuSign Inc. - Electronic signature and document management",
                    
                    # Financial & Industrial
                    "BRK-B": "Berkshire Hathaway Inc. - Conglomerate with diverse business holdings",
                    "UNH": "UnitedHealth Group Inc. - Healthcare insurance and services",
                    "JNJ": "Johnson & Johnson - Healthcare products and pharmaceuticals",
                    "JPM": "JPMorgan Chase & Co. - Banking and financial services",
                    "V": "Visa Inc. - Payment processing and financial services",
                    "PG": "Procter & Gamble Co. - Consumer goods and household products",
                    "HD": "Home Depot Inc. - Home improvement retail",
                    "MA": "Mastercard Inc. - Payment processing and financial services",
                    "DIS": "Walt Disney Co. - Entertainment, media, and theme parks",
                    "XOM": "Exxon Mobil Corporation - Oil and gas exploration and production"
                }
                
                # Add description column
                rpt['Description'] = rpt.index.map(lambda x: ticker_descriptions.get(x, 'N/A'))
                
                st.subheader("📈 Portfolio Performance Summary")
                st.dataframe(rpt.style.format({"last_price":"{:.2f}","ytd_return":"{:.2%}","ann_return":"{:.2%}","ann_vol":"{:.2%}","sharpe":"{:.2f}","max_drawdown":"{:.2%}"}), use_container_width=True)
                cumfig = go.Figure()
                for t in tickers:
                    h = fetch_price_history(t, period=period, interval="1d")
                    if h.empty: continue
                    cum = (h["Close"].pct_change().fillna(0)+1).cumprod()-1
                    cumfig.add_trace(go.Scatter(x=cum.index, y=cum.values, name=t))
                cumfig.update_layout(height=420, title="📊 Cumulative Return Comparison (normalized)")
                st.plotly_chart(cumfig, use_container_width=True)
                
                # Portfolio insights
                st.subheader("💡 Portfolio Insights")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    best_performer = rpt.loc[rpt['ann_return'].idxmax()]
                    st.metric("🏆 Best Performer", f"{rpt['ann_return'].idxmax()}")
                    st.caption(f"Annual Return: {best_performer['ann_return']:.1%}")
                
                with col2:
                    worst_performer = rpt.loc[rpt['ann_return'].idxmin()]
                    st.metric("📉 Worst Performer", f"{rpt['ann_return'].idxmin()}")
                    st.caption(f"Annual Return: {worst_performer['ann_return']:.1%}")
                
                with col3:
                    avg_return = rpt['ann_return'].mean()
                    st.metric("📊 Portfolio Average", f"{avg_return:.1%}")
                    st.caption("Average Annual Return")
                
                # Risk analysis
                st.subheader("⚠️ Risk Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    highest_vol = rpt.loc[rpt['ann_vol'].idxmax()]
                    st.markdown(f"**Highest Volatility**: {rpt['ann_vol'].idxmax()} ({highest_vol['ann_vol']:.1%})")
                    
                    lowest_vol = rpt.loc[rpt['ann_vol'].idxmin()]
                    st.markdown(f"**Lowest Volatility**: {rpt['ann_vol'].idxmin()} ({lowest_vol['ann_vol']:.1%})")
                
                with col2:
                    best_sharpe = rpt.loc[rpt['sharpe'].idxmax()]
                    st.markdown(f"**Best Risk-Adjusted**: {rpt['sharpe'].idxmax()} (Sharpe: {best_sharpe['sharpe']:.2f})")
                    
                    worst_drawdown = rpt.loc[rpt['max_drawdown'].idxmin()]
                    st.markdown(f"**Worst Drawdown**: {rpt['max_drawdown'].idxmin()} ({worst_drawdown['max_drawdown']:.1%})")
                
                # Export options
                st.subheader("📤 Export Options")
                col1, col2 = st.columns(2)
                
                with col1:
                    xls = make_excel({"Portfolio_Analysis": rpt.reset_index()})
                    st.download_button(
                        "⬇️ Download Excel Report", 
                        data=xls, 
                        file_name="portfolio_analysis.xlsx", 
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
                        key="rpt_xls"
                    )
                
                with col2:
                    # Create PDF report
                    pdf_lines = [
                        f"Portfolio Analysis Report - {period}",
                        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                        "",
                        "Performance Summary:",
                    ]
                    
                    for ticker in rpt.index:
                        desc = ticker_descriptions.get(ticker, 'N/A')
                        pdf_lines.append(f"- {ticker}: {desc}")
                        pdf_lines.append(f"  Annual Return: {rpt.loc[ticker, 'ann_return']:.1%}")
                        pdf_lines.append(f"  Volatility: {rpt.loc[ticker, 'ann_vol']:.1%}")
                        pdf_lines.append(f"  Sharpe Ratio: {rpt.loc[ticker, 'sharpe']:.2f}")
                        pdf_lines.append("")
                    
                    pdf_data = make_pdf_report("Portfolio Analysis Report", pdf_lines)
                    if pdf_data:
                        st.download_button(
                            "⬇️ Download PDF Report",
                            data=pdf_data,
                            file_name="portfolio_analysis.pdf",
                            mime="application/pdf",
                            key="rpt_pdf"
                        )

# =============================
# Options Analysis
# =============================
elif page == "Options Analysis":
    st.header("📊 Options Analysis")
    st.caption("Option chain viewer (Yahoo), educational only.")
    
    # Ticker descriptions for quick reference
    ticker_descriptions = {
        "AAPL": "Apple Inc. - Consumer electronics, software, and services",
        "MSFT": "Microsoft Corporation - Software, cloud computing, and technology",
        "GOOGL": "Alphabet Inc. (Google) - Internet services, advertising, and technology",
        "AMZN": "Amazon.com Inc. - E-commerce, cloud computing, and digital services",
        "NVDA": "NVIDIA Corporation - Graphics processing units and AI computing",
        "META": "Meta Platforms Inc. - Social media and digital advertising",
        "TSLA": "Tesla Inc. - Electric vehicles, energy storage, and solar panels",
        "SPY": "SPDR S&P 500 ETF - Tracks the S&P 500 index (500 largest US companies)",
        "QQQ": "Invesco QQQ Trust - Tracks NASDAQ-100 (top 100 non-financial NASDAQ stocks)",
        "IWM": "iShares Russell 2000 ETF - Tracks small-cap US stocks"
    }
    
    col1, col2 = st.columns([2, 1])
    with col1:
        ticker = st.text_input("Underlying ticker", value="AAPL", key="opt_ticker")
    with col2:
        if ticker.upper() in ticker_descriptions:
            st.info(f"**{ticker.upper()}**: {ticker_descriptions[ticker.upper()]}")
        else:
            st.info("Enter a ticker symbol to view options data")
    expiries = get_option_expiries(ticker) if ticker else []
    if not expiries:
        st.warning("No options available for this ticker.")
    else:
        expiry = st.selectbox("Expiration", options=expiries, key="opt_expiry")
        chain = get_option_chain(ticker, expiry)
        if chain["calls"].empty and chain["puts"].empty:
            st.error("Failed to fetch option chain.")
        else:
            calls = chain["calls"].copy(); puts = chain["puts"].copy()
            st.subheader("Option Chain")
            tabs = st.tabs(["Calls","Puts"])
            with tabs[0]:
                st.dataframe(calls[["contractSymbol","strike","lastPrice","bid","ask","volume","openInterest"]], use_container_width=True, height=360)
            with tabs[1]:
                st.dataframe(puts[["contractSymbol","strike","lastPrice","bid","ask","volume","openInterest"]], use_container_width=True, height=360)

# =============================
# Pattern Recognition Scanner
# =============================
elif page == "Pattern Scanner":
    st.header("📊 Advanced Pattern Recognition Scanner")
    st.caption("Identify technical patterns and chart formations for winning stock opportunities")
    
    # Ticker descriptions for quick reference
    ticker_descriptions = {
        "AAPL": "Apple Inc. - Consumer electronics, software, and services",
        "MSFT": "Microsoft Corporation - Software, cloud computing, and technology",
        "GOOGL": "Alphabet Inc. (Google) - Internet services, advertising, and technology",
        "AMZN": "Amazon.com Inc. - E-commerce, cloud computing, and digital services",
        "NVDA": "NVIDIA Corporation - Graphics processing units and AI computing",
        "META": "Meta Platforms Inc. - Social media and digital advertising",
        "TSLA": "Tesla Inc. - Electric vehicles, energy storage, and solar panels",
        "SPY": "SPDR S&P 500 ETF - Tracks the S&P 500 index (500 largest US companies)",
        "QQQ": "Invesco QQQ Trust - Tracks NASDAQ-100 (top 100 non-financial NASDAQ stocks)",
        "IWM": "iShares Russell 2000 ETF - Tracks small-cap US stocks"
    }
    
    # Scanner Configuration
    st.subheader("🔧 Scanner Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📋 Watchlist Selection**")
        watchlist_type = st.selectbox(
            "Watchlist Type",
            ["SP500", "Tech Leaders", "Growth Stocks", "Custom Tickers"],
            help="**SP500**: S&P 500 companies\n**Tech Leaders**: Major technology companies\n**Growth Stocks**: High-growth potential stocks\n**Custom Tickers**: Your own ticker list"
        )
        
        pattern_type = st.selectbox(
            "Pattern Type",
            ["All Patterns", "Reversal Patterns", "Continuation Patterns", "Candlestick Patterns"],
            help="**Reversal**: Head & Shoulders, Double Top/Bottom\n**Continuation**: Flags, Pennants, Triangles\n**Candlestick**: Doji, Hammer, Shooting Star"
        )
    
    with col2:
        st.markdown("**📊 Analysis Parameters**")
        time_period = st.selectbox(
            "Time Period",
            ["1mo", "3mo", "6mo", "1y"],
            index=2,
            help="**1mo**: Short-term patterns\n**3mo**: Medium-term patterns\n**6mo**: Longer-term patterns\n**1y**: Major trend patterns"
        )
        
        min_volume = st.number_input(
            "Minimum Volume (M)",
            min_value=0.1,
            max_value=1000.0,
            value=1.0,
            step=0.1,
            help="Filter stocks by minimum average volume"
        )
    
    with col3:
        st.markdown("**🎯 Pattern Filters**")
        confidence_threshold = st.slider(
            "Minimum Confidence (%)",
            min_value=50,
            max_value=95,
            value=70,
            step=5,
            help="Pattern recognition confidence level"
        )
        
        include_volume_analysis = st.checkbox(
            "Include Volume Analysis",
            value=True,
            help="Analyze volume patterns for confirmation"
        )
    
    # Custom tickers input
    if watchlist_type == "Custom Tickers":
        st.markdown("**📝 Custom Ticker List**")
        custom_tickers = st.text_area(
            "Enter Tickers (one per line)",
            value="AAPL\nMSFT\nGOOGL\nAMZN\nNVDA\nTSLA\nMETA\nNFLX\nADBE\nCRM",
            help="Enter one ticker per line, separated by new lines"
        )
        custom_ticker_list = [t.strip().upper() for t in custom_tickers.split('\n') if t.strip()]
    else:
        custom_ticker_list = []
    
    # Pattern descriptions
    with st.expander("📚 Pattern Types Explained", expanded=False):
        st.markdown("""
        **🔄 Reversal Patterns**:
        - **Head & Shoulders**: Bearish reversal pattern with three peaks
        - **Inverse Head & Shoulders**: Bullish reversal pattern with three troughs
        - **Double Top**: Bearish reversal with two peaks at same level
        - **Double Bottom**: Bullish reversal with two troughs at same level
        - **Rounding Bottom**: Gradual bullish reversal (saucer pattern)
        
        **📈 Continuation Patterns**:
        - **Flags**: Short consolidation after strong move
        - **Pennants**: Triangle-like consolidation
        - **Triangles**: Ascending, descending, or symmetrical
        - **Wedges**: Rising or falling wedge patterns
        - **Channels**: Parallel support/resistance lines
        
        **🕯️ Candlestick Patterns**:
        - **Doji**: Indecision, potential reversal
        - **Hammer**: Bullish reversal signal
        - **Shooting Star**: Bearish reversal signal
        - **Engulfing**: Strong reversal signal
        - **Morning/Evening Star**: Three-candle reversal patterns
        """)
    
    # Run scanner button
    if st.button("🔍 Run Pattern Scanner", type="primary"):
        with st.spinner("Scanning for patterns..."):
            # Define watchlists
            if watchlist_type == "SP500":
                tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH", "JNJ", "JPM", "V", "PG", "HD", "MA", "DIS", "PYPL", "BAC", "ADBE", "CRM"]
            elif watchlist_type == "Tech Leaders":
                tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "NFLX", "ADBE", "CRM", "PYPL", "INTC", "AMD", "ORCL", "CSCO", "IBM", "QCOM", "TXN", "AVGO", "MU"]
            elif watchlist_type == "Growth Stocks":
                tickers = ["NVDA", "TSLA", "META", "NFLX", "ADBE", "CRM", "PYPL", "ZM", "SHOP", "SQ", "ROKU", "CRWD", "OKTA", "DOCU", "SNOW", "PLTR", "COIN", "RBLX", "HOOD", "LCID"]
            else:
                tickers = custom_ticker_list
            
            # Pattern detection results
            pattern_results = []
            
            for ticker in tickers:
                try:
                    # Fetch price data
                    df = fetch_price_history(ticker, period=time_period, interval="1d")
                    if df.empty or len(df) < 20:
                        continue
                    
                    # Calculate technical indicators
                    df['SMA_20'] = df['Close'].rolling(window=20).mean()
                    df['SMA_50'] = df['Close'].rolling(window=50).mean()
                    df['RSI'] = calculate_rsi(df['Close'], 14)
                    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
                    
                    # Simple pattern detection (basic implementation)
                    patterns_found = []
                    confidence = 0
                    
                    # Check for basic patterns
                    current_price = df['Close'].iloc[-1]
                    sma_20 = df['SMA_20'].iloc[-1]
                    sma_50 = df['SMA_50'].iloc[-1]
                    rsi = df['RSI'].iloc[-1]
                    
                    # Trend analysis
                    if current_price > sma_20 > sma_50:
                        patterns_found.append("Uptrend")
                        confidence += 20
                    elif current_price < sma_20 < sma_50:
                        patterns_found.append("Downtrend")
                        confidence += 20
                    
                    # RSI analysis
                    if rsi < 30:
                        patterns_found.append("Oversold")
                        confidence += 15
                    elif rsi > 70:
                        patterns_found.append("Overbought")
                        confidence += 15
                    
                    # Volume analysis
                    if include_volume_analysis:
                        recent_volume = df['Volume'].iloc[-5:].mean()
                        avg_volume = df['Volume_MA'].iloc[-1]
                        if recent_volume > avg_volume * 1.5:
                            patterns_found.append("High Volume")
                            confidence += 10
                    
                    # Price action patterns
                    recent_highs = df['High'].iloc[-20:].max()
                    recent_lows = df['Low'].iloc[-20:].min()
                    price_range = (recent_highs - recent_lows) / recent_lows
                    
                    if price_range > 0.15:
                        patterns_found.append("High Volatility")
                        confidence += 10
                    
                    # Check if confidence meets threshold
                    if confidence >= confidence_threshold and patterns_found:
                        # Calculate momentum
                        momentum = ((current_price - df['Close'].iloc[-20]) / df['Close'].iloc[-20]) * 100
                        
                        # Volume check
                        avg_volume_m = df['Volume'].mean() / 1_000_000
                        if avg_volume_m >= min_volume:
                            pattern_results.append({
                                'Ticker': ticker,
                                'Patterns': ', '.join(patterns_found),
                                'Confidence': f"{confidence}%",
                                'Current Price': f"${current_price:.2f}",
                                'Momentum': f"{momentum:.1f}%",
                                'Volume (M)': f"{avg_volume_m:.1f}",
                                'RSI': f"{rsi:.1f}",
                                'Trend': "Bullish" if current_price > sma_20 else "Bearish"
                            })
                
                except Exception as e:
                    pass  # Silent error handling
            
            # Display results
            if pattern_results:
                st.subheader("🎯 Pattern Detection Results")
                
                # Convert to DataFrame
                results_df = pd.DataFrame(pattern_results)
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Stocks Scanned", len(tickers))
                with col2:
                    st.metric("Patterns Found", len(pattern_results))
                with col3:
                    bullish_count = len(results_df[results_df['Trend'] == 'Bullish'])
                    st.metric("Bullish Patterns", bullish_count)
                with col4:
                    avg_confidence = np.mean([float(c.replace('%', '')) for c in results_df['Confidence']])
                    st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                
                # Results table
                st.dataframe(
                    results_df,
                    use_container_width=True,
                    column_config={
                        "Ticker": st.column_config.TextColumn(
                            "Ticker",
                            help="Stock symbol"
                        ),
                        "Patterns": st.column_config.TextColumn(
                            "Detected Patterns",
                            help="Technical patterns identified"
                        ),
                        "Confidence": st.column_config.TextColumn(
                            "Confidence",
                            help="Pattern recognition confidence level"
                        ),
                        "Current Price": st.column_config.TextColumn(
                            "Price",
                            help="Current stock price"
                        ),
                        "Momentum": st.column_config.TextColumn(
                            "Momentum",
                            help="20-day price momentum"
                        ),
                        "Volume (M)": st.column_config.TextColumn(
                            "Volume",
                            help="Average daily volume in millions"
                        ),
                        "RSI": st.column_config.TextColumn(
                            "RSI",
                            help="Relative Strength Index (14-period)"
                        ),
                        "Trend": st.column_config.TextColumn(
                            "Trend",
                            help="Overall price trend direction"
                        )
                    }
                )
                
                # Export functionality
                col1, col2 = st.columns(2)
                with col1:
                    csv_data = results_df.to_csv(index=False)
                    st.download_button(
                        "📊 Download Results (CSV)",
                        csv_data,
                        file_name=f"pattern_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    st.info("💡 **Tip**: Focus on stocks with high confidence patterns and strong volume confirmation for best results.")
            
            else:
                st.warning("No patterns found matching your criteria. Try adjusting the confidence threshold or watchlist selection.")

# =============================
# Options Flow Analysis
# =============================
elif page == "Options Flow":
    st.header("📊 Advanced Options Flow Analysis")
    st.caption("Monitor unusual options activity and institutional flow for trading opportunities")
    
    # Ticker descriptions for quick reference
    ticker_descriptions = {
        "AAPL": "Apple Inc. - Consumer electronics, software, and services",
        "MSFT": "Microsoft Corporation - Software, cloud computing, and technology",
        "GOOGL": "Alphabet Inc. (Google) - Internet services, advertising, and technology",
        "AMZN": "Amazon.com Inc. - E-commerce, cloud computing, and digital services",
        "NVDA": "NVIDIA Corporation - Graphics processing units and AI computing",
        "META": "Meta Platforms Inc. - Social media and digital advertising",
        "TSLA": "Tesla Inc. - Electric vehicles, energy storage, and solar panels",
        "SPY": "SPDR S&P 500 ETF - Tracks the S&P 500 index (500 largest US companies)",
        "QQQ": "Invesco QQQ Trust - Tracks NASDAQ-100 (top 100 non-financial NASDAQ stocks)",
        "IWM": "iShares Russell 2000 ETF - Tracks small-cap US stocks"
    }
    
    # Scanner Configuration
    st.subheader("🔧 Flow Scanner Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📋 Scanner Settings**")
        flow_ticker = st.text_input(
            "Ticker Symbol",
            value="AAPL",
            help="Enter ticker to analyze options flow"
        )
        
        if flow_ticker.upper() in ticker_descriptions:
            st.info(f"**{flow_ticker.upper()}**: {ticker_descriptions[flow_ticker.upper()]}")
        
        expiry_filter = st.selectbox(
            "Expiration Focus",
            ["All Expirations", "This Week", "Next Week", "This Month", "Next Month"],
            help="Focus on specific expiration periods"
        )
    
    with col2:
        st.markdown("**📊 Flow Filters**")
        min_volume = st.number_input(
            "Min Volume",
            min_value=1,
            max_value=10000,
            value=100,
            help="Minimum options volume to display"
        )
        
        min_open_interest = st.number_input(
            "Min Open Interest",
            min_value=1,
            max_value=10000,
            value=50,
            help="Minimum open interest to display"
        )
        
        unusual_activity_threshold = st.slider(
            "Unusual Activity Threshold",
            min_value=1.5,
            max_value=10.0,
            value=3.0,
            step=0.5,
            help="Volume vs Open Interest ratio for unusual activity"
        )
    
    with col3:
        st.markdown("**🎯 Analysis Options**")
        include_puts = st.checkbox("Include Puts", value=True)
        include_calls = st.checkbox("Include Calls", value=True)
        
        flow_analysis_type = st.selectbox(
            "Analysis Type",
            ["All Activity", "Unusual Activity Only", "Large Trades Only"],
            help="Filter options activity by type"
        )
    
    # Run analysis button
    if st.button("🔍 Analyze Options Flow", type="primary"):
        with st.spinner("Analyzing options flow..."):
            try:
                # Get option expirations
                expirations = get_option_expiries(flow_ticker)
                if not expirations:
                    st.error(f"No options data available for {flow_ticker}")
                    st.stop()
                
                # Filter expirations based on selection
                if expiry_filter == "This Week":
                    # Logic to filter for this week's expirations
                    pass
                
                flow_results = []
                
                # Analyze each expiration
                for expiry in expirations[:5]:  # Limit to first 5 expirations
                    try:
                        option_chain = get_option_chain(flow_ticker, expiry)
                        calls = option_chain.get('calls', pd.DataFrame())
                        puts = option_chain.get('puts', pd.DataFrame())
                        
                        # Analyze calls
                        if include_calls and not calls.empty:
                            for _, call in calls.iterrows():
                                try:
                                    volume = call.get('volume', 0)
                                    open_interest = call.get('openInterest', 0)
                                    last_price = call.get('lastPrice', 0)
                                    strike = call.get('strike', 0)
                                    
                                    if volume >= min_volume and open_interest >= min_open_interest:
                                        # Calculate unusual activity
                                        volume_oi_ratio = volume / open_interest if open_interest > 0 else 0
                                        
                                        if flow_analysis_type == "All Activity" or \
                                           (flow_analysis_type == "Unusual Activity Only" and volume_oi_ratio >= unusual_activity_threshold) or \
                                           (flow_analysis_type == "Large Trades Only" and volume >= 500):
                                            
                                            flow_results.append({
                                                'Expiration': expiry,
                                                'Type': 'Call',
                                                'Strike': f"${strike:.0f}",
                                                'Volume': volume,
                                                'Open Interest': open_interest,
                                                'Volume/OI Ratio': f"{volume_oi_ratio:.2f}",
                                                'Last Price': f"${last_price:.2f}",
                                                'Unusual': "Yes" if volume_oi_ratio >= unusual_activity_threshold else "No"
                                            })
                                except:
                                    pass
                        
                        # Analyze puts
                        if include_puts and not puts.empty:
                            for _, put in puts.iterrows():
                                try:
                                    volume = put.get('volume', 0)
                                    open_interest = put.get('openInterest', 0)
                                    last_price = put.get('lastPrice', 0)
                                    strike = put.get('strike', 0)
                                    
                                    if volume >= min_volume and open_interest >= min_open_interest:
                                        # Calculate unusual activity
                                        volume_oi_ratio = volume / open_interest if open_interest > 0 else 0
                                        
                                        if flow_analysis_type == "All Activity" or \
                                           (flow_analysis_type == "Unusual Activity Only" and volume_oi_ratio >= unusual_activity_threshold) or \
                                           (flow_analysis_type == "Large Trades Only" and volume >= 500):
                                            
                                            flow_results.append({
                                                'Expiration': expiry,
                                                'Type': 'Put',
                                                'Strike': f"${strike:.0f}",
                                                'Volume': volume,
                                                'Open Interest': open_interest,
                                                'Volume/OI Ratio': f"{volume_oi_ratio:.2f}",
                                                'Last Price': f"${last_price:.2f}",
                                                'Unusual': "Yes" if volume_oi_ratio >= unusual_activity_threshold else "No"
                                            })
                                except:
                                    pass
                    
                    except Exception as e:
                        pass  # Silent error handling
                
                # Display results
                if flow_results:
                    st.subheader("📊 Options Flow Analysis Results")
                    
                    # Convert to DataFrame
                    flow_df = pd.DataFrame(flow_results)
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Options", len(flow_results))
                    with col2:
                        calls_count = len(flow_df[flow_df['Type'] == 'Call'])
                        st.metric("Calls", calls_count)
                    with col3:
                        puts_count = len(flow_df[flow_df['Type'] == 'Put'])
                        st.metric("Puts", puts_count)
                    with col4:
                        unusual_count = len(flow_df[flow_df['Unusual'] == 'Yes'])
                        st.metric("Unusual Activity", unusual_count)
                    
                    # Results table
                    st.dataframe(
                        flow_df,
                        use_container_width=True,
                        column_config={
                            "Expiration": st.column_config.TextColumn(
                                "Expiration",
                                help="Option expiration date"
                            ),
                            "Type": st.column_config.TextColumn(
                                "Type",
                                help="Call or Put option"
                            ),
                            "Strike": st.column_config.TextColumn(
                                "Strike Price",
                                help="Option strike price"
                            ),
                            "Volume": st.column_config.NumberColumn(
                                "Volume",
                                help="Today's trading volume"
                            ),
                            "Open Interest": st.column_config.NumberColumn(
                                "Open Interest",
                                help="Total open contracts"
                            ),
                            "Volume/OI Ratio": st.column_config.TextColumn(
                                "Volume/OI",
                                help="Volume to Open Interest ratio"
                            ),
                            "Last Price": st.column_config.TextColumn(
                                "Last Price",
                                help="Last traded price"
                            ),
                            "Unusual": st.column_config.TextColumn(
                                "Unusual",
                                help="Unusual activity indicator"
                            )
                        }
                    )
                    
                    # Export functionality
                    col1, col2 = st.columns(2)
                    with col1:
                        csv_data = flow_df.to_csv(index=False)
                        st.download_button(
                            "📊 Download Flow Data (CSV)",
                            csv_data,
                            file_name=f"options_flow_{flow_ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        st.info("💡 **Tip**: High Volume/OI ratios often indicate institutional activity and potential price movement.")
                
                else:
                    st.warning("No options flow data found matching your criteria.")
            
            except Exception as e:
                st.error(f"Error analyzing options flow: {str(e)}")



# =============================
# Options Strategy Builder (Advanced Options Analysis)
# =============================
elif page == "Options Strategy Builder":
    st.header("🎯 Advanced Options Strategy Builder")
    st.caption("Discover high-probability options strategies with detailed analysis and risk assessment")

    # Ticker descriptions for quick reference
    ticker_descriptions = {
        "AAPL": "Apple Inc. - Consumer electronics, software, and services",
        "MSFT": "Microsoft Corporation - Software, cloud computing, and technology",
        "GOOGL": "Alphabet Inc. (Google) - Internet services, advertising, and technology",
        "AMZN": "Amazon.com Inc. - E-commerce, cloud computing, and digital services",
        "NVDA": "NVIDIA Corporation - Graphics processing units and AI computing",
        "META": "Meta Platforms Inc. - Social media and digital advertising",
        "TSLA": "Tesla Inc. - Electric vehicles, energy storage, and solar panels",
        "SPY": "SPDR S&P 500 ETF - Tracks the S&P 500 index (500 largest US companies)",
        "QQQ": "Invesco QQQ Trust - Tracks NASDAQ-100 (top 100 non-financial NASDAQ stocks)",
        "IWM": "iShares Russell 2000 ETF - Tracks small-cap US stocks"
    }

    # Inputs
    col1, col2 = st.columns([2, 1])
    with col1:
        sym = st.text_input("Ticker symbol", value="AAPL", key="best_ticker")
    with col2:
        if sym.upper() in ticker_descriptions:
            st.info(f"**{sym.upper()}**: {ticker_descriptions[sym.upper()]}")
        else:
            st.info("Enter a ticker symbol to analyze options")
    # Strategy selection with detailed descriptions
    st.subheader("📊 Strategy Selection")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        strategy_category = st.selectbox(
            "Strategy Category",
            ["Premium Collection", "Directional", "All Strategies"],
            key="strategy_category",
            help="Premium Collection: High probability income strategies. Directional: Betting on price movement. All: Complete analysis."
        )
    
    with col2:
        if strategy_category == "Premium Collection":
            strategies = st.multiselect(
                "Premium Collection Strategies",
                ["Iron Condor", "Cash-Secured Put", "Covered Call", "Calendar Spread"],
                default=["Iron Condor"],
                key="best_strats",
                help="High-probability strategies that collect premium and profit from time decay"
            )
        elif strategy_category == "Directional":
            strategies = st.multiselect(
                "Directional Strategies",
                ["Bull Put Spread", "Bear Call Spread", "Butterfly Spread", "Straddle/Strangle"],
                default=["Bull Put Spread", "Bear Call Spread"],
                key="best_strats",
                help="Strategies that profit from directional price movement"
            )
        else:
            strategies = st.multiselect(
                "All Available Strategies",
                ["Bull Put Spread", "Bear Call Spread", "Iron Condor", "Cash-Secured Put", "Covered Call", "Calendar Spread", "Butterfly Spread", "Straddle/Strangle"],
                default=["Bull Put Spread", "Bear Call Spread", "Iron Condor"],
                key="best_strats",
                help="Complete strategy analysis for all market conditions"
            )
    # Advanced parameters
    st.subheader("⚙️ Strategy Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        target_delta = st.slider(
            "Target Delta (Short Leg)", 
            0.10, 0.40, 0.25, 0.01, 
            key="best_delta",
            help="Delta measures option sensitivity to stock price. 0.25 = 25% chance of being in-the-money at expiration."
        )
        min_pop = st.slider(
            "Minimum POP (%)", 
            50, 90, 70, 5, 
            key="min_pop",
            help="Probability of Profit - minimum threshold for strategy selection"
        )
    
    with col2:
        wing_pct = st.slider(
            "Wing Width (% of spot)", 
            1.0, 15.0, 5.0, 0.5, 
            key="best_wing",
            help="Distance between strike prices in spreads. Wider = more risk/reward, narrower = less risk/reward."
        )
        min_roi = st.slider(
            "Minimum ROI (%)", 
            10, 100, 20, 5, 
            key="min_roi",
            help="Return on Investment - minimum risk-reward ratio"
        )
    
    with col3:
        scan_n = st.slider(
            "Expirations to Scan", 
            1, 8, 4, 1, 
            key="best_scan",
            help="Number of option expiration dates to analyze. More expirations = more opportunities but longer scan time."
        )
        max_risk = st.number_input(
            "Max Risk per Trade ($)", 
            min_value=100, max_value=10000, value=1000, step=100,
            key="max_risk",
            help="Maximum dollar risk per strategy (for position sizing)"
        )
    
    # Market condition analysis
    st.subheader("📊 Market Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        market_outlook = st.selectbox(
            "Market Outlook",
            ["Bullish", "Bearish", "Neutral", "High Volatility", "Low Volatility"],
            key="market_outlook",
            help="Current market sentiment to filter appropriate strategies"
        )
    
    with col2:
        include_analysis = st.checkbox(
            "Include Detailed Analysis",
            value=True,
            key="include_analysis",
            help="Show strategy explanations, risk assessment, and management tips"
        )

    # Helpers
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0))) if hasattr(math, "erf") else 0.0

    def _bs_delta_call(S, K, r, T, sigma):
        try:
            if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
                return float("nan")
            d1 = (math.log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*math.sqrt(T))
            return _norm_cdf(d1)
        except Exception:
            return float("nan")

    def _bs_delta_put(S, K, r, T, sigma):
        dc = _bs_delta_call(S, K, r, T, sigma)
        return dc - 1 if dc==dc else float("nan")

    def _mid(row):
        b = float(row.get("bid", np.nan)) if "bid" in row else np.nan
        a = float(row.get("ask", np.nan)) if "ask" in row else np.nan
        lp = float(row.get("lastPrice", np.nan)) if "lastPrice" in row else np.nan
        vals = [x for x in [b, a, lp] if not np.isnan(x)]
        return float(np.mean(vals)) if vals else 0.0

    # Build suggestions
    if st.button("🔍 Analyze Options Strategies", key="best_btn") and sym:
        S = fetch_live_price(sym)
        if np.isnan(S):
            st.warning("Live price unavailable; using last close.")
            hist = fetch_price_history(sym, period="5d", interval="1d")
            if not hist.empty:
                S = float(hist["Close"].iloc[-1])
        if np.isnan(S):
            st.error("Unable to determine spot price.")
        else:
            expiries = get_option_expiries(sym)
            if not expiries:
                st.error("No expirations available.")
            else:
                rf = st.session_state.get("risk_free", RISK_FREE_DEFAULT)
                picks = []
                for expiry in expiries[:scan_n]:
                    chain = get_option_chain(sym, expiry)
                    calls = chain["calls"].copy(); puts = chain["puts"].copy()
                    for df in (calls, puts):
                        if "impliedVolatility" not in df.columns:
                            df["impliedVolatility"] = np.nan
                    try:
                        days = (pd.to_datetime(expiry) - pd.Timestamp.today().normalize()).days
                        T = max(days, 1) / 365.0
                    except Exception:
                        T = 30/365.0

                    if not calls.empty:
                        calls["delta_est"] = calls.apply(lambda r: _bs_delta_call(S, float(r["strike"]), rf, T, float(r.get("impliedVolatility", np.nan) or np.nan)), axis=1)
                    if not puts.empty:
                        puts["delta_est"] = puts.apply(lambda r: _bs_delta_put (S, float(r["strike"]), rf, T, float(r.get("impliedVolatility", np.nan) or np.nan)), axis=1)

                    # Bull Put Spread
                    if "Bull Put Spread" in strategies and not puts.empty:
                        cand_sp = puts.copy(); cand_sp["abs_delta"] = cand_sp["delta_est"].abs()
                        cand_sp = cand_sp.loc[(cand_sp["abs_delta"] >= target_delta-0.08) & (cand_sp["abs_delta"] <= target_delta+0.08)]
                        if not cand_sp.empty:
                            short_put = cand_sp.iloc[(cand_sp["strike"] - S*(1.0 - wing_pct/100.0)).abs().argsort()[:1]].iloc[0]
                            long_put  = puts.iloc[(puts["strike"] - float(short_put["strike"]) * (1.0 - wing_pct/100.0)).abs().argsort()[:1]].iloc[0]
                            width = float(short_put["strike"]) - float(long_put["strike"])
                            credit = max(0.0, _mid(short_put) - _mid(long_put))
                            max_loss = max(width - credit, 1e-6)
                            pop = 1.0 - abs(float(short_put.get("delta_est", np.nan))) if short_put.get("delta_est", np.nan)==short_put.get("delta_est", np.nan) else np.nan
                            roi = credit / max_loss
                            score = (roi * pop) if pop==pop else 0.0
                            picks.append({
                                "Strategy":"Bull Put Spread","Expiry":expiry,
                                "Short": float(short_put["strike"]), "Long": float(long_put["strike"]),
                                "Width": width, "Credit": credit, "MaxLoss": max_loss,
                                "POP": pop, "ROI": roi, "Score": score
                            })

                    # Bear Call Spread
                    if "Bear Call Spread" in strategies and not calls.empty:
                        cand_sc = calls.copy(); cand_sc["abs_delta"] = cand_sc["delta_est"].abs()
                        cand_sc = cand_sc.loc[(cand_sc["abs_delta"] >= target_delta-0.08) & (cand_sc["abs_delta"] <= target_delta+0.08)]
                        if not cand_sc.empty:
                            short_call = cand_sc.iloc[(cand_sc["strike"] - S*(1.0 + wing_pct/100.0)).abs().argsort()[:1]].iloc[0]
                            long_call  = calls.iloc[(calls["strike"] - float(short_call["strike"]) * (1.0 + wing_pct/100.0)).abs().argsort()[:1]].iloc[0]
                            width = float(long_call["strike"]) - float(short_call["strike"])
                            credit = max(0.0, _mid(short_call) - _mid(long_call))
                            max_loss = max(width - credit, 1e-6)
                            pop = 1.0 - float(short_call.get("delta_est", np.nan)) if short_call.get("delta_est", np.nan)==short_call.get("delta_est", np.nan) else np.nan
                            roi = credit / max_loss
                            score = (roi * pop) if pop==pop else 0.0
                            picks.append({
                                "Strategy":"Bear Call Spread","Expiry":expiry,
                                "Short": float(short_call["strike"]), "Long": float(long_call["strike"]),
                                "Width": width, "Credit": credit, "MaxLoss": max_loss,
                                "POP": pop, "ROI": roi, "Score": score
                            })

                    # Iron Condor
                    if "Iron Condor" in strategies and not calls.empty and not puts.empty:
                        cand_sc = calls.copy(); cand_sc["abs_delta"] = cand_sc["delta_est"].abs()
                        cand_sp = puts.copy();  cand_sp["abs_delta"] = cand_sp["delta_est"].abs()
                        cand_sc = cand_sc.loc[(cand_sc["abs_delta"] >= target_delta-0.08) & (cand_sc["abs_delta"] <= target_delta+0.08)]
                        cand_sp = cand_sp.loc[(cand_sp["abs_delta"] >= target_delta-0.08) & (cand_sp["abs_delta"] <= target_delta+0.08)]
                        if not cand_sc.empty and not cand_sp.empty:
                            sc = cand_sc.iloc[(cand_sc["strike"] - S*(1.0 + wing_pct/100.0)).abs().argsort()[:1]].iloc[0]
                            sp = cand_sp.iloc[(cand_sp["strike"] - S*(1.0 - wing_pct/100.0)).abs().argsort()[:1]].iloc[0]
                            lc = calls.iloc[(calls["strike"] - float(sc["strike"]) * (1.0 + wing_pct/100.0)).abs().argsort()[:1]].iloc[0]
                            lp = puts .iloc[(puts ["strike"] - float(sp["strike"]) * (1.0 - wing_pct/100.0)).abs().argsort()[:1]].iloc[0]
                            width_call = float(lc["strike"]) - float(sc["strike"])
                            width_put  = float(sp["strike"]) - float(lp["strike"])
                            credit = max(0.0, (_mid(sc) - _mid(lc)) + (_mid(sp) - _mid(lp)))
                            worst_width = max(width_call, width_put)
                            max_loss = max(worst_width - credit, 1e-6)
                            pop_call = 1.0 - float(sc.get("delta_est", np.nan)) if sc.get("delta_est", np.nan)==sc.get("delta_est", np.nan) else np.nan
                            pop_put  = 1.0 - abs(float(sp.get("delta_est", np.nan))) if sp.get("delta_est", np.nan)==sp.get("delta_est", np.nan) else np.nan
                            pop = (pop_call * pop_put) if (pop_call==pop_call and pop_put==pop_put) else np.nan
                            roi = credit / max_loss
                            score = (roi * pop) if pop==pop else 0.0
                            picks.append({
                                "Strategy":"Iron Condor","Expiry":expiry,
                                "Short": f"P {float(sp['strike']):.2f} / C {float(sc['strike']):.2f}",
                                "Long":  f"P {float(lp['strike']):.2f} / C {float(lc['strike']):.2f}",
                                "Width": worst_width, "Credit": credit, "MaxLoss": max_loss,
                                "POP": pop, "ROI": roi, "Score": score
                            })

                if not picks:
                    st.warning("No viable candidates found. Try adjusting parameters or scanning more expirations.")
                else:
                    # Filter results based on criteria
                    dfp = pd.DataFrame(picks)
                    dfp = dfp[(dfp['POP'] >= min_pop/100) & (dfp['ROI'] >= min_roi/100)]
                    
                    if dfp.empty:
                        st.warning(f"No strategies meet your criteria (POP ≥ {min_pop}%, ROI ≥ {min_roi}%). Try relaxing your requirements.")
                    else:
                        dfp = dfp.sort_values("Score", ascending=False).head(5).reset_index(drop=True)
                        
                        # Display results with enhanced formatting
                        st.subheader("🎯 Top Strategy Recommendations")
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Strategies Found", len(dfp))
                        with col2:
                            avg_pop = dfp['POP'].mean()
                            st.metric("Avg POP", f"{avg_pop:.1%}")
                        with col3:
                            avg_roi = dfp['ROI'].mean()
                            st.metric("Avg ROI", f"{avg_roi:.1%}")
                        with col4:
                            best_score = dfp['Score'].max()
                            st.metric("Best Score", f"{best_score:.3f}")
                        
                        # Enhanced results table
                        st.dataframe(
                            dfp.style.format({
                                "Width":"{:.2f}", "Credit":"{:.2f}", "MaxLoss":"{:.2f}",
                                "POP":"{:.0%}", "ROI":"{:.1%}", "Score":"{:.3f}"
                            }),
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Strategy": st.column_config.TextColumn("Strategy", width="medium"),
                                "Expiry": st.column_config.TextColumn("Expiry", width="small"),
                                "Short": st.column_config.NumberColumn("Short Strike", format="%.2f"),
                                "Long": st.column_config.NumberColumn("Long Strike", format="%.2f"),
                                "Width": st.column_config.NumberColumn("Width", format="%.2f"),
                                "Credit": st.column_config.NumberColumn("Credit", format="%.2f"),
                                "MaxLoss": st.column_config.NumberColumn("Max Loss", format="%.2f"),
                                "POP": st.column_config.NumberColumn("POP", format="%.0%"),
                                "ROI": st.column_config.NumberColumn("ROI", format="%.1%"),
                                "Score": st.column_config.NumberColumn("Score", format="%.3f"),
                            }
                        )
                        
                        # Detailed analysis for each strategy
                        if include_analysis:
                            st.subheader("📊 Detailed Strategy Analysis")
                            
                            for idx, row in dfp.iterrows():
                                with st.expander(f"🔍 {row['Strategy']} - {row['Expiry']} (Score: {row['Score']:.3f})", expanded=(idx==0)):
                                    col1, col2 = st.columns([2, 1])
                                    
                                    with col1:
                                        st.markdown(f"""
                                        **Strategy Details:**
                                        - **Type**: {row['Strategy']}
                                        - **Expiration**: {row['Expiry']}
                                        - **Short Strike**: ${row['Short']:.2f}
                                        - **Long Strike**: ${row['Long']:.2f}
                                        - **Spread Width**: ${row['Width']:.2f}
                                        """)
                                        
                                        st.markdown(f"""
                                        **Risk/Reward Analysis:**
                                        - **Credit Received**: ${row['Credit']:.2f}
                                        - **Maximum Loss**: ${row['MaxLoss']:.2f}
                                        - **Probability of Profit**: {row['POP']:.1%}
                                        - **Return on Investment**: {row['ROI']:.1%}
                                        - **Risk-Reward Ratio**: 1:{row['ROI']:.1f}
                                        """)
                                        
                                        # Position sizing
                                        if row['MaxLoss'] > 0:
                                            max_contracts = int(max_risk / row['MaxLoss'])
                                            st.markdown(f"""
                                        **Position Sizing:**
                                        - **Max Contracts**: {max_contracts} (based on ${max_risk} risk)
                                        - **Total Credit**: ${row['Credit'] * max_contracts:.2f}
                                        - **Total Risk**: ${row['MaxLoss'] * max_contracts:.2f}
                                        """)
                                    
                                    with col2:
                                        # Strategy-specific advice
                                        if "Bull Put" in row['Strategy']:
                                            st.info("""
                                            **Bull Put Spread Tips:**
                                            - Best for bullish outlook
                                            - Close at 50-80% profit
                                            - Roll if challenged
                                            - Avoid earnings
                                            """)
                                        elif "Bear Call" in row['Strategy']:
                                            st.info("""
                                            **Bear Call Spread Tips:**
                                            - Best for bearish outlook
                                            - Close at 50-80% profit
                                            - Roll if challenged
                                            - Avoid earnings
                                            """)
                                        elif "Iron Condor" in row['Strategy']:
                                            st.info("""
                                            **Iron Condor Tips:**
                                            - Best for neutral outlook
                                            - Close at 50-80% profit
                                            - Roll if challenged
                                            - Avoid earnings
                                            """)
                                        
                                        # Risk management
                                        st.warning(f"""
                                        **Risk Management:**
                                        - Stop Loss: ${row['Credit'] * 2:.2f}
                                        - Profit Target: ${row['Credit'] * 0.7:.2f}
                                        - Max Risk: ${max_risk}
                                        """)
                        
                        # Order tickets
                        st.subheader("📋 Order Tickets")
                        tickets = []
                        for _, r in dfp.iterrows():
                            if r["Strategy"] == "Bull Put Spread":
                                tickets.append(f"SELL 1 {r['Expiry']} PUT {r['Short']:.2f}  /  BUY 1 PUT {r['Long']:.2f}  Net +${r['Credit']:.2f}")
                            elif r["Strategy"] == "Bear Call Spread":
                                tickets.append(f"SELL 1 {r['Expiry']} CALL {r['Short']:.2f}  /  BUY 1 CALL {r['Long']:.2f}  Net +${r['Credit']:.2f}")
                            else:
                                tickets.append(f"SELL 1 {r['Expiry']} IRON CONDOR  (P {r['Short'].split('/')[0].strip()}  /  C {r['Short'].split('/')[1].strip()})  Net +${r['Credit']:.2f}")
                        
                        st.code("\n".join(tickets))
                        
                        # Market condition summary
                        st.subheader("📈 Market Condition Summary")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.info(f"""
                            **Current Market Outlook**: {market_outlook}
                            **Recommended Strategies**: {', '.join(dfp['Strategy'].unique())}
                            **Average POP**: {avg_pop:.1%}
                            **Average ROI**: {avg_roi:.1%}
                            """)
                        with col2:
                            st.success(f"""
                            **Risk Management Applied**:
                            - Max Risk per Trade: ${max_risk}
                            - Minimum POP: {min_pop}%
                            - Minimum ROI: {min_roi}%
                            - Strategies Analyzed: {len(picks)}
                            """)

# =============================
# Weekly Watchlist
# =============================
elif page == "Weekly Watchlist":
    st.header("📊 Advanced Weekly Watchlist Scanner")
    st.caption("Discover high-potential stocks and ETFs with comprehensive analysis and customizable filters")
    
    # Main screen filters and controls
    st.subheader("🔧 Scanner Configuration")
    
    # Filter controls in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📋 Watchlist Selection**")
        watchlist_type = st.selectbox(
            "Watchlist Type",
            ["Curated List", "Custom Tickers", "Sector ETFs", "Market Leaders"],
            help="**Curated List**: Hand-picked mix of major stocks and ETFs\n**Custom Tickers**: Add your own tickers\n**Sector ETFs**: Focus on sector-specific ETFs\n**Market Leaders**: Top market cap companies"
        )
        
        time_period = st.selectbox(
            "Analysis Period",
            ["5d", "1wk", "2wk", "1mo"],
            index=0,
            help="**5d**: Short-term momentum (1 week)\n**1wk**: Weekly momentum\n**2wk**: Bi-weekly momentum\n**1mo**: Monthly momentum"
        )
    
    with col2:
        st.markdown("**📊 Analysis Options**")
        use_health = st.checkbox(
            "Include Health Score", 
            value=True, 
            help="**Health Score**: Combines momentum (25%), profitability (25%), growth (25%), and debt metrics (25%). Higher scores indicate stronger fundamentals."
        )
        
        min_market_cap = st.selectbox(
            "Minimum Market Cap",
            ["Any", "Micro ($300M+)", "Small ($2B+)", "Mid ($10B+)", "Large ($50B+)"],
            index=2,
            help="**Market Cap Filter**:\n• Micro: $300M+ (higher risk/reward)\n• Small: $2B+ (growth potential)\n• Mid: $10B+ (balanced)\n• Large: $50B+ (stability)"
        )
    
    with col3:
        st.markdown("**📈 Additional Filters**")
        min_momentum = st.slider(
            "Minimum Momentum (%)",
            min_value=-50.0,
            max_value=50.0,
            value=-20.0,
            step=5.0,
            help="Filter stocks by minimum momentum performance. Negative values include declining stocks."
        )
        
        min_health_score = st.slider(
            "Minimum Health Score",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=5.0,
            help="Filter stocks by minimum health score. Higher values show stronger fundamentals."
        )
    
    # Custom tickers input (full width when selected)
    if watchlist_type == "Custom Tickers":
        st.markdown("**📝 Custom Ticker List**")
        custom_tickers = st.text_area(
            "Enter Tickers (one per line)",
            value="AAPL\nMSFT\nGOOGL\nAMZN\nNVDA\nTSLA\nMETA\nNFLX\nADBE\nCRM",
            help="**Instructions**:\n• Enter one ticker per line\n• Use uppercase (AAPL, not aapl)\n• Include major stocks and ETFs\n• Separate with new lines\n\n**Examples**:\nAAPL\nMSFT\nGOOGL\nSPY\nQQQ\nIWM"
        )
        custom_ticker_list = [t.strip().upper() for t in custom_tickers.split('\n') if t.strip()]
    else:
        custom_ticker_list = []
    
    # Watchlist descriptions
    with st.expander("📚 Watchlist Descriptions & Use Cases", expanded=False):
        st.markdown("""
        **🎯 Curated List**: 
        - **Best for**: General market analysis and diversified exposure
        - **Includes**: Major ETFs (SPY, QQQ, DIA), sector ETFs (XLF, XLV, XLE), and top stocks (AAPL, MSFT, NVDA)
        - **Use case**: Daily market overview and broad market sentiment
        
        **📝 Custom Tickers**: 
        - **Best for**: Focused analysis on specific stocks or sectors
        - **Includes**: Any stocks/ETFs you specify
        - **Use case**: Tracking specific companies, sectors, or personal watchlists
        
        **🏭 Sector ETFs**: 
        - **Best for**: Sector rotation analysis and sector-specific opportunities
        - **Includes**: Technology (XLK), Financials (XLF), Healthcare (XLV), Energy (XLE), and ARK funds
        - **Use case**: Identifying leading sectors and sector rotation strategies
        
        **👑 Market Leaders**: 
        - **Best for**: Large-cap stability and blue-chip analysis
        - **Includes**: Top 30 companies by market cap (AAPL, MSFT, GOOGL, AMZN, etc.)
        - **Use case**: Conservative investing and large-cap momentum analysis
        """)
    
    # Analysis period descriptions
    with st.expander("⏰ Analysis Periods Explained", expanded=False):
        st.markdown("""
        **📅 5 Days (1 Week)**:
        - **Best for**: Short-term momentum and swing trading
        - **Shows**: Recent price action and immediate momentum
        - **Use case**: Quick market sentiment and short-term opportunities
        
        **📅 1 Week**:
        - **Best for**: Weekly momentum analysis
        - **Shows**: Weekly performance trends
        - **Use case**: Weekly trading strategies and momentum confirmation
        
        **📅 2 Weeks**:
        - **Best for**: Medium-term momentum analysis
        - **Shows**: Bi-weekly trends and momentum building
        - **Use case**: Medium-term position sizing and trend confirmation
        
        **📅 1 Month**:
        - **Best for**: Monthly trend analysis and position building
        - **Shows**: Monthly performance and longer-term momentum
        - **Use case**: Monthly portfolio rebalancing and trend analysis
        """)

    # Define watchlists
    curated_list = ["SPY", "QQQ", "DIA", "IWM", "XLF", "XLV", "XLE", "XLK", "SMH", "ARKK",
                    "AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "GOOGL", "JPM", "XOM", "UNH",
                    "V", "PG", "HD", "MA", "DIS", "JNJ", "BRK-B", "NFLX", "ADBE", "CRM"]
    
    sector_etfs = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLP", "XLU", "XLB", "XLRE", "XLC",
                   "SMH", "ARKK", "ARKG", "ARKF", "ARKW", "ARKQ", "ARKX", "ARKO"]
    
    market_leaders = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", 
                     "UNH", "JNJ", "JPM", "V", "PG", "HD", "MA", "DIS", "XOM", "PFE", 
                     "ABBV", "KO", "PEP", "TMO", "AVGO", "COST", "MRK", "WMT", "BAC", 
                     "LLY", "ABT", "CVX"]
    
    # Select ticker list based on user choice
    if watchlist_type == "Curated List":
        ticker_list = curated_list
    elif watchlist_type == "Custom Tickers":
        ticker_list = custom_ticker_list
    elif watchlist_type == "Sector ETFs":
        ticker_list = sector_etfs
    else:  # Market Leaders
        ticker_list = market_leaders

    @st.cache_data(ttl=600)
    def _ww_fetch_fundamentals(ticker: str) -> dict:
        """Fetch fundamental data with enhanced error handling"""
        out = {"market_cap": np.nan, "profitMargins": np.nan, "returnOnEquity": np.nan,
               "revenueGrowth": np.nan, "debtToEquity": np.nan, "beta": np.nan, "volume": np.nan}
        try:
            tk = yf.Ticker(ticker.upper().strip())
            
            # Try fast_info first
            fi = getattr(tk, "fast_info", None)
            if fi and getattr(fi, "market_cap", None) is not None:
                out["market_cap"] = float(fi.market_cap)
            if fi and getattr(fi, "volume", None) is not None:
                out["volume"] = float(fi.volume)
            
            # Get detailed info
            try:
                info = tk.get_info()
                for k in ["profitMargins", "returnOnEquity", "revenueGrowth", "debtToEquity", "marketCap", "beta"]:
                    v = info.get(k)
                    if v is not None:
                        if k == "marketCap" and np.isnan(out["market_cap"]):
                            out["market_cap"] = float(v)
                        elif k in out:
                            out[k] = float(v)
            except Exception:
                pass
                
        except Exception as e:
            pass  # Silently handle errors for individual tickers
        return out

    def _bucket(mc: float) -> str:
        """Categorize stocks by market cap"""
        if not isinstance(mc, (int, float)) or np.isnan(mc):
            return "Unknown"
        if mc >= 200e9: return "Mega (≥$200B)"
        if mc >= 50e9:  return "Large ($50–200B)"
        if mc >= 10e9:  return "Mid ($10–50B)"
        if mc >= 2e9:   return "Small ($2–10B)"
        if mc >= 3e8:   return "Micro ($0.3–2B)"
        return "Nano (<$0.3B)"

    def _get_market_cap_filter():
        """Get market cap filter value"""
        if min_market_cap == "Any": return 0
        elif min_market_cap == "Micro ($300M+)": return 3e8
        elif min_market_cap == "Small ($2B+)": return 2e9
        elif min_market_cap == "Mid ($10B+)": return 10e9
        elif min_market_cap == "Large ($50B+)": return 50e9
        return 0

    # Progress bar for data fetching
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    rows = []
    total_tickers = len(ticker_list)
    
    for i, t in enumerate(ticker_list):
        status_text.text(f"Analyzing {t}... ({i+1}/{total_tickers})")
        progress_bar.progress((i + 1) / total_tickers)
        
        try:
            # Fetch price history
            h = fetch_price_history(t, period=time_period, interval="1d")
            if h.empty or len(h) < 2:
                continue
                
            # Calculate momentum
            mom_period = (h["Close"].iloc[-1] / h["Close"].iloc[0]) - 1
            
            # Fetch fundamentals
            f = _ww_fetch_fundamentals(t)
            mcap = f.get("market_cap", np.nan)
            pm = f.get("profitMargins", np.nan)
            roe = f.get("returnOnEquity", np.nan)
            rg = f.get("revenueGrowth", np.nan)
            de = f.get("debtToEquity", np.nan)
            beta = f.get("beta", np.nan)
            volume = f.get("volume", np.nan)
            
            # Apply market cap filter
            if mcap < _get_market_cap_filter():
                continue
            
            # Apply momentum filter
            if mom_period < min_momentum / 100:
                continue
            
            # Calculate health score
            score = np.nan
            if use_health:
                comps, weights = [], []
                if not np.isnan(mom_period): comps.append(mom_period); weights.append(0.25)
                if not np.isnan(pm): comps.append(pm); weights.append(0.25)
                if not np.isnan(roe): comps.append(roe); weights.append(0.25)
                if not np.isnan(rg): comps.append(rg); weights.append(0.25)
                if weights:
                    base_score = sum(c*w for c, w in zip(comps, weights))
                    penalty = (min(max(de, 0.0), 5.0) * 0.10) if not np.isnan(de) else 0.0
                    score = 100.0*(base_score - penalty)
            
            # Apply health score filter
            if use_health and not np.isnan(score) and score < min_health_score:
                continue
            
            # Get ticker description
            description = ticker_descriptions.get(t, 'N/A')
            
            rows.append({
                "Ticker": t,
                "Description": description,
                f"{time_period} %": mom_period,
                "Market Cap": mcap,
                "Cap Bucket": _bucket(mcap),
                "Profit Margin": pm,
                "ROE": roe,
                "Revenue Growth": rg,
                "Debt/Equity": de,
                "Beta": beta,
                "Volume": volume,
                "Health Score": score,
            })
            
        except Exception as e:
            pass  # Silently handle errors for individual tickers
            continue
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Create and display results
    if not rows:
        st.warning("No stocks found matching your criteria. Try adjusting your filters.")
    else:
        dfw = pd.DataFrame(rows)
        
        # Sorting options
        col1, col2 = st.columns([1, 2])
        with col1:
            sort_by = st.selectbox(
                "Sort by",
                ["Health Score", f"{time_period} %", "Market Cap", "ROE", "Revenue Growth"],
                index=0 if use_health else 1
            )
        with col2:
            sort_ascending = st.checkbox("Sort ascending", value=False)
        
        # Sort dataframe
        sort_col = sort_by.replace(" ", "") if sort_by != "Health Score" else "Health Score"
        dfw = dfw.sort_values(sort_col, ascending=sort_ascending).reset_index(drop=True)
        
        # Add external links
        dfw["Yahoo"] = dfw["Ticker"].apply(lambda x: f"https://finance.yahoo.com/quote/{x}")
        dfw["Finviz"] = dfw["Ticker"].apply(lambda x: f"https://finviz.com/quote.ashx?t={x}")
        dfw["Chart"] = dfw["Ticker"].apply(lambda x: f"https://www.tradingview.com/symbols/{x}")
        
        # Display results
        
                st.subheader(f"📈 {scan_type} — {len(df)} result(s)")

                # Add links
                def _yahoo_sym(t):  # Yahoo uses '-' for classes
                    return t.replace('.', '-')
                def _finviz_sym(t): # Finviz uses '.' for classes
                    return t.replace('-', '.')
                df["Yahoo"] = df["Ticker"].apply(lambda t: f"https://finance.yahoo.com/quote/{_yahoo_sym(t)}")
                df["Finviz"] = df["Ticker"].apply(lambda t: f"https://finviz.com/quote.ashx?t={_finviz_sym(t)}")

                try:
                    from streamlit import column_config as cc
                    colcfg = {
                        "Price": cc.NumberColumn(format="%.2f"),
                        "Change %": cc.NumberColumn(format="%.2f%%"),
                        "Volume Ratio": cc.NumberColumn(format="%.2f") if "Volume Ratio" in df.columns else cc.NumberColumn(format="%.2f"),
                        "RSI": cc.NumberColumn(format="%.1f"),
                        "Yahoo": cc.LinkColumn(display_text="Yahoo Finance"),
                        "Finviz": cc.LinkColumn(display_text="Finviz"),
                    }
                    st.dataframe(df, use_container_width=True, hide_index=True, column_config=colcfg)
                except Exception:
                    # Fallback: plain HTML table with clickable links
                    st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)
# Quick chart
                pick = st.selectbox("Quick chart", df["Ticker"].tolist())
                hist = fetch_price_history(pick, period="3mo", interval="1d")
                if not hist.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=hist.index, open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"], name=pick))
                    fig.update_layout(height=420, title=f"{pick} — 3mo daily")
                    st.plotly_chart(fig, use_container_width=True)

                # Export
                st.download_button(
                    "⬇️ Download CSV",
                    df.to_csv(index=False),
                    file_name=f"daily_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )

# =============================
# Sector Tracker
# =============================
elif page == "Sector Tracker":
    st.header("🔄 Sector Rotation Tracker")
    st.caption("Monitor sector performance and identify rotation opportunities")
    
    # Sector ETFs
    sectors = {
        'Technology': 'XLK',
        'Financials': 'XLF', 
        'Healthcare': 'XLV',
        'Energy': 'XLE',
        'Industrials': 'XLI',
        'Consumer Staples': 'XLP',
        'Utilities': 'XLU',
        'Materials': 'XLB',
        'Real Estate': 'XLRE',
        'Communication Services': 'XLC'
    }
    
    # Time period selection
    period = st.selectbox(
        "Time Period", 
        ["1d", "5d", "1mo", "3mo", "6mo"], 
        index=1, 
        key="sector_period",
        help="Time frame for sector performance analysis. Shorter periods show recent momentum, longer periods show trends."
    )
    
    if st.button("📊 Update Sector Analysis", key="update_sectors"):
        with st.spinner("Analyzing sector performance..."):
            sector_data = []
            
            for sector_name, ticker in sectors.items():
                try:
                    # Get sector performance
                    hist = fetch_price_history(ticker, period=period, interval="1d")
                    if hist.empty or len(hist) < 2:
                        continue
                    
                    # Calculate metrics
                    current_price = hist['Close'].iloc[-1]
                    start_price = hist['Close'].iloc[0]
                    total_return = (current_price - start_price) / start_price
                    
                    # Volatility
                    returns = hist['Close'].pct_change().dropna()
                    volatility = returns.std() * np.sqrt(252)
                    
                    # RSI
                    delta = hist['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    current_rsi = rsi.iloc[-1]
                    
                    # Volume trend
                    avg_volume = hist['Volume'].mean()
                    recent_volume = hist['Volume'].tail(5).mean()
                    volume_trend = (recent_volume - avg_volume) / avg_volume
                    
                    sector_data.append({
                        'Sector': sector_name,
                        'Ticker': ticker,
                        'Return %': total_return,
                        'Volatility': volatility,
                        'RSI': current_rsi,
                        'Volume Trend': volume_trend,
                        'Current Price': current_price
                    })
                    
                except Exception as e:
                    continue
            
            if sector_data:
                df_sectors = pd.DataFrame(sector_data)
                df_sectors = df_sectors.sort_values('Return %', ascending=False)
                
                # Display sector performance
                st.subheader(f"📈 Sector Performance ({period})")
                
                # Performance table
                st.dataframe(
                    df_sectors.style.format({
                        'Return %': '{:.2%}',
                        'Volatility': '{:.2%}',
                        'RSI': '{:.1f}',
                        'Volume Trend': '{:.1%}',
                        'Current Price': '{:.2f}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Sector performance chart
                fig = go.Figure()
                colors = ['green' if x > 0 else 'red' for x in df_sectors['Return %']]
                fig.add_trace(go.Bar(
                    x=df_sectors['Sector'],
                    y=df_sectors['Return %'],
                    text=[f"{x:.1%}" for x in df_sectors['Return %']],
                    textposition='auto',
                    marker_color=colors
                ))
                fig.update_layout(
                    title=f"Sector Performance ({period})",
                    xaxis_title="Sector",
                    yaxis_title="Return %",
                    height=500,
                    xaxis_tickangle=-45
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Sector rotation insights
                st.subheader("🔄 Rotation Insights")
                
                # Top performers
                top_sectors = df_sectors.head(3)
                st.markdown("**🔥 Leading Sectors:**")
                for _, sector in top_sectors.iterrows():
                    st.markdown(f"- **{sector['Sector']}** ({sector['Ticker']}): {sector['Return %']:.1%}")
                
                # Laggards
                bottom_sectors = df_sectors.tail(3)
                st.markdown("**📉 Lagging Sectors:**")
                for _, sector in bottom_sectors.iterrows():
                    st.markdown(f"- **{sector['Sector']}** ({sector['Ticker']}): {sector['Return %']:.1%}")
                
                # RSI analysis
                st.subheader("📊 Technical Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🔴 Overbought (RSI > 70):**")
                    overbought = df_sectors[df_sectors['RSI'] > 70]
                    if not overbought.empty:
                        for _, sector in overbought.iterrows():
                            st.markdown(f"- {sector['Sector']}: RSI {sector['RSI']:.1f}")
                    else:
                        st.markdown("- None")
                
                with col2:
                    st.markdown("**🟢 Oversold (RSI < 30):**")
                    oversold = df_sectors[df_sectors['RSI'] < 30]
                    if not oversold.empty:
                        for _, sector in oversold.iterrows():
                            st.markdown(f"- {sector['Sector']}: RSI {sector['RSI']:.1f}")
                    else:
                        st.markdown("- None")
                
                # Correlation matrix
                st.subheader("🔗 Sector Correlations")
                st.info("💡 **Low correlation sectors** can help diversify your portfolio")
                
                # Export data
                csv_data = df_sectors.to_csv(index=False)
                st.download_button(
                    "⬇️ Download Sector Analysis",
                    data=csv_data,
                    file_name=f"sector_analysis_{period}.csv",
                    mime="text/csv"
                )
                
            else:
                st.error("Unable to fetch sector data")
    
    # Sector rotation strategies
    with st.expander("📚 Sector Rotation Strategies", expanded=False):
        st.markdown("""
        **🔄 Economic Cycle Rotation:**
        - **Early Recovery**: Financials, Technology, Consumer Discretionary
        - **Mid-Cycle**: Technology, Industrials, Materials
        - **Late Cycle**: Energy, Materials, Utilities
        - **Recession**: Consumer Staples, Utilities, Healthcare
        
        **📊 Momentum Rotation:**
        - **Momentum**: Invest in sectors showing strongest recent performance
        - **Mean Reversion**: Invest in sectors showing weakest recent performance
        
        **⚡ Quick Rotation Signals:**
        - **Bullish**: Technology, Financials leading with high volume
        - **Bearish**: Utilities, Consumer Staples leading (defensive rotation)
        - **Neutral**: Mixed sector performance, low correlation
        """)

# =============================
# Settings
# =============================
elif page == "Settings":
    st.header("Settings & About")
    
    # App features overview
    st.subheader("📈 App Features")
    st.write(
        '''
        **AI Finance**
        - **Watchlist**: quotes + chart with ticker descriptions.
        - **Portfolio Analysis**: comprehensive performance metrics, risk analysis, and portfolio insights.
        - **Options Analysis**: option chain viewer.
        - **Options Strategy Builder**: advanced options strategy suggestions with detailed analysis.
        - **Weekly Watchlist**: advanced scanner with customizable filters, multiple watchlists, and comprehensive analysis.
        - **AI Next Year Prediction**: advanced prediction with technical and fundamental analysis.
        - **Intrinsic Value**: DCF, EPS Growth, and Dividend Discount models.
        - **Daily Scanner**: momentum, volume, and breakout scanners.
        - **Sector Tracker**: sector rotation analysis and performance tracking.
        - **Pattern Scanner**: advanced technical pattern recognition for winning stock opportunities.
        - **Options Flow**: monitor unusual options activity and institutional flow for trading signals.
        '''
    )
    
    # Financial Terminology Glossary
    st.subheader("📚 Financial Terminology Glossary")
    
    with st.expander("📊 **Technical Analysis Terms**", expanded=False):
        st.markdown("""
        **📈 Moving Averages (MA)**
        - **20MA/50MA**: Average closing prices over 20 or 50 days. Used to identify trends and support/resistance levels.
        - **Golden Cross**: When 50MA crosses above 200MA (bullish signal).
        - **Death Cross**: When 50MA crosses below 200MA (bearish signal).
        
        **📊 RSI (Relative Strength Index)**
        - Momentum oscillator measuring speed and magnitude of price changes (0-100 scale).
        - **Overbought**: RSI > 70 (stock may be overvalued, potential sell signal).
        - **Oversold**: RSI < 30 (stock may be undervalued, potential buy signal).
        
        **📈 Volume Analysis**
        - **Volume Ratio**: Current volume compared to average volume (e.g., 2x = twice normal volume).
        - **Volume Spike**: Unusually high trading volume, often indicating significant price movement ahead.
        - **Breakout**: Price moves above resistance level with high volume (bullish signal).
        
        **📊 Support & Resistance**
        - **Support**: Price level where stock tends to stop falling (buying pressure).
        - **Resistance**: Price level where stock tends to stop rising (selling pressure).
        """)
    
    with st.expander("📈 **Common Ticker Symbols**", expanded=False):
        st.markdown("""
        **🏢 Major ETFs**
        - **SPY**: SPDR S&P 500 ETF - Tracks the S&P 500 index (500 largest US companies)
        - **QQQ**: Invesco QQQ Trust - Tracks NASDAQ-100 (top 100 non-financial NASDAQ stocks)
        - **DIA**: SPDR Dow Jones Industrial Average ETF - Tracks the Dow Jones Industrial Average
        - **IWM**: iShares Russell 2000 ETF - Tracks small-cap US stocks
        
        **🔧 Sector ETFs**
        - **XLK**: Technology Select Sector SPDR Fund - Technology sector ETF
        - **XLF**: Financial Select Sector SPDR Fund - Financial sector ETF
        - **XLV**: Health Care Select Sector SPDR Fund - Healthcare sector ETF
        - **XLE**: Energy Select Sector SPDR Fund - Energy sector ETF
        - **XLI**: Industrial Select Sector SPDR Fund - Industrial sector ETF
        - **XLP**: Consumer Staples Select Sector SPDR Fund - Consumer staples sector ETF
        - **XLU**: Utilities Select Sector SPDR Fund - Utilities sector ETF
        - **XLB**: Materials Select Sector SPDR Fund - Materials sector ETF
        - **XLRE**: Real Estate Select Sector SPDR Fund - Real estate sector ETF
        - **XLC**: Communication Services Select Sector SPDR Fund - Communication services sector ETF
        
        **💻 Major Tech Companies**
        - **AAPL**: Apple Inc. - Consumer electronics, software, and services
        - **MSFT**: Microsoft Corporation - Software, cloud computing, and technology
        - **GOOGL**: Alphabet Inc. (Google) - Internet services, advertising, and technology
        - **AMZN**: Amazon.com Inc. - E-commerce, cloud computing, and digital services
        - **NVDA**: NVIDIA Corporation - Graphics processing units and AI computing
        - **META**: Meta Platforms Inc. - Social media and digital advertising
        - **TSLA**: Tesla Inc. - Electric vehicles, energy storage, and solar panels
        
        **🏦 Financial & Industrial**
        - **JPM**: JPMorgan Chase & Co. - Banking and financial services
        - **V**: Visa Inc. - Payment processing and financial services
        - **MA**: Mastercard Inc. - Payment processing and financial services
        - **BRK-B**: Berkshire Hathaway Inc. - Conglomerate with diverse business holdings
        """)
    
    with st.expander("💰 **Fundamental Analysis Terms**", expanded=False):
        st.markdown("""
        **📊 Valuation Metrics**
        - **P/E Ratio (Price-to-Earnings)**: Stock price divided by earnings per share. Lower = potentially undervalued.
        - **Market Cap**: Total value of company (shares × price). Categories: Mega (>$200B), Large ($10-200B), Mid ($2-10B), Small ($300M-2B), Micro (<$300M).
        - **Intrinsic Value**: True worth of stock based on fundamentals, not market price.
        
        **📈 Financial Ratios**
        - **ROE (Return on Equity)**: Net income ÷ shareholder equity. Higher = better profitability.
        - **Profit Margin**: Net income ÷ revenue. Higher = more efficient operations.
        - **Debt-to-Equity**: Total debt ÷ shareholder equity. Lower = less financial risk.
        - **Revenue Growth**: Year-over-year revenue increase percentage.
        
        **💵 Cash Flow & Dividends**
        - **Free Cash Flow (FCF)**: Cash available after operating expenses and capital expenditures.
        - **Dividend Yield**: Annual dividend ÷ stock price (percentage return from dividends).
        - **Dividend Growth Rate**: Annual increase in dividend payments.
        """)
    
    with st.expander("🎯 **Options Trading Terms**", expanded=False):
        st.markdown("""
        **📋 Basic Options**
        - **Call Option**: Right to buy stock at specific price (strike) by expiration date.
        - **Put Option**: Right to sell stock at specific price (strike) by expiration date.
        - **Strike Price**: Price at which option can be exercised.
        - **Expiration Date**: Last day to exercise the option.
        
        **📊 Options Greeks**
        - **Delta**: How much option price changes for $1 stock price change.
        - **Gamma**: How much delta changes for $1 stock price change.
        - **Theta**: Daily time decay of option value.
        - **Vega**: How much option price changes for 1% volatility change.
        
        **🎯 Options Strategies**
        - **Bull Put Spread**: Sell put at higher strike, buy put at lower strike (bullish).
        - **Bear Call Spread**: Sell call at lower strike, buy call at higher strike (bearish).
        - **Iron Condor**: Sell call spread + sell put spread (neutral, income strategy).
        - **POP (Probability of Profit)**: Likelihood strategy will be profitable at expiration.
        """)
    
    with st.expander("💰 **Premium Collection Strategies**", expanded=False):
        st.markdown("""
        **🎯 Iron Condor Strategy**
        - **Best for**: Low volatility, sideways markets
        - **Setup**: Sell OTM put spread + sell OTM call spread
        - **Advantage**: Collects premium from both sides, profits from time decay
        - **Risk**: Limited but defined, max loss = width - credit received
        - **Ideal Market**: Range-bound stocks with low volatility
        
        **💵 Cash-Secured Put Selling**
        - **Best for**: Stocks you want to own at lower prices
        - **Setup**: Sell puts below current price, collect premium
        - **Advantage**: High probability of profit, potential stock acquisition
        - **Risk**: Stock assignment if price drops below strike
        - **Ideal Market**: Stable, dividend-paying stocks
        
        **📈 Covered Call Writing**
        - **Best for**: Stocks you already own
        - **Setup**: Sell calls against owned shares
        - **Advantage**: Generates income, reduces cost basis
        - **Risk**: Limited upside if stock rallies above strike
        - **Ideal Market**: Sideways or slightly bullish markets
        
        **⏰ Calendar Spreads**
        - **Best for**: Neutral outlook with time decay
        - **Setup**: Sell short-term option, buy longer-term option
        - **Advantage**: Profits from accelerated time decay
        - **Risk**: Limited but defined
        - **Ideal Market**: Low volatility, time-sensitive opportunities
        """)
    
    with st.expander("📊 **Directional Strategies**", expanded=False):
        st.markdown("""
        **📈 Bull Put Spreads**
        - **Best for**: Moderately bullish outlook
        - **Setup**: Sell put at higher strike, buy put at lower strike
        - **Advantage**: Defined risk, collects premium, high probability
        - **Risk**: Limited to spread width minus premium
        - **Ideal Market**: Bullish momentum, support levels
        
        **📉 Bear Call Spreads**
        - **Best for**: Moderately bearish outlook
        - **Setup**: Sell call at lower strike, buy call at higher strike
        - **Advantage**: Defined risk, collects premium, high probability
        - **Risk**: Limited to spread width minus premium
        - **Ideal Market**: Bearish momentum, resistance levels
        
        **🦋 Butterfly Spreads**
        - **Best for**: Neutral outlook, pinpoint price targets
        - **Setup**: Complex 3-leg strategy (buy 1 low strike, sell 2 middle strikes, buy 1 high strike)
        - **Advantage**: High reward if stock stays at target
        - **Risk**: Limited but can be complex
        - **Ideal Market**: Earnings announcements, known catalysts
        
        **🎭 Straddle/Strangle Selling**
        - **Best for**: High volatility, expecting decrease
        - **Setup**: Sell both calls and puts (straddle = same strike, strangle = different strikes)
        - **Advantage**: Collects premium from both sides
        - **Risk**: Unlimited if stock moves significantly
        - **Ideal Market**: High IV, post-earnings, event-driven
        """)
    
    with st.expander("🎯 **Winning Strategy Selection**", expanded=False):
        st.markdown("""
        **📊 High Probability Criteria**
        - **POP > 70%**: Higher probability of profit
        - **ROI > 20%**: Good risk-reward ratio
        - **Theta > 0**: Positive time decay
        - **IV Rank > 50%**: Not overpriced options
        
        **🎯 Market Conditions**
        - **Bullish**: Bull Put Spreads, Cash-Secured Puts
        - **Bearish**: Bear Call Spreads, Put Debit Spreads
        - **Neutral**: Iron Condors, Calendar Spreads
        - **High Vol**: Strangle Selling, Straddle Selling
        - **Low Vol**: Long Straddles, Long Strangles
        
        **⏰ Time Decay Optimization**
        - **30-45 DTE**: Optimal for most strategies
        - **< 7 DTE**: High gamma risk, rapid decay
        - **> 60 DTE**: Lower decay, higher capital requirement
        
        **💰 Risk Management**
        - **Max 2% per trade**: Portfolio risk management
        - **Stop Loss**: 2x credit received
        - **Profit Target**: 50-80% of max profit
        - **Rolling**: Extend time when challenged
        """)
    
    with st.expander("📈 **Market Analysis Terms**", expanded=False):
        st.markdown("""
        **🔄 Sector Rotation**
        - **Economic Cycle**: Different sectors perform better in different economic phases.
        - **Momentum**: Investing in sectors/stocks with recent strong performance.
        - **Mean Reversion**: Investing in sectors/stocks with recent poor performance.
        
        **📊 Risk Metrics**
        - **Volatility**: Standard deviation of returns (measure of price swings).
        - **Sharpe Ratio**: Risk-adjusted return (higher = better risk-adjusted performance).
        - **Maximum Drawdown**: Largest peak-to-trough decline in portfolio value.
        - **Beta**: Stock volatility compared to market (1.0 = market average).
        
        **📈 Market Indicators**
        - **Relative Strength**: Stock performance vs. market index (S&P 500).
        - **Gap**: Price jump between trading sessions (gap up = bullish, gap down = bearish).
        - **Breakout**: Price moves above resistance with volume confirmation.
        """)
    
    with st.expander("💡 **Trading Strategy Terms**", expanded=False):
        st.markdown("""
        **🎯 Position Sizing**
        - **Risk Management**: Limiting potential losses per trade (typically 1-2% of portfolio).
        - **Position Size**: Amount of money invested in single trade.
        - **Stop Loss**: Automatic sell order at predetermined loss level.
        
        **📊 Technical Patterns**
        - **Breakout**: Price moves above resistance level with volume.
        - **Breakdown**: Price moves below support level with volume.
        - **Consolidation**: Sideways price movement in narrow range.
        - **Trend**: Sustained price movement in one direction.
        
        **🔄 Market Timing**
        - **Entry Point**: When to buy (based on technical/fundamental signals).
        - **Exit Point**: When to sell (profit target or stop loss).
        - **Holding Period**: How long to hold position.
        """)
    
    # Risk disclaimer
    st.subheader("⚠️ Important Disclaimers")
    st.warning("""
    **Educational Purpose Only**
    - This app is for educational and informational purposes only.
    - Data provided by Yahoo Finance via `yfinance` (may be delayed).
    - Not investment advice - always do your own research.
    - Past performance does not guarantee future results.
    - Options trading involves substantial risk of loss.
    - Consult with financial advisor before making investment decisions.
    """)
    
    # Data sources
    st.subheader("📊 Data Sources")
    st.info("""
    **Primary Data Provider**: Yahoo Finance
    - Real-time and historical price data
    - Fundamental financial data
    - Options chain data
    - Company information and ratios
    
    **Update Frequency**: 
    - Price data: Real-time (with delays)
    - Fundamental data: Quarterly/annual reports
    - Options data: Real-time during market hours
    """)
