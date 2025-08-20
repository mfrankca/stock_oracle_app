
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

        # Display results
        try:
            from streamlit import column_config as cc
            st.dataframe(
                dfw,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                    "Yahoo": cc.LinkColumn(display_text="Yahoo Finance"),
                    "Finviz": cc.LinkColumn(display_text="Finviz"),
                    "Chart": cc.LinkColumn(display_text="TradingView"),
                },
            )
        except Exception:
            st.markdown(dfw.to_html(escape=False, index=False), unsafe_allow_html=True)

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


# =============================
# Weekly Watchlist
# =============================
elif page == "Weekly Watchlist":
    st.header("📊 Advanced Weekly Watchlist Scanner")
    st.caption("Real‑time(ish) momentum & health scan. Uses 1‑minute data when available; educational only.")

    # Guard: descriptions
    try:
        ticker_descriptions
    except NameError:
        ticker_descriptions = {}

    # --- Controls
    c1, c2, c3 = st.columns([1.5,1,1])
    with c1:
        universe = st.selectbox(
            "Scan Universe",
            ["Curated (ETFs + Mega‑caps)", "Tech Leaders", "Growth Focus", "Custom"],
            help="Choose a group to scan or provide your own list."
        )
    with c2:
        scan_type = st.selectbox(
            "Scan Type",
            ["Momentum (today)", "Momentum (5d)", "Volume Spike", "RSI Extremes", "Breakout Check"],
            help="What to look for in today's action."
        )
    with c3:
        min_mcap = st.selectbox(
            "Min Market Cap",
            ["Any","$2B+","$10B+","$50B+"],
            index=2
        )

    # Custom tickers
    custom_list = []
    if universe == "Custom":
        custom_text = st.text_area("Tickers (one per line)", value="""AAPL
MSFT
NVDA
SPY
QQQ""")
        custom_list = [t.strip().upper() for t in custom_text.splitlines() if t.strip()]

    # Universe definitions
    universes = {
        "Curated (ETFs + Mega‑caps)": ["SPY","QQQ","DIA","IWM","SMH","XLK","XLF","XLV","XLE",
                                       "AAPL","MSFT","NVDA","AMZN","META","TSLA","GOOGL","BRK-B"],
        "Tech Leaders": ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","AMD","NFLX","ADBE","CRM"],
        "Growth Focus": ["NVDA","TSLA","META","SHOP","CRWD","PLTR","SNOW","ROKU","SQ","ZM","COIN"],
        "Custom": custom_list
    }
    tickers = [t for t in universes[universe] if t]

    # Market‑cap filter helper (uses yfinance info; fallback allows all)
    import yfinance as yf
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    def _passes_mcap(t):
        try:
            info = yf.Ticker(t).fast_info
            mc = getattr(info, "market_cap", None)
        except Exception:
            mc = None
        thr = {"Any":0, "$2B+":2_000_000_000, "$10B+":10_000_000_000, "$50B+":50_000_000_000}[min_mcap]
        return True if (mc is None or mc >= thr) else False

    @st.cache_data(ttl=45)
    def _intraday(t):
        try:
            df = yf.Ticker(t).history(period="1d", interval="1m", auto_adjust=False)
            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index)
                return df
        except Exception:
            pass
        return pd.DataFrame()

    @st.cache_data(ttl=120)
    def _daily(t, period="3mo", interval="1d"):
        try:
            df = yf.Ticker(t).history(period=period, interval=interval, auto_adjust=False)
            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index)
                return df
        except Exception:
            pass
        return pd.DataFrame()

    def _rsi(series, window=14):
        if series is None or series.empty:
            return np.nan
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.dropna().iloc[-1]) if not rsi.dropna().empty else np.nan

    st.markdown("---")
    r1, r2, r3 = st.columns([1,1,2])
    with r1:
        run = st.button("▶ Run Weekly Scan", type="primary", key="run_weekly")
    with r2:
        refresh = st.button("🔄 Refresh now", key="refresh_weekly")
    with r3:
        st.write(f"Last updated: **{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**")

    if run or refresh:
        with st.spinner("Scanning..."):
            rows = []
            for t in tickers:
                if not _passes_mcap(t):
                    continue
                try:
                    intr = _intraday(t)
                    hist5d = _daily(t, period="5d", interval="1d")
                    hist60 = _daily(t, period="3mo", interval="1d")

                    # price now
                    price_now = np.nan
                    try:
                        fi = getattr(yf.Ticker(t), "fast_info", None)
                        if fi and getattr(fi, "last_price", None) is not None:
                            price_now = float(fi.last_price)
                    except Exception:
                        pass
                    if np.isnan(price_now):
                        if not intr.empty:
                            price_now = float(intr["Close"].iloc[-1])
                        elif not hist5d.empty:
                            price_now = float(hist5d["Close"].iloc[-1])
                        else:
                            continue

                    # Today momentum vs prev close
                    if not hist5d.empty and len(hist5d) >= 2:
                        prev_close = float(hist5d["Close"].iloc[-2])
                        mom_today = (price_now / prev_close - 1.0) if prev_close else np.nan
                    else:
                        mom_today = np.nan

                    # Volume ratio
                    if not intr.empty:
                        vol_today = float(intr["Volume"].sum())
                    else:
                        vol_today = float(hist5d["Volume"].iloc[-1]) if not hist5d.empty else np.nan
                    vol_avg5 = float(hist5d["Volume"].iloc[:-1].tail(5).mean()) if not hist5d.empty else np.nan
                    vol_ratio = (vol_today / vol_avg5) if (vol_avg5 and vol_avg5==vol_avg5 and vol_avg5>0) else np.nan

                    # 5d momentum
                    if not hist5d.empty and len(hist5d) >= 2:
                        mom_5d = float(hist5d["Close"].iloc[-1] / hist5d["Close"].iloc[0] - 1.0)
                    else:
                        mom_5d = np.nan

                    # RSI
                    px = intr["Close"] if not intr.empty else (hist5d["Close"] if not hist5d.empty else pd.Series(dtype=float))
                    rsi = _rsi(px, 14)

                    # MAs
                    ma20 = float(hist60["Close"].rolling(20).mean().iloc[-1]) if not hist60.empty and len(hist60)>=20 else np.nan
                    ma50 = float(hist60["Close"].rolling(50).mean().iloc[-1]) if not hist60.empty and len(hist60)>=50 else np.nan

                    score_map = {
                        "Momentum (today)": mom_today,
                        "Momentum (5d)": mom_5d,
                        "Volume Spike": vol_ratio,
                        "RSI Extremes": rsi,
                        "Breakout Check": (1.0 if (price_now>ma20 and price_now>ma50 and ma20==ma20 and ma50==ma50) else 0.0),
                    }
                    score = score_map.get(scan_type, np.nan)

                    rows.append({
                        "Ticker": t,
                        "Price": price_now,
                        "Today %": mom_today,
                        "5d %": mom_5d,
                        "Vol Ratio": vol_ratio,
                        "RSI": rsi,
                        "Above 20/50": "Yes" if (price_now>ma20 and price_now>ma50 and ma20==ma20 and ma50==ma50) else "No",
                        "Score": score,
                    })
                except Exception:
                    continue

            if not rows:
                st.warning("No results. Try a different universe or loosen filters.")
            else:
                df = pd.DataFrame(rows)

                # Links
                def _yahoo_sym(t):  # Yahoo uses '-' for classes
                    return t.replace('.', '-')
                def _finviz_sym(t): # Finviz uses '.' for classes
                    return t.replace('-', '.')
                df["Yahoo"] = df["Ticker"].apply(lambda t: f"https://finance.yahoo.com/quote/{_yahoo_sym(t)}")
                df["Finviz"] = df["Ticker"].apply(lambda t: f"https://finviz.com/quote.ashx?t={_finviz_sym(t)}")

                # Rank per scan type
                if scan_type in ["Momentum (today)","Momentum (5d)"]:
                    df = df.sort_values("Score", ascending=False)
                elif scan_type == "Volume Spike":
                    df = df.sort_values("Vol Ratio", ascending=False)
                elif scan_type == "RSI Extremes":
                    df["Score"] = (50 - (df["RSI"] - 50).abs())
                    df = df.sort_values("Score")
                elif scan_type == "Breakout Check":
                    df = df.sort_values(["Above 20/50","5d %"], ascending=[False, False])

                st.subheader(f"Results – {len(df)} tickers")

                try:
                    from streamlit import column_config as cc
                    colcfg = {
                        "Price": cc.NumberColumn(format="%.2f"),
                        "Today %": cc.NumberColumn(format="%.2f%%"),
                        "5d %": cc.NumberColumn(format="%.2f%%"),
                        "Vol Ratio": cc.NumberColumn(format="%.2f"),
                        "RSI": cc.NumberColumn(format="%.1f"),
                        "Yahoo": cc.LinkColumn(display_text="Yahoo Finance"),
                        "Finviz": cc.LinkColumn(display_text="Finviz"),
                    }
                    st.dataframe(df, use_container_width=True, hide_index=True, column_config=colcfg)
                except Exception:
                    st.dataframe(df, use_container_width=True, hide_index=True)

                # Quick chart
                pick = st.selectbox("Quick chart", df["Ticker"].tolist())
                hist = _daily(pick, period="3mo", interval="1d")
                if not hist.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=hist.index, open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"], name=pick))
                    fig.update_layout(height=420, title=f"{pick} – 3mo daily")
                    st.plotly_chart(fig, use_container_width=True)

                # Export
                st.download_button(
                    "⬇️ Download CSV",
                    df.to_csv(index=False),
                    file_name=f"weekly_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )



# =============================
# Daily Scanner
# =============================
elif page == "Daily Scanner":
    st.header("📊 Daily Momentum Scanner")
    st.caption("Real‑time(ish) intraday scanner using 1‑minute data and daily context. Educational use only.")

    # --------- Universe ---------
    watchlists = {
        "SPY + QQQ + DIA + IWM": ["SPY","QQQ","DIA","IWM"],
        "Tech Leaders": ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","AVGO","AMD","ADBE","CRM","NFLX"],
        "Large Caps": ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","BRK-B","JPM","V","MA","JNJ","UNH","PG","XOM"],
        "Growth Focus": ["SHOP","CRWD","PLTR","SNOW","ROKU","SQ","ZM","COIN","NET","DDOG","U"],
        "Custom": []
    }

    colU, colS, colC = st.columns([1.4,1,1])
    with colU:
        selected_watchlist = st.selectbox("Scan Universe", list(watchlists.keys()), key="scan_universe_daily")
    with colS:
        scan_type = st.selectbox(
            "Scanner Type",
            ["Momentum Movers", "Volume Spikes", "Breakout Candidates", "Gap Up/Down", "RSI Extremes"],
            help="Pick the condition to screen for."
        )
    with colC:
        min_mcap_label = st.selectbox("Min Market Cap", ["Any","$2B+","$10B+","$50B+"], index=2)

    if selected_watchlist == "Custom":
        custom_text = st.text_area(
            "Tickers (one per line)",
            value="""AAPL
MSFT
NVDA
SPY
QQQ"""
        )
        watchlists["Custom"] = [t.strip().upper() for t in custom_text.splitlines() if t.strip()]

    tickers_to_scan = watchlists[selected_watchlist]

    # --------- Controls ---------
    colA, colB, colC2, colD = st.columns([1,1,1,2])
    with colA:
        run = st.button("🔍 Run Scanner", key="run_scanner_daily", type="primary")
    with colB:
        refresh = st.button("🔄 Refresh", key="refresh_daily")
    with colC2:
        loosen = st.toggle("Loosen filters", value=False, help="Relax thresholds if you get empty results.")
    with colD:
        st.write(f"Last updated: **{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**")

    # Thresholds
    vol_thr = 1.5 if loosen else 2.0
    mom_thr = 0.03 if loosen else 0.05      # Momentum Movers threshold
    brk_mom = 0.01 if loosen else 0.02      # Breakout minimal momentum
    gap_thr = 0.02 if loosen else 0.03      # Gap magnitude
    rsi_hi, rsi_lo = 70, 30

    # --------- Helpers ---------
    import yfinance as yf
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    @st.cache_data(ttl=60)
    def _intraday(t):
        try:
            df = yf.Ticker(t).history(period="1d", interval="1m", auto_adjust=False)
            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index)
                return df
        except Exception:
            pass
        return pd.DataFrame()

    @st.cache_data(ttl=120)
    def _daily_3mo(t):
        try:
            df = yf.Ticker(t).history(period="3mo", interval="1d", auto_adjust=False)
            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index)
                return df
        except Exception:
            pass
        return pd.DataFrame()

    def _fast_mcap(tk: yf.Ticker):
        try:
            fi = getattr(tk, "fast_info", None)
            return getattr(fi, "market_cap", None) if fi else None
        except Exception:
            return None

    min_cap_value = {"Any":0, "$2B+":2_000_000_000, "$10B+":10_000_000_000, "$50B+":50_000_000_000}[min_mcap_label]

    if run or refresh:
        with st.spinner("Scanning for opportunities..."):
            results = []

            for ticker in tickers_to_scan:
                try:
                    tk = yf.Ticker(ticker)

                    # Market cap (do not exclude if missing)
                    mc = _fast_mcap(tk)
                    if mc is not None and mc < min_cap_value:
                        continue

                    # Data
                    d3 = _daily_3mo(ticker)
                    if d3.empty or len(d3) < 5:
                        continue

                    intr = _intraday(ticker)

                    # Price and change vs prior close
                    if not d3.empty and len(d3) >= 2:
                        prev_close = float(d3["Close"].iloc[-2])
                    else:
                        prev_close = np.nan

                    if not intr.empty:
                        last_price = float(intr["Close"].iloc[-1])
                        today_open = float(intr["Open"].iloc[0])
                        todays_volume = float(intr["Volume"].sum())
                    else:
                        last_price = float(d3["Close"].iloc[-1])
                        todays_volume = float(d3["Volume"].iloc[-1]) if "Volume" in d3 else np.nan
                        today_open = float(d3["Open"].iloc[-1]) if "Open" in d3 else last_price

                    if prev_close and not np.isnan(prev_close) and prev_close != 0:
                        price_change = (last_price / prev_close) - 1.0
                    else:
                        price_change = np.nan

                    # Baseline volume: prior 10 days, excluding today
                    vol_window = d3["Volume"].iloc[:-1].tail(10) if "Volume" in d3 else pd.Series(dtype=float)
                    avg_volume = float(vol_window.mean()) if not vol_window.empty else np.nan
                    volume_ratio = (todays_volume / avg_volume) if (avg_volume and avg_volume > 0) else np.nan

                    # Moving averages
                    ma20 = float(d3["Close"].rolling(20).mean().iloc[-1]) if len(d3) >= 20 else np.nan
                    ma50 = float(d3["Close"].rolling(50).mean().iloc[-1]) if len(d3) >= 50 else np.nan

                    # RSI(14) on daily closes
                    delta = d3["Close"].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    current_rsi = float(rsi.iloc[-1]) if not rsi.dropna().empty else np.nan

                    # Conditions
                    include = False
                    if scan_type == "Momentum Movers" and not np.isnan(price_change) and price_change > mom_thr:
                        include = True
                    elif scan_type == "Volume Spikes" and not np.isnan(volume_ratio) and volume_ratio > vol_thr:
                        include = True
                    elif scan_type == "Breakout Candidates" and (not np.isnan(ma20)) and last_price > ma20 and price_change > brk_mom:
                        include = True
                    elif scan_type == "Gap Up/Down" and not np.isnan(price_change) and abs(price_change) > gap_thr:
                        include = True
                    elif scan_type == "RSI Extremes" and (not np.isnan(current_rsi)) and (current_rsi >= rsi_hi or current_rsi <= rsi_lo):
                        include = True

                    if include:
                        # Optional sector/industry (avoid blocking on failures)
                        sector = "N/A"
                        industry = "N/A"
                        try:
                            info = tk.info
                            if isinstance(info, dict):
                                sector = info.get("sector", "N/A")
                                industry = info.get("industry", "N/A")
                        except Exception:
                            pass

                        results.append({
                            "Ticker": ticker,
                            "Price": last_price,
                            "Change %": price_change,
                            "Volume Ratio": volume_ratio,
                            "RSI": current_rsi,
                            "Above 20/50": "Yes" if (not np.isnan(ma20) and not np.isnan(ma50) and last_price > ma20 and last_price > ma50) else "No",
                            "Sector": sector,
                            "Industry": industry,
                            "Market Cap": mc if mc is not None else 0
                        })
                except Exception:
                    continue

            if not results:
                st.warning("No matches with the current thresholds. Try **Loosen filters**, switch universe, or hit **Refresh** during market hours.")
            else:
                df = pd.DataFrame(results)

                # Add links
                def _yahoo_sym(t):  # Yahoo uses '-' for classes
                    return t.replace('.', '-')
                def _finviz_sym(t): # Finviz uses '.' for classes
                    return t.replace('-', '.')
                df["Yahoo"] = df["Ticker"].apply(lambda t: f"https://finance.yahoo.com/quote/{_yahoo_sym(t)}")
                df["Finviz"] = df["Ticker"].apply(lambda t: f"https://finviz.com/quote.ashx?t={_finviz_sym(t)}")

                # Sorting heuristic by scan type
                if scan_type == "Momentum Movers":
                    df = df.sort_values("Change %", ascending=False)
                elif scan_type == "Volume Spikes":
                    df = df.sort_values("Volume Ratio", ascending=False)
                elif scan_type == "Breakout Candidates":
                    df = df.sort_values(["Above 20/50","Change %"], ascending=[False, False])
                elif scan_type == "Gap Up/Down":
                    df["Abs Change %"] = df["Change %"].abs()
                    df = df.sort_values("Abs Change %", ascending=False)
                elif scan_type == "RSI Extremes":
                    df["RSI Dist"] = (df["RSI"] - 50).abs()
                    df = df.sort_values("RSI Dist", ascending=False)

                st.subheader(f"📈 {scan_type} — {len(df)} result(s)")

                try:
                    from streamlit import column_config as cc
                    colcfg = {
                        "Price": cc.NumberColumn(format="%.2f"),
                        "Change %": cc.NumberColumn(format="%.2f%%"),
                        "Volume Ratio": cc.NumberColumn(format="%.2f"),
                        "RSI": cc.NumberColumn(format="%.1f"),
                        "Yahoo": cc.LinkColumn(display_text="Yahoo Finance"),
                        "Finviz": cc.LinkColumn(display_text="Finviz"),
                    }
                    st.dataframe(df, use_container_width=True, hide_index=True, column_config=colcfg)
                except Exception:
                    st.dataframe(df, use_container_width=True, hide_index=True)

                # Quick chart
                pick = st.selectbox("Quick chart", df["Ticker"].tolist())
                hist = _daily_3mo(pick)
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


