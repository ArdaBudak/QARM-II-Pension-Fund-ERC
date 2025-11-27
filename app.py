import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import cvxpy as cp
import streamlit.components.v1 as components
import base64
from fpdf import FPDF
import io
from scipy.optimize import minimize_scalar
from datetime import datetime, timedelta
from pandas.tseries.offsets import MonthEnd
from sklearn.covariance import LedoitWolf

# Custom styling
st.set_page_config(page_title="Pension Fund Optimizer", layout="wide")

# --- THEME CONFIGURATION ---
BUTTON_COLOR = "#E0E0E0"    # Light Grey
BUTTON_TEXT = "#000000"     # Black Text
LIGHT_BG = "#FFFFFF"        # Main Background
SIDEBAR_BG = "#F5F5F5"      # Light Grey Sidebar
TEXT_COLOR = "#000000"      # Black Text
TAB_UNDERLINE = "#999999"   # Dark Grey for Tabs

# --- IMAGE HELPERS ---
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return ""

banner_base64 = get_base64_of_bin_file("Gray-Manhattan-Morning-Wallpaper-Mural.jpg")
logo_base64 = get_base64_of_bin_file("ERC Portfolio.png")

# --- CSS STYLING ---
st.markdown(
    f"""
    <style>
    :root {{
        --primary-color: {BUTTON_COLOR};
        --background-color: {LIGHT_BG};
        --secondary-background-color: {SIDEBAR_BG};
        --text-color: {TEXT_COLOR};
        --font: 'Times New Roman', serif;
    }}
    
    .stApp {{
        background-color: {LIGHT_BG};
        color: {TEXT_COLOR};
        font-family: 'Times New Roman', serif;
    }}
    
    header {{
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        background-image: url("data:image/jpg;base64,{banner_base64}") !important;
        background-size: cover !important;        
        background-position: center 45% !important; 
        background-repeat: no-repeat !important;
        height: 8rem !important;                  
        z-index: 1001 !important;
        background-color: #FFFFFF !important;
        border-bottom: 1px solid #ccc;
    }}
    
    header::after {{
        content: "";
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 100%;
        max-width: 300px;
        height: 80%;
        background-image: url("data:image/png;base64,{logo_base64}");
        background-size: contain;
        background-repeat: no-repeat;
        background-position: center;
        z-index: 1002;
        pointer-events: none;
    }}
    
    header .decoration {{ display: none; }}
    
    .block-container {{
        padding-top: 9rem !important; 
        padding-bottom: 1rem !important;
    }}
    
    [data-testid="stAppViewContainer"] {{
        overflow-x: hidden;
        overflow-y: auto;
    }}
    
    div[data-baseweb="tab-list"] {{
        position: -webkit-sticky !important;
        position: sticky !important;
        top: 0 !important;
        z-index: 999 !important;
        background-color: {LIGHT_BG} !important;
        padding-top: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #E0E0E0;
        box-shadow: 0 4px 4px -2px rgba(0,0,0,0.05);
    }}

    div[data-baseweb="tab-highlight"] {{
        background-color: {TAB_UNDERLINE} !important;
    }}
    div[data-baseweb="tab-list"] button {{
        font-family: 'Times New Roman', serif !important;
        font-weight: bold !important;
    }}

    .stSidebar {{ background-color: {SIDEBAR_BG}; }}
    section[data-testid="stSidebar"] {{ background-color: {SIDEBAR_BG}; color: {TEXT_COLOR}; }}

    .stButton>button {{ 
        background-color: {BUTTON_COLOR}; 
        color: {BUTTON_TEXT}; 
        border-radius: 8px; 
        padding: 10px 24px; 
        font-family: 'Times New Roman', serif; 
        border: 1px solid #CCCCCC;
        font-weight: bold;
        transition: all 0.3s ease;
    }}
    .stButton>button:hover {{ 
        background-color: #D5D5D5; 
        border-color: #999999;
    }}

    span[data-baseweb="tag"] {{
        background-color: #E8E8E8 !important;
        color: {TEXT_COLOR} !important;
        border: 1px solid #d0d0d0;
    }}

    h1, h2, h3, h4, h5, h6, .stHeader, p, label, span, div {{ 
        color: {TEXT_COLOR} !important; 
        font-family: 'Times New Roman', serif; 
    }}
    
    @media print {{
        section[data-testid="stSidebar"], 
        .stButton, 
        iframe, 
        .vfrc-widget--chat,
        header, 
        div[data-baseweb="tab-list"] {{
            display: none !important;
        }}
        .block-container {{
            padding-top: 0 !important;
            margin: 0 !important;
        }}
        .stApp {{
            background-color: white !important;
        }}
        .js-plotly-plot {{
            break-inside: avoid;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- DATA LOADING ---
@st.cache_data
def load_data_bundle():
    returns_wide = pd.DataFrame()
    rf_series = pd.Series(dtype=float)
    tx_cost_series = pd.Series(dtype=float)

    try:
        comp = pd.read_parquet("compustat_git.parquet")
        etf = pd.read_parquet("etf_git.parquet")

        if "RF" in comp.columns:
            comp["date"] = pd.to_datetime(comp["date"])
            rf_raw = comp.groupby("date")["RF"].mean().sort_index()
            rf_series = rf_raw.fillna(0.0)

        comp_ret = comp[["date", "company_name", "monthly_return"]].copy()
        comp_ret["date"] = pd.to_datetime(comp_ret["date"])
        comp_ret = comp_ret.rename(columns={"company_name": "asset", "monthly_return": "ret"})

        etf_ret = etf[["date", "ETF", "return_monthly"]].copy()
        etf_ret["date"] = pd.to_datetime(etf_ret["date"])
        etf_ret = etf_ret.rename(columns={"ETF": "asset", "return_monthly": "ret"})

        returns_long = pd.concat([comp_ret, etf_ret], ignore_index=True)
        returns_wide = returns_long.pivot(index="date", columns="asset", values="ret").sort_index()
        returns_wide.index = pd.to_datetime(returns_wide.index)

    except Exception as e:
        st.error(f"CRITICAL: Error loading market data: {e}")
        return pd.DataFrame(), pd.Series(), pd.Series()

    try:
        tx_file = pd.read_parquet("OW_tx_costs.parquet")
        if "date" in tx_file.columns and "OW_tx_cost" in tx_file.columns:
            tx_file["date"] = pd.to_datetime(tx_file["date"])
            tx_cost_series = tx_file.set_index("date")["OW_tx_cost"].sort_index()
    except Exception as e:
        st.warning("Using default 10bps transaction costs.")
            
    return returns_wide, rf_series, tx_cost_series

@st.cache_data
def load_country_mapping():
    try:
        comp = pd.read_parquet("compustat_git.parquet")
        if "country_code" in comp.columns:
            mapping = comp[["company_name", "country_code"]].drop_duplicates()
            return mapping.set_index("company_name")["country_code"].to_dict()
    except:
        pass
    return {}

def get_valid_assets(custom_data, start_date, end_date):
    start_date = pd.to_datetime(start_date) + MonthEnd(0)
    end_date = pd.to_datetime(end_date) + MonthEnd(0)
    
    if custom_data.empty: 
        return {"stocks": [], "etfs": []}

    subset = custom_data.loc[start_date:end_date]
    available_assets = set(subset.columns[subset.notna().any()].tolist())
    
    try:
        comp = pd.read_parquet("compustat_git.parquet")
        all_stocks = set(comp["company_name"].unique())
        etf = pd.read_parquet("etf_git.parquet")
        all_etfs = set(etf["ETF"].unique())
        
        valid_stocks = sorted(list(available_assets.intersection(all_stocks)))
        valid_etfs = sorted(list(available_assets.intersection(all_etfs)))
        
        return {"stocks": valid_stocks, "etfs": valid_etfs}
    except:
        return {"stocks": sorted(list(available_assets)), "etfs": []}

def get_common_start_date(custom_data, selected_assets, user_start_date):
    user_start_date = pd.to_datetime(user_start_date) + MonthEnd(0)
    first_valid_series = custom_data[selected_assets].apply(lambda col: col.first_valid_index())
    overall_first_valid = first_valid_series.min()
    
    if pd.isna(overall_first_valid):
         st.error("No valid data found for any selected asset.")
         return None

    if overall_first_valid > user_start_date:
        st.warning(f"⚠️ No data available at start date. Optimization will begin on **{overall_first_valid.date()}**.")
        return overall_first_valid
        
    return user_start_date

def compute_rebalance_indices(dates, freq_label):
    freq_map = {"Quarterly": 3, "Semi-Annually": 6, "Annually": 12}
    step = freq_map.get(freq_label, 12)
    n = len(dates)
    idxs = list(range(0, n, step))
    if idxs[-1] != n - 1:
        idxs.append(n - 1)
    return idxs

# --- OPTIMIZATION ---

def solve_erc_weights(cov_matrix):
    n = cov_matrix.shape[0]
    y = cp.Variable(n)
    objective = cp.Minimize(0.5 * cp.quad_form(y, cov_matrix) - cp.sum(cp.log(y)))
    constraints = [y >= 1e-8] 
    prob = cp.Problem(objective, constraints)
    
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            prob.solve(solver=cp.SCS, verbose=False)

        if prob.value is None or y.value is None:
            return None
        y_val = np.array(y.value).flatten()
        w_star = y_val / np.sum(y_val)
        return w_star
    except Exception as e:
        return None

def compute_max_drawdown(cumulative_returns):
    running_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - running_max) / running_max
    return drawdowns.min() * 100

@st.cache_data(show_spinner=True)
def perform_optimization(selected_assets, start_date_user, end_date_user, rebalance_freq, _custom_data, _rf_data, _tx_cost_data, lookback_months=36, ann_factor=12, _version=10):
    custom_data = _custom_data 
    rf_data = _rf_data
    tx_cost_data = _tx_cost_data
    country_map = load_country_mapping()
    
    try:
        start_date_user = pd.to_datetime(start_date_user) + MonthEnd(0)
        end_date_user = pd.to_datetime(end_date_user) + MonthEnd(0)
        common_start = get_common_start_date(custom_data, selected_assets, start_date_user)
        if common_start is None: return None
        
        first_rebalance_date = common_start + pd.DateOffset(months=lookback_months) + MonthEnd(0)
        if first_rebalance_date > end_date_user:
            st.error(f"Not enough data for lookback. Need data until {first_rebalance_date.date()}")
            return None
            
        full_returns = custom_data[selected_assets].sort_index().loc[common_start:end_date_user]
        period_returns = full_returns.loc[first_rebalance_date:end_date_user]
        
        if period_returns.empty: return None
            
        period_dates = period_returns.index
        rebalance_indices = compute_rebalance_indices(period_dates, rebalance_freq)
        
        n = len(selected_assets)
        previous_weights = np.zeros(n)
        port_returns = pd.Series(index=period_dates, dtype=float).fillna(0.0)
        weights_over_time = {}
        rc_over_time = {} 
        country_exposure_over_time = {}
        total_tc = 0.0
        rc_pct = np.zeros(n) 

        for j, reb_idx in enumerate(rebalance_indices):
            rebal_date = period_dates[reb_idx]
            global_reb_pos = full_returns.index.get_loc(rebal_date)
            start_pos = max(0, global_reb_pos - lookback_months)
            
            est_window = full_returns.iloc[start_pos:global_reb_pos]
            est_window_clean = est_window.dropna(axis=1, how='any')
            valid_assets = est_window_clean.columns.tolist()
            
            current_weights = np.zeros(n)
            current_rc = np.zeros(n)
            
            if len(valid_assets) > 0:
                try:
                    if len(valid_assets) == 1:
                         w_active = np.array([1.0])
                         rc_active = np.array([100.0]) 
                    else:
                         lw = LedoitWolf().fit(est_window_clean.values)
                         cov = lw.covariance_ * ann_factor
                         w_active = solve_erc_weights(cov)
                         if w_active is None: raise ValueError("Solver failed")
                         
                         port_var = w_active @ cov @ w_active
                         sigma_p = np.sqrt(port_var)
                         mrc = cov @ w_active
                         rc_abs = w_active * mrc / sigma_p
                         rc_active = (rc_abs / np.sum(rc_abs)) * 100
                    
                    for asset_name, w_val, rc_val in zip(valid_assets, w_active, rc_active):
                        idx = selected_assets.index(asset_name)
                        current_weights[idx] = w_val
                        current_rc[idx] = rc_val
                except:
                    # Inverse Volatility Fallback
                    try:
                        vols = est_window_clean.std()
                        inv_vols = 1.0 / vols
                        w_active = inv_vols / inv_vols.sum()
                        for asset_name, w_val in zip(valid_assets, w_active.values):
                            idx = selected_assets.index(asset_name)
                            current_weights[idx] = w_val
                            current_rc[idx] = 100.0 / len(valid_assets)
                    except:
                         if np.sum(previous_weights) > 0.9: current_weights = previous_weights

            rc_over_time[rebal_date] = current_rc
            rc_pct = current_rc

            if not tx_cost_data.empty:
                try:
                    if not tx_cost_data.index.is_monotonic_increasing: tx_cost_data = tx_cost_data.sort_index()
                    current_tx_rate = tx_cost_data.asof(rebal_date)
                    if pd.isna(current_tx_rate): current_tx_rate = 0.0010
                except: current_tx_rate = 0.0010
            else: current_tx_rate = 0.0010 

            traded_volume = np.sum(np.abs(current_weights - previous_weights))
            cost = traded_volume * current_tx_rate
            total_tc += cost
            
            previous_weights = current_weights.copy()
            weights_over_time[rebal_date] = current_weights
            
            country_exp = {}
            for asset, w in zip(selected_assets, current_weights):
                c = country_map.get(asset, "Unknown")
                country_exp[c] = country_exp.get(c, 0) + w
            country_exposure_over_time[rebal_date] = country_exp

            if j == len(rebalance_indices) - 1: end_slice = len(period_dates)
            else: end_slice = rebalance_indices[j+1]
                
            sub_ret = period_returns.iloc[reb_idx:end_slice].fillna(0.0)
            if not sub_ret.empty:
                period_port_ret = sub_ret.values @ current_weights
                if len(period_port_ret) > 0: period_port_ret[0] -= cost 
                port_returns.iloc[reb_idx:end_slice] = period_port_ret

        if not rf_data.empty:
            aligned_rf = rf_data.reindex(port_returns.index, method='ffill').fillna(0.0)
            port_excess_returns = port_returns - aligned_rf
        else:
            port_excess_returns = port_returns
            
        benchmark_asset = "SPDR S&P 500 ETF"
        cum_benchmark = pd.Series(dtype=float) 
        if benchmark_asset in custom_data.columns:
             bench_ret = custom_data[benchmark_asset].reindex(port_returns.index).fillna(0.0)
             if not rf_data.empty:
                 aligned_rf_bench = rf_data.reindex(port_returns.index, method='ffill').fillna(0.0)
                 bench_excess = bench_ret - aligned_rf_bench
             else: bench_excess = bench_ret
             cum_benchmark = (1 + bench_excess).cumprod()

        ann_vol = port_returns.std() * np.sqrt(ann_factor)
        ann_excess_ret = port_excess_returns.mean() * ann_factor
        sharpe = ann_excess_ret / ann_vol if ann_vol > 0 else 0.0
        cum_port_excess = (1 + port_excess_returns).cumprod()
        max_drawdown = compute_max_drawdown(cum_port_excess)

        return {
            "selected_assets": selected_assets,
            "weights": current_weights,
            "risk_contrib_pct": rc_pct,
            "expected_return": ann_excess_ret * 100, 
            "volatility": ann_vol * 100,             
            "sharpe": sharpe,
            "port_returns": port_excess_returns,
            "cum_port": cum_port_excess,
            "cum_benchmark": cum_benchmark,
            "total_tc": total_tc * 100,
            "weights_df": pd.DataFrame(weights_over_time, index=selected_assets).T.sort_index(),
            "rc_df": pd.DataFrame(rc_over_time, index=selected_assets).T.sort_index(),
            "corr_matrix": est_window_clean.corr() if 'est_window_clean' in locals() else pd.DataFrame(),
            "country_exposure_over_time": country_exposure_over_time,
            "max_drawdown": max_drawdown,
            # Added for SOTA Monte Carlo
            "hist_data": est_window_clean if 'est_window_clean' in locals() else pd.DataFrame()
        }
    except Exception as e:
        st.error(f"Optimization Error: {e}")
        return None

# --- SOTA MONTE CARLO (HISTORICAL BOOTSTRAP) ---
@st.cache_data
def run_monte_carlo(hist_returns_df, weights, years=10, simulations=1000, initial_capital=100000):
    """
    State-of-the-Art Monte Carlo: Multivariate Historical Bootstrap.
    Instead of assuming a normal distribution (GBM), we sample from REAL historical
    vectors. This preserves:
    1. Cross-Asset Correlations (Asset A vs Asset B)
    2. Fat Tails (Real market crashes)
    """
    if hist_returns_df.empty:
        return [], [], [], [], []
        
    # 1. Calculate Portfolio Historical Returns
    port_hist_returns = hist_returns_df.values @ weights
    
    n_steps = int(years * 12) # Monthly steps
    
    # 2. Bootstrap Engine (Sample from history with replacement)
    random_indices = np.random.choice(len(port_hist_returns), size=(simulations, n_steps))
    
    # 3. Construct Paths
    simulated_returns = port_hist_returns[random_indices]
    
    # Growth factors: (1 + r)
    growth_factors = 1 + simulated_returns
    
    # Accumulate
    cumulative_growth = np.cumprod(growth_factors, axis=1)
    
    # Scale by capital
    price_paths = initial_capital * np.hstack([np.ones((simulations, 1)), cumulative_growth])
    
    # 4. Statistics
    dates = [datetime.now() + timedelta(days=30*i) for i in range(n_steps + 1)]
    median_path = np.median(price_paths, axis=0)
    p95 = np.percentile(price_paths, 95, axis=0) # Bull case
    p05 = np.percentile(price_paths, 5, axis=0)  # Bear case (Tail Risk)
    
    return dates, median_path, p95, p05, price_paths

def plot_monte_carlo(dates, median, p95, p05):
    fig = go.Figure()
    
    # Fan Chart "Cone of Uncertainty"
    fig.add_trace(go.Scatter(
        x=dates, y=p95, mode='lines', 
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=p05, mode='lines', 
        line=dict(width=0), fill='tonexty', 
        fillcolor='rgba(94, 106, 210, 0.2)', 
        name='95% Confidence Interval'
    ))
    
    # Median Path
    fig.add_trace(go.Scatter(
        x=dates, y=median, 
        mode='lines', 
        line=dict(color='#5e6ad2', width=3), 
        name='Median Projection'
    ))
    
    fig.update_layout(
        title="10-Year Monte Carlo Projection (Historical Bootstrap)",
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(color="black", family="Times New Roman"),
        yaxis_title="Portfolio Value ($)",
        height=600,
        template="plotly_white"
    )
    return fig

# --- CHARTS ---
def plot_cumulative_performance(results):
    cum_series = results["cum_port"]
    cum_bench = results.get("cum_benchmark", pd.Series(dtype=float))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum_series.index, y=cum_series.values, mode="lines", name="Portfolio", line=dict(color="#5e6ad2", width=3)))
    if not cum_bench.empty:
        fig.add_trace(go.Scatter(x=cum_bench.index, y=cum_bench.values, mode="lines", name="S&P 500 (Excess)", line=dict(color="#333333", width=2, dash="dash")))
    
    # Log Scale
    min_val, max_val = cum_series.min(), cum_series.max()
    if min_val > 0 and max_val > 0:
        log_min, log_max = np.log10(min_val), np.log10(max_val)
        raw_dtick = (log_max - log_min) / 2.5
        magnitude = 10 ** np.floor(np.log10(raw_dtick))
        normalized = raw_dtick / magnitude
        if normalized < 1.5: nice_dtick = 1.0 * magnitude
        elif normalized < 3.5: nice_dtick = 2.0 * magnitude
        elif normalized < 7.5: nice_dtick = 5.0 * magnitude
        else: nice_dtick = 10.0 * magnitude
    else: nice_dtick = 1

    fig.update_layout(
        title="Cumulative Excess Return (Log Scale)", paper_bgcolor="white", plot_bgcolor="white",
        font=dict(color="black", family="Times New Roman"), yaxis_title="Growth of $1 (Log)",
        yaxis=dict(type="log", dtick=nice_dtick, tickformat=".2f", minor=dict(showgrid=False)),
        height=650, template="plotly_white"
    )
    return fig

def plot_weights_over_time(results):
    df = results["weights_df"]
    fig = px.area(df, x=df.index, y=df.columns)
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", font=dict(color="black", family="Times New Roman"), title="Weights Evolution (Stacked)", height=500, template="plotly_white")
    return fig

def plot_risk_evolution(results):
    if "rc_df" not in results: return go.Figure()
    df = results["rc_df"]
    fig = px.line(df, x=df.index, y=df.columns)
    fig.update_layout(title="Risk Contribution Evolution (Target: Equal Risk)", paper_bgcolor="white", plot_bgcolor="white", font=dict(color="black", family="Times New Roman"), yaxis_title="Risk Contribution (%)", height=500, template="plotly_white")
    return fig

def plot_country_exposure_over_time(results):
    df = pd.DataFrame(results["country_exposure_over_time"]).T
    df.index = pd.to_datetime(df.index)
    fig = go.Figure()
    for country in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[country]*100, mode="lines", name=str(country)))
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", font=dict(color="black", family="Times New Roman"), yaxis_title="Exposure (%)", height=500, template="plotly_white")
    return fig

# --- PDF GENERATION ---
def create_pdf_report(results):
    class PDF(FPDF):
        def header(self):
            self.set_font('Helvetica', 'B', 15)
            self.cell(0, 10, 'Pension Fund Optimizer - ERC Report', border=False, align='C')
            self.ln(20)

        def footer(self):
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', align='C')

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)

    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, "1. Executive Summary", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", size=11)
    metrics = [
        ("Expected Return (Ann.)", f"{results['expected_return']:.2f}%"),
        ("Volatility (Ann.)", f"{results['volatility']:.2f}%"),
        ("Sharpe Ratio", f"{results['sharpe']:.2f}"),
        ("Max Drawdown", f"{results['max_drawdown']:.2f}%"),
        ("Transaction Costs", f"{results['total_tc']:.2f}%"),
    ]
    
    col_width = pdf.w / 2.5
    row_height = 8
    
    for key, value in metrics:
        pdf.cell(col_width, row_height, key, border=1)
        pdf.cell(col_width, row_height, value, border=1, ln=True)
        
    pdf.ln(10)

    def add_plot_to_pdf(fig, title):
        pdf.add_page()
        pdf.set_font("Helvetica", 'B', 14)
        pdf.cell(0, 10, title, ln=True)
        pdf.ln(5)
        img_bytes = fig.to_image(format="png", width=1200, height=700, scale=2)
        with io.BytesIO(img_bytes) as img_stream:
            pdf.image(img_stream, x=10, w=190) 

    fig_cum = plot_cumulative_performance(results)
    add_plot_to_pdf(fig_cum, "2. Cumulative Performance")

    fig_weights = plot_weights_over_time(results)
    add_plot_to_pdf(fig_weights, "3. Asset Allocation Evolution")

    fig_risk = plot_risk_evolution(results)
    add_plot_to_pdf(fig_risk, "4. Risk Contribution Evolution")
    
    return bytes(pdf.output(dest='S'))

# --- MAIN APP LAYOUT ---

tab0, tab1, tab2, tab3, tab4 = st.tabs(["How to Use", "Asset Selection", "Portfolio Results", "Monte Carlo", "About Us"])

with tab0:
    components.html(
        """
        <style> body { margin: 0; padding: 0; background-color: #FFFFFF; height: 100vh; width: 100%; overflow: hidden; } .vfrc-widget--chat { background-color: #FFFFFF !important; height: 100% !important; } </style>
        <script type="text/javascript">
          (function(d, t) {
              var v = d.createElement(t), s = d.getElementsByTagName(t)[0];
              v.onload = function() {
                window.voiceflow.chat.load({
                  verify: { projectID: '69283f7c489631e28656d2c1' },
                  url: 'https://general-runtime.voiceflow.com',
                  versionID: 'production',
                  render: { mode: 'embedded', target: document.body },
                  autostart: true
                });
              }
              v.src = "https://cdn.voiceflow.com/widget-next/bundle.mjs";
              v.type = "text/javascript";
              s.parentNode.insertBefore(v, s);
          })(document, 'script');
        </script>
        """,
        height=850, scrolling=False
    )

with tab1:
    
    custom_data, rf_data, tx_cost_data = load_data_bundle()
    if custom_data.empty:
        st.error("Data error.")
    else:
        min_date = custom_data.index.min().date()
        max_date = datetime(2024, 12, 31).date()
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
        end_date = col2.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
        if start_date < end_date:
            valid = get_valid_assets(custom_data, start_date, end_date)
            col1, col2 = st.columns(2)
            selected_stocks = col1.multiselect("Select Stocks", valid["stocks"])
            selected_etfs = col2.multiselect("Select ETFs", valid["etfs"])
            selected_assets = selected_stocks + selected_etfs
            rebalance_freq = st.selectbox("Rebalance Frequency", ["Quarterly", "Semi-Annually", "Annually"], index=2)
            
            if st.button("Optimize My Portfolio"):
                if not selected_assets: st.error("Select assets.")
                else:
                    with st.spinner("Optimizing..."):
                        results = perform_optimization(selected_assets, start_date, end_date, rebalance_freq, custom_data, rf_data, tx_cost_data)
                        if results:
                            st.session_state.results = results
                            st.success("Portfolio results ready!")
        else: st.error("End Date must be after Start Date.")

with tab2:
    
    if "results" in st.session_state:
        res = st.session_state.results
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Excess Return", f"{res['expected_return']:.2f}%")
        col2.metric("Volatility", f"{res['volatility']:.2f}%")
        col3.metric("Sharpe Ratio", f"{res['sharpe']:.2f}")
        col4.metric("Max Drawdown", f"{res['max_drawdown']:.2f}%")
        col5.metric("Trans. Costs", f"{res['total_tc']:.2f}%")
        
        st.plotly_chart(plot_cumulative_performance(res), use_container_width=True)
        c1, c2 = st.columns(2)
        c1.subheader("Weights Evolution")
        c1.plotly_chart(plot_weights_over_time(res), use_container_width=True)
        c2.subheader("Risk Contribution")
        c2.plotly_chart(plot_risk_evolution(res), use_container_width=True)
        st.subheader("Country Exposure")
        st.plotly_chart(plot_country_exposure_over_time(res), use_container_width=True)
        
        st.divider()

        if st.button("Generate PDF Report"):
            with st.spinner("Generating PDF... (This uses Kaleido and might take a moment)"):
                try:
                    pdf_data = create_pdf_report(res)
                    
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_data,
                        file_name=f"ERC_Report_{datetime.now().date()}.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"PDF Generation Error: {e}")
                    st.warning("Ensure 'kaleido==0.2.1' and 'fpdf2' are installed.")
    else:
        st.info("Run optimization first.")

with tab3:
    
    st.write("This simulation projects 10 years into the future using Historical Bootstrap based on your portfolio's assets history.")
    
    if "results" in st.session_state:
        res = st.session_state.results
        
        # User controls for the simulation
        c1, c2 = st.columns(2)
        initial_inv = c1.number_input("Initial Investment ($)", value=100000, step=10000)
        sim_years = c2.slider("Projection Years", 5, 30, 10)
        
        with st.spinner("Running SOTA Historical Bootstrap Simulation..."):
            # SOTA Monte Carlo Call
            dates, median, p95, p05, paths = run_monte_carlo(
                hist_returns_df=res['hist_data'],
                weights=res['weights'],
                years=sim_years,
                initial_capital=initial_inv
            )
            
            if len(dates) > 0:
                # Metrics
                final_median = median[-1]
                final_95 = p95[-1]
                final_05 = p05[-1]
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Median Ending Value", f"${final_median:,.0f}")
                m2.metric("Bull Case (95th)", f"${final_95:,.0f}", delta=f"{((final_95/initial_inv)-1)*100:.0f}%")
                m3.metric("Bear Case (5th)", f"${final_05:,.0f}", delta=f"{((final_05/initial_inv)-1)*100:.0f}%")
                
                # Chart
                st.plotly_chart(plot_monte_carlo(dates, median, p95, p05), use_container_width=True)
                
                # Interpretation
                st.info(f"""
                **Methodology: Historical Bootstrap**
                Unlike basic simulations that assume markets are 'Normal', this simulation samples from **actual historical events** in your assets' history. This accurately captures:
                1. **Fat Tails:** Real market crashes and booms.
                2. **Correlation Spikes:** How your assets move together during crises.
                """)
            else:
                st.error("Insufficient historical data to run bootstrap simulation.")
            
    else:
        st.info("Please optimize a portfolio in the 'Asset Selection' tab first to enable simulations.")

with tab4:
    
    st.write("""
    Welcome to the Pension Fund Optimizer!

    We are a dedicated team of financial experts and developers passionate about helping individuals and institutions optimize their pension funds for maximum efficiency and risk management.

    Our tool uses advanced optimization techniques, specifically Dynamic Equal Risk Contribution (ERC) with annual rebalancing, to create balanced portfolios that aim to equalize the risk contributions from each asset over time.

    Built with Streamlit and powered by open-source libraries, this app provides an intuitive interface for selecting assets, analyzing historical data, and visualizing results.

    If you have any questions or feedback, feel free to reach out at support@pensionoptimizer.com.

    Thank you for using our tool! 🎉
    """)

    st.markdown("---")
    st.markdown("## 👥 Meet the Team")
    st.markdown("<br>", unsafe_allow_html=True)

    team = [
        {
            "name": "Lucas Jaccard",
            "role": "Frontend Developer",
            "desc": "Lucas designs the app’s visual experience, combining clarity, interactivity, and elegance to make financial analysis more accessible.",
            "photo": "https://raw.githubusercontent.com/quantquant-max/QARM-II-Pension-Fund-ERC/main/team_photos/Lucas_JACCARD.JPG"
        },
        {
            "name": "Audrey Champion",
            "role": "Financial Engineer",
            "desc": "Audrey focuses on translating theory into practice, helping design the pension fund strategy and ensuring academic rigor in implementation.",
            "photo": "https://raw.githubusercontent.com/quantquant-max/QARM-II-Pension-Fund-ERC/main/team_photos/Audrey_CHAMPION.JPG"
        },
        {
            "name": "Arda Budak",
            "role": "Quantitative Analyst",
            "desc": "Arda applies quantitative methods and stochastic simulations to enhance risk control and portfolio diversification within the project.",
            "photo": "https://raw.githubusercontent.com/quantquant-max/QARM-II-Pension-Fund-ERC/main/team_photos/Arda_BUDAK.JPG"
        },
        {
            "name": "Rihem Rhaiem",
            "role": "Data Scientist",
            "desc": "Rihem specializes in financial data analytics and portfolio optimization models, contributing quantitative insight to the ERC framework.",
            "photo": "https://raw.githubusercontent.com/quantquant-max/QARM-II-Pension-Fund-ERC/main/team_photos/Rihem_RHAIEM.JPG"
        },
        {
            "name": "Edward Arion",
            "role": "Backend Developer",
            "desc": "Edward ensures computational stability and performance, integrating optimization algorithms efficiently within the Streamlit app.",
            "photo": "https://raw.githubusercontent.com/quantquant-max/QARM-II-Pension-Fund-ERC/main/team_photos/Edward_ARION.JPG"
        },
    ]

    cols = st.columns(len(team))
    for i, member in enumerate(team):
        with cols[i]:
            st.image(member["photo"], width=150)
            st.markdown(f"### {member['name']}")
            st.markdown(f"**{member['role']}**")
            st.write(member["desc"])
