import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import cvxpy as cp
from scipy.optimize import minimize_scalar
from datetime import datetime, timedelta
from fpdf import FPDF
import io
from sklearn.covariance import LedoitWolf

# Custom styling
st.set_page_config(page_title="Pension Fund Optimizer", layout="wide")
try:
    st.logo("ERC Portfolio.png")
except:
    pass # Handle case where logo file is missing

st.markdown(
    """
    <style>
    :root { --primary-color: #f0f0f0; }
    .stApp { background-color: #000000; color: #f0f0f0; font-family: 'Times New Roman', serif; }
    .stSidebar { background-color: #111111; color: #f0f0f0; font-family: 'Times New Roman', serif; }
    .stButton>button { background-color: #f0f0f0; color: #000000; border-radius: 8px; padding: 10px 20px; font-family: 'Times New Roman', serif; }
    .stButton>button:hover { background-color: #dddddd; }
    .stHeader { color: #f0f0f0; font-size: 32px; font-weight: bold; font-family: 'Times New Roman', serif; }
    .stExpander { background-color: #222222; color: #f0f0f0; font-family: 'Times New Roman', serif; }
    .stMultiSelect [data-testid=stMarkdownContainer] { color: #f0f0f0; font-family: 'Times New Roman', serif; }
    .stPlotlyChart { background-color: #000000; }
    .stDateInput label { color: #f0f0f0 !important; font-family: 'Times New Roman', serif; }
    .stTable { color: #f0f0f0 !important; font-family: 'Times New Roman', serif; }
    table { color: #f0f0f0 !important; font-family: 'Times New Roman', serif; }
    th, td { color: #f0f0f0 !important; font-family: 'Times New Roman', serif; }
    .stMetric, .stMetric label, .stMetricValue, [data-testid="stMetric"], [data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color: #f0f0f0 !important; font-family: 'Times New Roman', serif; }
    header { background-color: #000000 !important; }
    div[data-testid="stAlert"] { background-color: #111111 !important; color: #f0f0f0 !important; border-color: #f0f0f0 !important; font-family: 'Times New Roman', serif; }
    </style>
    """,
    unsafe_allow_html=True
)

# --- DATA LOADING FUNCTIONS ---

@st.cache_data
def load_rf_data():
    """Loads the Risk Free Rate data."""
    try:
        rf = pd.read_parquet("risk_free_rate.parquet")
        # Ensure index is datetime
        if "date" in rf.columns:
            rf = rf.set_index("date")
        rf.index = pd.to_datetime(rf.index)
        
        # Keep only the first column and rename it to 'RF' for consistency
        rf = rf.iloc[:, [0]]
        rf.columns = ["RF"]
        return rf
    except Exception as e:
        # If file is missing, return empty DF to signal fallback to 0%
        return pd.DataFrame()

@st.cache_data
def load_custom_data():
    try:
        comp = pd.read_parquet("compustat_git.parquet")
        etf = pd.read_parquet("etf_git.parquet")

        comp = comp[["date", "company_name", "monthly_return"]].copy()
        comp["date"] = pd.to_datetime(comp["date"])
        comp = comp.rename(columns={"company_name": "asset", "monthly_return": "ret"})

        etf = etf[["date", "ETF", "return_monthly"]].copy()
        etf["date"] = pd.to_datetime(etf["date"])
        etf = etf.rename(columns={"ETF": "asset", "return_monthly": "ret"})

        returns_long = pd.concat([comp, etf], ignore_index=True)
        returns_wide = returns_long.pivot(index="date", columns="asset", values="ret").sort_index()
        returns_wide.index = pd.to_datetime(returns_wide.index)
        return returns_wide
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_country_mapping():
    try:
        comp = pd.read_parquet("compustat_git.parquet")
        mapping = comp[["company_name", "country_code"]].drop_duplicates()
        mapping = mapping.set_index("company_name")["country_code"].to_dict()
        return mapping
    except:
        return {}

def get_valid_assets(custom_data, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Reload raw files just to get unique lists (or could derive from columns)
    # Using columns from custom_data is faster/safer if already loaded
    if custom_data.empty: 
        return {"stocks": [], "etfs": []}

    subset = custom_data.loc[start_date:end_date]
    available_assets = subset.columns[subset.notna().any()].tolist()
    
    # We try to separate Stocks vs ETFs based on the original logic
    # Since we merged them in load_custom_data, we can't easily distinguish 
    # unless we reload source files or cache the lists. 
    # For speed, let's just reload the source lists once.
    try:
        comp = pd.read_parquet("compustat_git.parquet")
        etf = pd.read_parquet("etf_git.parquet")
        comp_assets = set(comp["company_name"].unique())
        etf_assets = set(etf["ETF"].unique())
    except:
        return {"stocks": available_assets, "etfs": []}

    valid_stocks = sorted(list(comp_assets & set(available_assets)))
    valid_etfs = sorted(list(etf_assets & set(available_assets)))

    return {"stocks": valid_stocks, "etfs": valid_etfs}

def get_common_start_date(custom_data, selected_assets, user_start_date):
    user_start_date = pd.to_datetime(user_start_date)
    missing = [a for a in selected_assets if a not in custom_data.columns]
    if missing:
        st.error(f"Assets not in database: {missing}")
        return None
    
    first_valid = custom_data[selected_assets].apply(lambda col: col.first_valid_index())
    common_start = first_valid.max()
    
    if pd.isna(common_start):
        st.error("No common valid data found.")
        return None
        
    if common_start > user_start_date:
        st.warning(f"⚠️ Adjusted start date to **{common_start.date()}** (data availability).")
        
    return max(common_start, user_start_date)

def compute_rebalance_indices(dates, freq_label):
    freq_map = {"Quarterly": 3, "Semi-Annually": 6, "Annually": 12}
    step = freq_map.get(freq_label, 12)
    n = len(dates)
    idxs = list(range(0, n, step))
    if idxs[-1] != n - 1:
        idxs.append(n - 1)
    return idxs

# --- OPTIMIZATION ENGINE ---

def solve_erc_weights(cov_matrix):
    n = cov_matrix.shape[0]
    
    def solve_with_rho(rho):
        w = cp.Variable(n)
        # ERC Objective: 1/2 w'Sw - rho * sum(log(w))
        objective = cp.Minimize(cp.quad_form(w, cov_matrix) - rho * cp.sum(cp.log(w)))
        constraints = [cp.sum(w) == 1, w >= 1e-6]
        prob = cp.Problem(objective, constraints)
        
        # FIX: Using CLARABEL instead of deprecated ECOS
        try:
            prob.solve(solver=cp.CLARABEL)
        except:
            # Fallback if CLARABEL not installed
            prob.solve(solver=cp.SCS)
            
        if prob.status == "optimal":
            return np.array(w.value).flatten()
        return None

    def rc_variance(rho):
        w = solve_with_rho(rho)
        if w is None: return np.inf
        var = w @ cov_matrix @ w
        sigma = np.sqrt(var)
        if sigma <= 0: return np.inf
        mrc = cov_matrix @ w
        rc = w * mrc / sigma
        return np.var(rc)

    res = minimize_scalar(rc_variance, bounds=(1e-6, 1e-1), method="bounded", tol=1e-5)
    w_star = solve_with_rho(res.x)
    
    if w_star is None: raise RuntimeError("ERC Optimization Failed")
    w_star = np.where(np.abs(w_star) < 1e-6, 0, w_star)
    w_star /= w_star.sum()
    return w_star

def compute_max_drawdown(cumulative_returns):
    running_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - running_max) / running_max
    return drawdowns.min() * 100

@st.cache_data(show_spinner=True)
def perform_optimization(selected_assets, start_date_user, end_date_user, rebalance_freq, _custom_data, lookback_months=36, ann_factor=12, tc_rate=0.001):
    # Note: _custom_data starts with underscore to prevent Streamlit from hashing the whole DF (speedup)
    custom_data = _custom_data 
    country_map = load_country_mapping()
    rf_data = load_rf_data() # Load Risk Free Rate
    
    try:
        start_date_user = pd.to_datetime(start_date_user)
        end_date_user = pd.to_datetime(end_date_user)
        
        common_start = get_common_start_date(custom_data, selected_assets, start_date_user)
        if common_start is None: return None
        
        first_rebalance_date = common_start + pd.DateOffset(months=lookback_months)
        if first_rebalance_date > end_date_user:
            st.error(f"Not enough data for lookback. Need data until {first_rebalance_date.date()}")
            return None
            
        full_returns = custom_data[selected_assets].sort_index().loc[common_start:end_date_user]
        period_returns = full_returns.loc[first_rebalance_date:end_date_user]
        
        if period_returns.empty:
            st.error("Optimization period is empty.")
            return None
            
        period_dates = period_returns.index
        rebalance_indices = compute_rebalance_indices(period_dates, rebalance_freq)
        
        n = len(selected_assets)
        previous_weights = np.zeros(n)
        port_returns = pd.Series(index=period_dates, dtype=float).fillna(0.0)
        weights_over_time = {}
        country_exposure_over_time = {}
        total_tc = 0.0
        last_cov = None

        for j, reb_idx in enumerate(rebalance_indices):
            rebal_date = period_dates[reb_idx]
            global_reb_pos = full_returns.index.get_loc(rebal_date)
            start_pos = max(0, global_reb_pos - lookback_months)
            
            est_window = full_returns.iloc[start_pos:global_reb_pos].dropna(how="any")
            
            if est_window.shape[0] < n + 1:
                st.warning(f"Insufficient data at {rebal_date.date()}, skipping rebalance.")
                weights = previous_weights
            else:
                lw = LedoitWolf().fit(est_window.values)
                cov = lw.covariance_ * ann_factor
                last_cov = cov
                try:
                    weights = solve_erc_weights(cov)
                except:
                    weights = previous_weights
            
            # Calculate Transaction Costs
            turnover = np.sum(np.abs(weights - previous_weights)) / 2
            cost = turnover * tc_rate
            total_tc += cost # Track total for metric
            
            previous_weights = weights.copy()
            weights_over_time[rebal_date] = weights
            
            # Country Exposure
            country_exp = {}
            for asset, w in zip(selected_assets, weights):
                c = country_map.get(asset, "Unknown")
                country_exp[c] = country_exp.get(c, 0) + w
            country_exposure_over_time[rebal_date] = country_exp

            # Apply Returns & Subtract Cost
            if j == len(rebalance_indices) - 1:
                end_slice = len(period_dates)
            else:
                end_slice = rebalance_indices[j+1]
                
            sub_ret = period_returns.iloc[reb_idx:end_slice].fillna(0.0)
            if not sub_ret.empty:
                # 1. Gross Portfolio Return
                period_port_ret = sub_ret.values @ weights
                
                # 2. FIX: Subtract Transaction Cost from the first day of the period
                # (Impact of rebalancing happens instantaneously at the start)
                if len(period_port_ret) > 0:
                    period_port_ret[0] -= cost
                
                port_returns.iloc[reb_idx:end_slice] = period_port_ret

        # Final Metrics
        cum_port = (1 + port_returns).cumprod()
        max_drawdown = compute_max_drawdown(cum_port)
        ann_return = port_returns.mean() * ann_factor
        ann_vol = port_returns.std() * np.sqrt(ann_factor)
        
        # FIX: Sharpe Ratio using Risk Free Rate
        if not rf_data.empty:
            # Align RF to portfolio dates (ffill)
            aligned_rf = rf_data.reindex(port_returns.index, method='ffill').fillna(0.0)["RF"]
            # Assume RF in file is monthly decimal. If annualized, logic differs. 
            # We calculate Excess Return series
            excess_ret = port_returns - aligned_rf
            sharpe = (excess_ret.mean() * ann_factor) / ann_vol if ann_vol > 0 else 0.0
        else:
            # Fallback (Risk Free = 0)
            sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

        # Risk Contributions (Last period)
        if last_cov is not None:
            port_var = weights @ last_cov @ weights
            sigma_p = np.sqrt(port_var)
            mrc = last_cov @ weights
            rc_abs = weights * mrc / sigma_p
            rc_pct = (rc_abs / rc_abs.sum()) * 100
        else:
            rc_pct = np.zeros(n)

        return {
            "selected_assets": selected_assets,
            "weights": weights,
            "risk_contrib_pct": rc_pct,
            "expected_return": ann_return * 100,
            "volatility": ann_vol * 100,
            "sharpe": sharpe,
            "port_returns": port_returns,
            "cum_port": cum_port,
            "total_tc": total_tc * 100,
            "weights_df": pd.DataFrame(weights_over_time, index=selected_assets).T.sort_index(),
            "corr_matrix": est_window.corr(),
            "country_exposure_over_time": country_exposure_over_time,
            "max_drawdown": max_drawdown
        }

    except Exception as e:
        st.error(f"Optimization Error: {e}")
        return None

# --- VISUALIZATION FUNCTIONS ---

def plot_final_weights(results):
    df = pd.DataFrame({"Asset": results["selected_assets"], "Weight (%)": results["weights"] * 100})
    df = df.sort_values("Weight (%)", ascending=True)
    fig = px.bar(df, x="Weight (%)", y="Asset", orientation="h")
    fig.update_traces(marker_color="#0D6EFD", texttemplate="%{x:.2f}%")
    fig.update_layout(paper_bgcolor="#000", plot_bgcolor="#000", font=dict(color="#E0E0E0", family="Times New Roman"))
    return fig

def plot_risk_contributions(results):
    df = pd.DataFrame({"Asset": results["selected_assets"], "Risk Contribution (%)": results["risk_contrib_pct"]})
    df = df.sort_values("Risk Contribution (%)", ascending=True)
    fig = px.bar(df, x="Risk Contribution (%)", y="Asset", orientation="h")
    fig.update_traces(marker_color="#00C2FF", texttemplate="%{x:.2f}%")
    fig.update_layout(paper_bgcolor="#000", plot_bgcolor="#000", font=dict(color="#E0E0E0", family="Times New Roman"))
    return fig

def plot_cumulative_performance(results):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results["cum_port"].index, y=results["cum_port"].values, mode="lines", name="Portfolio", line=dict(color="#0D6EFD", width=3)))
    fig.update_layout(title="Net Cumulative Return (After Fees)", paper_bgcolor="#000", plot_bgcolor="#000", font=dict(color="#E0E0E0", family="Times New Roman"))
    return fig

def plot_weights_over_time(results):
    df = results["weights_df"]
    fig = px.area(df, x=df.index, y=df.columns)
    fig.update_layout(paper_bgcolor="#000", plot_bgcolor="#000", font=dict(color="#E0E0E0", family="Times New Roman"))
    return fig

def plot_correlation_matrix(results):
    fig = px.imshow(results["corr_matrix"], color_continuous_scale=["#0D6EFD", "#FFFFFF"], aspect="auto")
    fig.update_layout(paper_bgcolor="#000", plot_bgcolor="#000", font=dict(color="#E0E0E0", family="Times New Roman"))
    return fig

def plot_country_exposure_over_time(results):
    df = pd.DataFrame(results["country_exposure_over_time"]).T
    df.index = pd.to_datetime(df.index)
    fig = go.Figure()
    for country in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[country]*100, mode="lines", name=str(country)))
    fig.update_layout(paper_bgcolor="#000", plot_bgcolor="#000", font=dict(color="#E0E0E0", family="Times New Roman"), yaxis_title="Exposure (%)")
    return fig

# --- MAIN APP LAYOUT ---

tab0, tab1, tab2, tab3 = st.tabs(["How to Use", "Asset Selection", "Portfolio Results", "About Us"])

with tab0:
    st.title("How to Use")
    st.markdown("""
    1. **Asset Selection**: Choose your date range and assets.
    2. **Optimization**: The app calculates the Equal Risk Contribution (ERC) portfolio.
    3. **Results**: View Net Returns (after transaction costs) and risk metrics.
    """)

with tab1:
    st.title("Asset Selection")
    custom_data = load_custom_data()
    
    if custom_data.empty:
        st.error("Data not loaded. Check parquet files.")
    else:
        min_date = custom_data.index.min().date()
        max_date = datetime(2024, 12, 31).date()
        
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
        end_date = col2.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
        
        if start_date < end_date:
            valid = get_valid_assets(custom_data, start_date, end_date)
            col1, col2 = st.columns(2)
            sel_stocks = col1.multiselect("Stocks", valid["stocks"])
            sel_etfs = col2.multiselect("ETFs", valid["etfs"])
            selected_assets = sel_stocks + sel_etfs
            
            rebalance_freq = st.selectbox("Rebalance Frequency", ["Quarterly", "Semi-Annually", "Annually"], index=2)
            
            if st.button("Optimize My Portfolio"):
                if not selected_assets:
                    st.error("Select assets.")
                else:
                    with st.spinner("Optimizing..."):
                        # Pass custom_data with underscore in args to prevent hashing if using the cached function
                        results = perform_optimization(selected_assets, start_date, end_date, rebalance_freq, custom_data)
                        if results:
                            st.session_state.results = results
                            st.success("Done! Go to Results tab.")
        else:
            st.error("Invalid dates.")

with tab2:
    st.title("Portfolio Results")
    if "results" in st.session_state:
        res = st.session_state.results
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Exp. Annual Return", f"{res['expected_return']:.2f}%")
        col2.metric("Annual Volatility", f"{res['volatility']:.2f}%")
        col3.metric("Sharpe Ratio", f"{res['sharpe']:.2f}")
        col4.metric("Max Drawdown", f"{res['max_drawdown']:.2f}%")
        col5.metric("Total Trans. Costs", f"{res['total_tc']:.2f}%")
        
        st.plotly_chart(plot_cumulative_performance(res), use_container_width=True)
        
        c1, c2 = st.columns(2)
        c1.subheader("Weights")
        c1.plotly_chart(plot_final_weights(res), use_container_width=True)
        c2.subheader("Risk Contributions")
        c2.plotly_chart(plot_risk_contributions(res), use_container_width=True)
        
        st.subheader("Weights Evolution")
        st.plotly_chart(plot_weights_over_time(res), use_container_width=True)
        
        st.subheader("Country Exposure")
        st.plotly_chart(plot_country_exposure_over_time(res), use_container_width=True)
    else:
        st.info("Run optimization first.")

with tab3:
    st.title("About Us")
    st.write("Pension Fund Optimizer Team.")
