import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import cvxpy as cp
from scipy.optimize import minimize_scalar
from datetime import datetime, timedelta
from fpdf import FPDF
import io
from sklearn.covariance import LedoitWolf
import plotly.express as px

# Custom styling
st.set_page_config(page_title="Pension Fund Optimizer", layout="wide")
st.logo("ERC Portfolio.png")

st.markdown(
    """
    <style>
    :root {
        --primary-color: #f0f0f0;
    }
    .stApp {
        background-color: #000000;
        color: #f0f0f0;
        font-family: 'Times New Roman', serif;
    }
    .stSidebar {
        background-color: #111111;
        color: #f0f0f0;
        font-family: 'Times New Roman', serif;
    }
    .stButton>button {
        background-color: #f0f0f0;
        color: #000000;
        border-radius: 8px;
        padding: 10px 20px;
        font-family: 'Times New Roman', serif;
    }
    .stButton>button:hover {
        background-color: #dddddd;
    }
    .stHeader {
        color: #f0f0f0;
        font-size: 32px;
        font-weight: bold;
        font-family: 'Times New Roman', serif;
    }
    .stExpander {
        background-color: #222222;
        color: #f0f0f0;
        font-family: 'Times New Roman', serif;
    }
    .stMultiSelect [data-testid=stMarkdownContainer] {
        color: #f0f0f0;
        font-family: 'Times New Roman', serif;
    }
    .stPlotlyChart {
        background-color: #000000;
    }
    .stDateInput label {
        color: #f0f0f0 !important;
        font-family: 'Times New Roman', serif;
    }
    .stTable {
        color: #f0f0f0 !important;
        font-family: 'Times New Roman', serif;
    }
    table {
        color: #f0f0f0 !important;
        font-family: 'Times New Roman', serif;
    }
    th, td {
        color: #f0f0f0 !important;
        font-family: 'Times New Roman', serif;
    }
    .stMetric {
        color: #f0f0f0 !important;
        font-family: 'Times New Roman', serif;
    }
    .stMetric label {
        color: #f0f0f0 !important;
        font-family: 'Times New Roman', serif;
    }
    .stMetricValue {
        color: #f0f0f0 !important;
        font-family: 'Times New Roman', serif;
    }
    [data-testid="stMetric"] {
        color: #f0f0f0 !important;
        font-family: 'Times New Roman', serif;
    }
    [data-testid="stMetricLabel"] {
        color: #f0f0f0 !important;
        font-family: 'Times New Roman', serif;
    }
    [data-testid="stMetricValue"] {
        color: #f0f0f0 !important;
        font-family: 'Times New Roman', serif;
    }
    div[data-testid="metric-container"] {
        color: #f0f0f0 !important;
        font-family: 'Times New Roman', serif;
    }
    div[data-testid="metric-container"] p {
        color: #f0f0f0 !important;
        font-family: 'Times New Roman', serif;
    }
    div[data-testid="metric-container"] div {
        color: #f0f0f0 !important;
        font-family: 'Times New Roman', serif;
    }
    header {
        background-color: #000000 !important;
    }
    header img {
        height: 60px !important;
        width: auto !important;
    }
    div[data-testid="stAlert"] {
        background-color: #111111 !important;
        color: #f0f0f0 !important;
        border-color: #f0f0f0 !important;
        font-family: 'Times New Roman', serif;
    }
    div.kind-error {
        background-color: #111111 !important;
        color: #f0f0f0 !important;
        font-family: 'Times New Roman', serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Data loading functions
@st.cache_data
def load_custom_data():
    comp = pd.read_parquet("compustat_git.parquet")
    etf = pd.read_parquet("etf_git.parquet")

    
    comp = comp[["date", "company_name", "monthly_return"]].copy()
    comp["date"] = pd.to_datetime(comp["date"])

    comp = comp.rename(columns={
        "company_name": "asset",  
        "monthly_return": "ret"
    })

    etf = etf.rename(columns={
        "ETF": "asset",
        "return_monthly": "ret"     
    })

    etf = etf[["date", "asset", "ret"]].copy()
    etf["date"] = pd.to_datetime(etf["date"])

    returns_long = pd.concat([comp, etf], ignore_index=True)

    returns_wide = (
        returns_long
        .pivot(index="date", columns="asset", values="ret")
        .sort_index()
    )
    returns_wide.index = pd.to_datetime(returns_wide.index)

    return returns_wide
    
@st.cache_data
def load_country_mapping():
    comp = pd.read_parquet("compustat_git.parquet")
    mapping = comp[["company_name", "country_code"]].drop_duplicates()
    mapping = mapping.set_index("company_name")["country_code"].to_dict()
    return mapping


def get_data(tickers, start, end, custom_data):
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    if custom_data.empty:
        return pd.DataFrame()

    missing = [t for t in tickers if t not in custom_data.columns]

    if len(missing) > 0:
        st.warning(f"⚠️ The following assets do not exist in the database: {missing}")
        tickers = [t for t in tickers if t in custom_data.columns]

    if len(tickers) == 0:
        return pd.DataFrame()

    data = custom_data.loc[start:end, tickers]

    data = data.sort_index()

    return data


@st.cache_data
def get_valid_assets(custom_data, start_date, end_date):

    start_date = pd.to_datetime(start_date)
    end_date   = pd.to_datetime(end_date)

    comp = pd.read_parquet("compustat_git.parquet")
    etf  = pd.read_parquet("etf_git.parquet")

    comp_assets = sorted(comp["company_name"].unique())
    etf_assets  = sorted(etf["ETF"].unique())

    subset = custom_data.loc[start_date:end_date]

    available_assets = subset.columns[subset.notna().any()].tolist()

    valid_stocks = sorted(list(set(comp_assets) & set(available_assets)))
    valid_etfs   = sorted(list(set(etf_assets)  & set(available_assets)))

    return {
        "stocks": valid_stocks,
        "etfs": valid_etfs
    }
import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize_scalar
from sklearn.covariance import LedoitWolf
import streamlit as st


def get_common_start_date(custom_data: pd.DataFrame,
                          selected_assets: list[str],
                          user_start_date) -> pd.Timestamp:
    user_start_date = pd.to_datetime(user_start_date)

    missing = [a for a in selected_assets if a not in custom_data.columns]
    if missing:
        st.error(f"The selected assets are not available in the database : {missing}")
        return None


    first_valid = custom_data[selected_assets].apply(lambda col: col.first_valid_index())


    common_start = first_valid.max()

    if pd.isna(common_start):
        st.error("No common valid date found for the selected assets.")
        return None


    if common_start > user_start_date:
        st.warning(
            f"⚠️ The chosen start date ({user_start_date.date()}) "
            f"is not available for the selected assets.\n\n"
            f"➡️ The optimisation will start on {common_start.date()}, "
            f"which is the first date where all return series are available."
        )

    return max(common_start, user_start_date)


def compute_rebalance_indices(dates: pd.DatetimeIndex, freq_label: str) -> list[int]:

    if freq_label == "Quarterly":
        step = 3
    elif freq_label == "Semi-Annually":
        step = 6
    elif freq_label == "Annually":
        step = 12
    else:
        raise ValueError(f"Unknown frequency : {freq_label}")

    n = len(dates)
    idxs = list(range(0, n, step))
    if idxs[-1] != n - 1:
        idxs.append(n - 1)  # on force un dernier rebalance à la fin

    return idxs


def solve_erc_weights(cov_matrix: np.ndarray) -> np.ndarray:
    n = cov_matrix.shape[0]

    def solve_with_rho(rho: float):
        w = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(w, cov_matrix) - rho * cp.sum(cp.log(w)))
        constraints = [cp.sum(w) == 1, w >= 1e-6]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)
        if prob.status == "optimal":
            return np.array(w.value).flatten()
        return None

    def rc_variance(rho: float) -> float:
        w = solve_with_rho(rho)
        if w is None:
            return np.inf
        var = w @ cov_matrix @ w
        sigma = np.sqrt(var)
        if sigma <= 0:
            return np.inf
        mrc = cov_matrix @ w
        rc = w * mrc / sigma
        return np.var(rc)

    res = minimize_scalar(
        rc_variance,
        bounds=(1e-6, 1e-1),
        method="bounded",
        tol=1e-5
    )
    best_rho = res.x
    w_star = solve_with_rho(best_rho)
    if w_star is None:
        raise RuntimeError("ERC Optimisation Failed (No optimal solution has been found).")

    w_star = np.where(np.abs(w_star) < 1e-6, 0, w_star)
    w_star = np.clip(w_star, 0, None)
    if w_star.sum() <= 0:
        raise RuntimeError("Non-valid ERC Solution (sum of weights <= 0).")
    w_star /= w_star.sum()
    return w_star

def compute_max_drawdown(cumulative_returns: pd.Series) -> float:
    running_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - running_max) / running_max
    return drawdowns.min() * 100   

def perform_optimization(
    selected_assets: list[str],
    start_date_user,
    end_date_user,
    rebalance_freq: str,
    custom_data: pd.DataFrame,
    lookback_months: int = 36,
    ann_factor: int = 12,
    tc_rate: float = 0.001,
):
    country_map = load_country_mapping()
    country_exposure_over_time = {}


    try:

        start_date_user = pd.to_datetime(start_date_user)
        end_date_user   = pd.to_datetime(end_date_user)

        if custom_data.empty:
            st.error("Market data is empty.")
            return None


        common_start = get_common_start_date(custom_data, selected_assets, start_date_user)
        if common_start is None:
            return None


        first_rebalance_date = common_start + pd.DateOffset(months=lookback_months)

        if first_rebalance_date > end_date_user:
            st.error(
                f"Not enough data to compute a {lookback_months}-month covariance window "
                f"before the selected end date {end_date_user.date()}."
            )
            return None

        if first_rebalance_date > start_date_user:
            st.warning(
                f"⚠️ The optimization cannot start at your chosen date **{start_date_user.date()}**.\n\n"
                f"➡️ It will start on **{first_rebalance_date.date()}**, which is "
                f"{lookback_months} months after the earliest date where *all* selected "
                f"assets have return data."
            )


        full_returns = custom_data[selected_assets].sort_index()
        full_returns = full_returns.loc[common_start:end_date_user]

        if full_returns.shape[0] < lookback_months + 1:
            st.error(
                f"Not enough total history to have {lookback_months} months of data "
                f"between {common_start.date()} and {end_date_user.date()}."
            )
            return None

        period_returns = full_returns.loc[first_rebalance_date:end_date_user]
        if period_returns.empty:
            st.error("No available return data after earliest optimisation date.")
            return None

        period_dates = period_returns.index

        rebalance_indices = compute_rebalance_indices(period_dates, rebalance_freq)

        n = len(selected_assets)
        previous_weights = np.zeros(n)
        port_returns = pd.Series(index=period_dates, dtype=float)
        weights_over_time = {}
        total_tc = 0.0

        last_cov = None  # pour RC à la fin


        for j, reb_idx in enumerate(rebalance_indices):

            rebal_date = period_dates[reb_idx]

            # Position globale de cette date dans full_returns
            global_reb_pos = full_returns.index.get_loc(rebal_date)
            start_pos = max(0, global_reb_pos - lookback_months)

            # Fenêtre d'estimation sur full_returns, pas period_returns
            est_window = full_returns.iloc[start_pos:global_reb_pos]

            est_window = est_window.dropna(how="all")
            est_window = est_window.dropna(how="any")

            if est_window.shape[0] < n + 1:
                st.error(
                    f"Not enough clean data to estimate covariance before "
                    f"rebalance date {rebal_date.date()}."
                )
                return None

    
            lw = LedoitWolf().fit(est_window.values)
            cov = lw.covariance_ * ann_factor
            last_cov = cov


            try:
                weights = solve_erc_weights(cov)
            except Exception as e:
                st.error(f"ERC optimisation failed on {rebal_date.date()} : {e}")
                return None


            turnover = np.sum(np.abs(weights - previous_weights)) / 2
            total_tc += turnover * tc_rate

            previous_weights = weights.copy()
            weights_over_time[rebal_date] = weights
            
            country_exp = {}
            for asset, w in zip(selected_assets, weights):
                if asset in country_map:   
                    c = country_map[asset]
                    country_exp[c] = country_exp.get(c, 0) + w

            country_exposure_over_time[rebal_date] = country_exp


            if j == len(rebalance_indices) - 1:
                start_slice = reb_idx
                end_slice   = len(period_dates)
            else:
                start_slice = reb_idx
                end_slice   = rebalance_indices[j + 1]

            sub_ret = period_returns.iloc[start_slice:end_slice].fillna(0.0)
            if not sub_ret.empty:
                port_ret = sub_ret.values @ weights
                port_returns.iloc[start_slice:end_slice] = port_ret


        port_returns = port_returns.dropna()
        if port_returns.empty:
            st.error("Final portfolio return series is empty.")
            return None

        cum_port = (1 + port_returns).cumprod()
        max_drawdown = compute_max_drawdown(cum_port)

        ann_return = port_returns.mean() * ann_factor
        ann_vol = port_returns.std() * np.sqrt(ann_factor)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0


        if last_cov is None:
            st.error("No covariance matrix found to compute risk contributions.")
            return None

        port_var = weights @ last_cov @ weights
        sigma_p = np.sqrt(port_var)
        mrc = last_cov @ weights
        rc_abs = weights * mrc / sigma_p
        rc_pct = (rc_abs / rc_abs.sum()) * 100 if rc_abs.sum() > 0 else np.zeros_like(rc_abs)

        weights_df = (
            pd.DataFrame(weights_over_time, index=selected_assets)
            .T.sort_index()
        )

        corr_matrix = est_window.corr()


        return {
            "selected_assets": selected_assets,
            "weights": weights,
            "risk_contrib_abs": rc_abs,
            "risk_contrib_pct": rc_pct,
            "expected_return": ann_return * 100,
            "volatility": ann_vol * 100,
            "sharpe": sharpe,
            "port_returns": port_returns,
            "cum_port": cum_port,
            "total_tc": total_tc * 100,
            "weights_df": weights_df,
            "corr_matrix": corr_matrix,
            "first_rebalance_date": first_rebalance_date,
            "common_start": common_start,
            "country_exposure_over_time": country_exposure_over_time,
            "max_drawdown": max_drawdown
        }

    except Exception as e:
        st.error(f"Unexpected error in optimisation: {e}")
        return None



def plot_final_weights(results):
    df = pd.DataFrame({
        "Asset": results["selected_assets"],
        "Weight (%)": results["weights"] * 100
    }).sort_values("Weight (%)", ascending=True)

    fig = px.bar(
        df,
        x="Weight (%)",
        y="Asset",
        orientation="h",
    )


    fig.update_traces(
        marker_color="#0D6EFD",
        text=df["Weight (%)"].map(lambda x: f"{x:.2f}%"),
        textposition="outside"
    )

    fig.update_layout(
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font=dict(color="#E0E0E0", family="Times New Roman"),

        xaxis=dict(
            title="Weight (%)",
            title_font=dict(color="#E0E0E0"),
            tickfont=dict(color="#E0E0E0"),
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="",
            title_font=dict(color="#E0E0E0"),
            tickfont=dict(color="#E0E0E0"),
            showgrid=False,
        ),

        margin=dict(l=120, r=40, t=60, b=40)
    )

    return fig
def plot_risk_contributions(results):
    df = pd.DataFrame({
        "Asset": results["selected_assets"],
        "Risk Contribution (%)": results["risk_contrib_pct"]
    }).sort_values("Risk Contribution (%)", ascending=True)

    fig = px.bar(
        df,
        x="Risk Contribution (%)",
        y="Asset",
        orientation="h",
    )

    fig.update_traces(
        marker_color="#00C2FF",
        text=df["Risk Contribution (%)"].map(lambda x: f"{x:.2f}%"),
        textposition="outside"
    )

    fig.update_layout(
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font=dict(color="#E0E0E0", family="Times New Roman"),

        xaxis=dict(
            title="Contribution (%)",
            title_font=dict(color="#E0E0E0"),
            tickfont=dict(color="#E0E0E0"),
            showgrid=False
        ),
        yaxis=dict(
            title="",
            tickfont=dict(color="#E0E0E0"),
            showgrid=False
        ),

        margin=dict(l=120, r=40, t=60, b=40)
    )

    return fig
def plot_cumulative_performance(results):
    cum = results["cum_port"]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=cum.index,
        y=cum.values,
        mode="lines",
        name="Portfolio",
        line=dict(color="#0D6EFD", width=3)
    ))

    fig.update_layout(
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font=dict(color="#E0E0E0", family="Times New Roman"),

        xaxis=dict(
            title="Date",
            title_font=dict(color="#E0E0E0"),
            tickfont=dict(color="#E0E0E0"),
            showgrid=False
        ),
        yaxis=dict(
            title="Cumulative Return",
            title_font=dict(color="#E0E0E0"),
            tickfont=dict(color="#E0E0E0"),
            showgrid=False
        ),

        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E0E0E0")
        ),

        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig
def plot_correlation_matrix(results):
    corr = results["corr_matrix"]

    fig = px.imshow(
        corr,
        color_continuous_scale=[ "#0D6EFD", "#00C2FF", "#FFFFFF"],
        aspect="auto"
    )

    fig.update_layout(
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font=dict(color="#E0E0E0", family="Times New Roman"),

        coloraxis_colorbar=dict(
            title="Corr",
            title_font=dict(color="#E0E0E0"),
            tickfont=dict(color="#E0E0E0")
        ),

        margin=dict(l=80, r=80, t=60, b=40)
    )

    fig.update_xaxes(showgrid=False, tickfont=dict(color="#E0E0E0"))
    fig.update_yaxes(showgrid=False, tickfont=dict(color="#E0E0E0"))

    return fig
def plot_weights_over_time(results):
    df = results["weights_df"]

    fig = px.area(
        df,
        x=df.index,
        y=df.columns,
        color_discrete_sequence=px.colors.qualitative.Dark24
    )

    fig.update_layout(
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font=dict(color="#E0E0E0", family="Times New Roman"),

        xaxis=dict(
            title="Date",
            title_font=dict(color="#E0E0E0"),
            tickfont=dict(color="#E0E0E0"),
            showgrid=False
        ),
        yaxis=dict(
            title="Weight",
            title_font=dict(color="#E0E0E0"),
            tickfont=dict(color="#E0E0E0"),
            showgrid=False
        ),

        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E0E0E0")
        ),

        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig

def plot_country_exposure_pie(results):
    exposures = results["country_exposure_over_time"]
    last_date = sorted(exposures.keys())[-1]
    data = exposures[last_date]

    labels = list(data.keys())
    values = [100 * x for x in data.values()]

    fig = go.Figure(
        data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            textfont=dict(color="#FFF"),  
        )]
    )

    fig.update_layout(
        paper_bgcolor="#000",
        font=dict(color="#DDD"),          
        legend=dict(
            font=dict(color="#FFF"),     
        ),
        margin=dict(l=40, r=40, t=40, b=40)
    )

    return fig

def plot_country_exposure_over_time(results):
    exposures = results["country_exposure_over_time"]

    df = pd.DataFrame(exposures).T
    df.index = pd.to_datetime(df.index)

    fig = go.Figure()

    for country in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[country] * 100,
            mode="lines",
            name=country
        ))

    fig.update_layout(
        title="Country Exposure Over Time",
        xaxis_title="Date",
        yaxis_title="Exposure (%)",
        paper_bgcolor="#000",
        plot_bgcolor="#000",
        font=dict(color="#DDD"),          
        legend=dict(
            font=dict(color="#FFF"),        
        ),
        margin=dict(l=40, r=40, t=40, b=40)
    )

    return fig





def export_csv(weights_df, filename):
    csv = weights_df.to_csv()
    st.download_button(label="Download Weights History as CSV", data=csv, file_name=filename, mime="text/csv")

def export_pdf(results):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Times", size=12)
    pdf.cell(200, 10, txt="Portfolio Results", ln=1, align='C')
    pdf.cell(200, 10, txt=f"Expected Annual Return: {results['expected_return']:.2f}%", ln=1)
    pdf.cell(200, 10, txt=f"Annual Volatility: {results['volatility']:.2f}%", ln=1)
    pdf.cell(200, 10, txt=f"Sharpe Ratio: {results['sharpe']:.2f}", ln=1)
    pdf.cell(200, 10, txt=f"Total Transaction Costs: {results['total_tc']:.2f}%", ln=1)
    pdf_buffer = io.BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)
    st.download_button(label="Download Report as PDF", data=pdf_buffer, file_name="portfolio_report.pdf", mime="application/pdf")

# Tabs
tab0, tab1, tab2, tab3 = st.tabs(["How to Use", "Asset Selection", "Portfolio Results", "About Us"])

with tab0:
    st.title("How to Use")
    st.write("""
    - **Set Date Range**: Select and confirm the start and end month/year for historical performance analysis.
    - **Select Assets**: Choose US stocks from the list of stocks with data in the selected range. Only stocks listed on or before the start date are available.
    - **Rebalance Frequency**: Choose quarterly, semi-annually, or annually.
    - **Optimize**: Click 'Optimize My Portfolio' to generate your results.
    - **Explore**: Review weights, risk contributions, and performance metrics in the Portfolio Results tab.
    """)

with tab1:
    st.title("Asset Selection")


    custom_data = load_custom_data()
    if custom_data.empty:
        st.error("Failed to load dataset.")
        st.stop()

    min_date = custom_data.index.min().date()
    max_date = custom_data.index.max().date()


    st.markdown("### Select Date Range")

    col1, col2 = st.columns(2)
    with col1:
        start_date_user = st.date_input(
            "📅 Start Date",
            value=min_date,
            min_value=min_date,
            max_value=max_date
        )
    with col2:
        end_date_user = st.date_input(
            "📅 End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )

    if start_date_user > end_date_user:
        st.error("Start date must be before end date.")
        st.stop()


    valid = get_valid_assets(custom_data, start_date_user, end_date_user)

    stocks = valid["stocks"]
    etfs   = valid["etfs"]
    all_assets = stocks + etfs

    st.markdown("### Choose Your Assets")

    col1, col2 = st.columns(2)
    with col1:
        selected_stocks = st.multiselect("Stocks", options=stocks)
    with col2:
        selected_etfs = st.multiselect("ETFs", options=etfs)

    selected_assets = selected_stocks + selected_etfs

    if not selected_assets:
        st.info("Select at least one stock or ETF to proceed.")
        st.stop()


    asset_first_dates = {
        a: custom_data[a].first_valid_index().date()
        for a in selected_assets
    }

    common_start = max(asset_first_dates.values())

    if common_start > start_date_user:
        st.warning(
            f"Some assets do not have data at your chosen start date. "
            f"Optimization will start at **{common_start}** instead of **{start_date_user}**."
        )


    rebalance_freq = st.selectbox(
        "Rebalance Frequency",
        options=["Quarterly", "Semi-Annually", "Annually"],
        index=2
    )


    if st.button("Optimize My Portfolio"):
        with st.spinner("Running optimization..."):
            results = perform_optimization(
                selected_assets=selected_assets,
                start_date_user=start_date_user,
                end_date_user=end_date_user,
                rebalance_freq=rebalance_freq,
                custom_data=custom_data,
            )
            if results is not None:
                st.session_state.results = results
                st.success("Optimization complete! See Portfolio Results tab.")


with tab2:
    st.title("Portfolio Results")

    if "results" not in st.session_state:
        st.info("Please run an optimization first.")
        st.stop()

    results = st.session_state.results

    st.subheader("Final Portfolio Weights")
    st.plotly_chart(plot_final_weights(results), use_container_width=True)

    st.subheader("Risk Contributions (%)")
    st.plotly_chart(plot_risk_contributions(results), use_container_width=True)

    st.subheader("Cumulative Portfolio Performance")
    st.plotly_chart(plot_cumulative_performance(results), use_container_width=True)

    st.subheader("Weights Evolution Over Time")
    st.plotly_chart(plot_weights_over_time(results), use_container_width=True)

    st.subheader("Correlation Matrix")
    st.plotly_chart(plot_correlation_matrix(results), use_container_width=True)

    st.subheader("Latest Country Allocation (Stocks ONLY)")
    st.plotly_chart(plot_country_exposure_pie(results), use_container_width=True)

    st.subheader("Country Allocation Over Time (Stocks ONLY)")
    st.plotly_chart(plot_country_exposure_over_time(results), use_container_width=True)

    st.markdown("## Performance Metrics Summary")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Expected Annual Return", f"{results['expected_return']:.2f}%")
    col2.metric("Annual Volatility", f"{results['volatility']:.2f}%")
    col3.metric("Sharpe Ratio", f"{results['sharpe']:.2f}")
    col4.metric("Maximum Drawdown", f"{results['max_drawdown']:.2f}%")
    col5.metric("Total Transaction Costs", f"{results['total_tc']:.2f}%")


st.markdown("<br>", unsafe_allow_html=True)

with tab3:
    st.title("About Us")

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
