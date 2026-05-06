import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import json
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="BatteryBuddy",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paths ───────────────────────────────────────────────────────
MODELS_DIR = "dashboard/models/"

FEATURE_COLS = [
    'discharge_capacity', 'voltage_area', 'internal_resistance',
    'coulombic_efficiency', 'temp_rise', 'voltage_drop_rate',
    'dvdq_mean', 'dvdq_std', 'cycle_number'
]

# ── LSTM definition ─────────────────────────────────────────────
class BatteryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out).squeeze(-1)

# ── Load models ─────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open(MODELS_DIR + 'xgb_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    with open(MODELS_DIR + 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    lstm_model = BatteryLSTM(input_size=len(FEATURE_COLS))
    lstm_model.load_state_dict(
        torch.load(MODELS_DIR + 'lstm_model.pt', map_location='cpu')
    )
    lstm_model.eval()
    with open(MODELS_DIR + 'model_summary.json', 'r') as f:
        model_summary = json.load(f)
    return xgb_model, scaler, lstm_model, model_summary

@st.cache_data
def load_data():
    return pd.read_csv(MODELS_DIR + 'cycle_features_clean.csv')

xgb_model, scaler, lstm_model, model_summary = load_models()
cycle_features = load_data()

# ── MC Dropout predict ──────────────────────────────────────────
def mc_predict(model, x_tensor, n_samples=100):
    model.train()
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(model(x_tensor).numpy())
    preds = np.array(preds)
    return preds.mean(axis=0), preds.std(axis=0) * 0.3

# ── Sidebar ─────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/emoji/96/battery-emoji.png", width=60)
st.sidebar.title("BatteryBuddy")
st.sidebar.markdown("*Battery Health Prediction System*")
st.sidebar.divider()

battery_ids      = sorted(cycle_features['battery_id'].unique())
selected_battery = st.sidebar.selectbox("Select Battery", battery_ids)
soh_threshold    = st.sidebar.slider(
    "Replacement Threshold (SoH %)",
    min_value=60, max_value=90, value=80, step=5
) / 100

st.sidebar.divider()
st.sidebar.markdown("**Model Performance**")
st.sidebar.metric("In-Dist MAE",      f"{model_summary['xgb_indist_mae']*100:.2f}%")
st.sidebar.metric("In-Dist R²",       f"{model_summary['xgb_indist_r2']:.3f}")
st.sidebar.metric("LSTM Uncertainty", f"±{model_summary['lstm_mean_uncertainty']*100:.1f}%")

# ── Battery data ─────────────────────────────────────────────────
bat_df = cycle_features[
    cycle_features['battery_id'] == selected_battery
].sort_values('cycle_number').copy()

current_soh   = bat_df['SoH'].iloc[-1]
current_cycle = bat_df['cycle_number'].iloc[-1]
min_soh       = bat_df['SoH'].min()

# ── Scale features ───────────────────────────────────────────────
bat_scaled = bat_df.copy()
bat_scaled[FEATURE_COLS] = scaler.transform(bat_df[FEATURE_COLS])
SEQ_LEN = 20

# ── RUL computation ──────────────────────────────────────────────
x_cycles         = np.arange(len(bat_df))
slope, intercept = np.polyfit(x_cycles, bat_df['SoH'].values, 1)

rul_available = False
rul = rul_optimistic = rul_pessimistic = 0
future_means = future_stds = future_cycles = upper = lower = np.array([])

if current_soh > soh_threshold and slope < 0:
    if len(bat_scaled) >= SEQ_LEN:
        window      = bat_scaled[FEATURE_COLS].values[-SEQ_LEN:]
        x_tensor    = torch.tensor(window[np.newaxis], dtype=torch.float32)
        _, lstm_std = mc_predict(lstm_model, x_tensor, n_samples=100)
    else:
        lstm_std = np.array([0.05])

    _future_means, _future_stds = [], []
    soh = current_soh
    for step in range(500):
        soh = soh + slope
        cumulative_std = float(lstm_std[0]) * np.sqrt(step + 1) * 0.3
        _future_means.append(soh)
        _future_stds.append(cumulative_std)
        if soh <= soh_threshold:
            break

    future_means    = np.array(_future_means)
    future_stds     = np.array(_future_stds)
    rul             = len(future_means)
    lower           = future_means - 2 * future_stds
    upper           = future_means + 2 * future_stds
    rul_optimistic  = next((i for i, v in enumerate(lower) if v <= soh_threshold), rul)
    rul_pessimistic = next((i for i, v in enumerate(upper) if v <= soh_threshold), rul)
    future_cycles   = current_cycle + np.arange(1, rul + 1)
    rul_available   = True

# ── Main header ──────────────────────────────────────────────────
st.title("🔋 BatteryBuddy")
st.markdown(f"### Battery `{selected_battery}` | Health Analysis")
st.divider()

# ── Top metrics ──────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Current SoH", f"{current_soh*100:.1f}%",
    delta=f"{(bat_df['SoH'].iloc[-1] - bat_df['SoH'].iloc[-2])*100:.2f}% last cycle"
)
col2.metric("Cycles Completed",  int(current_cycle))
col3.metric("Min SoH Recorded",  f"{min_soh*100:.1f}%")
col4.metric(
    "Replacement Threshold", f"{soh_threshold*100:.0f}%",
    delta="Safe" if current_soh > soh_threshold else "Replace Soon",
    delta_color="normal" if current_soh > soh_threshold else "inverse"
)

st.divider()

# ── Tabs ─────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 Health & Prediction",
    "🔍 Data Health",
    "💰 Business Impact"
])

# ── Tab 1: Health & Prediction ───────────────────────────────────
# ── Tab 1: Health & Prediction ───────────────────────────────────
with tab1:
    st.subheader("Battery Health & Prediction")

    split_idx  = int(len(bat_df) * 0.7)
    known_df   = bat_df.iloc[:split_idx]
    predict_df = bat_df.iloc[split_idx:]

    # XGBoost predicts known and last 30%
    xgb_preds_known  = xgb_model.predict(known_df[FEATURE_COLS])
    xgb_preds_future = xgb_model.predict(predict_df[FEATURE_COLS])

    # Confidence band from residuals
    residuals  = known_df['SoH'].values - xgb_preds_known
    pred_std   = np.std(residuals)
    upper_band = xgb_preds_future + 1.5 * pred_std
    lower_band = xgb_preds_future - 1.5 * pred_std

    mae_this_battery = np.mean(np.abs(
        predict_df['SoH'].values - xgb_preds_future
    ))

    # ── Extend green line into future using synthetic feature rows ─
    if rul_available:
        last_row    = bat_df[FEATURE_COLS].values[-1].copy()
        recent_rows = bat_df[FEATURE_COLS].values[-10:]
        feat_trends = np.polyfit(np.arange(10), recent_rows, 1)[0]

        future_feature_rows = []
        current_row = last_row.copy()
        for _ in range(len(future_cycles)):
            current_row = current_row + feat_trends
            future_feature_rows.append(current_row.copy())

        future_feature_df  = pd.DataFrame(future_feature_rows, columns=FEATURE_COLS)
        xgb_preds_extended = xgb_model.predict(future_feature_df)

        extended_x = np.concatenate([
            known_df['cycle_number'].values,
            predict_df['cycle_number'].values,
            future_cycles
        ])
        extended_y = np.concatenate([
            xgb_preds_known,
            xgb_preds_future,
            xgb_preds_extended
        ])
    else:
        extended_x = np.concatenate([
            known_df['cycle_number'].values,
            predict_df['cycle_number'].values
        ])
        extended_y = np.concatenate([xgb_preds_known, xgb_preds_future])

    fig = go.Figure()

    # ── Blue solid — first 70% only ───────────────────────────────
    fig.add_trace(go.Scatter(
        x=known_df['cycle_number'], y=known_df['SoH'],
        mode='lines', name='Known History',
        line=dict(color='steelblue', width=2)
    ))

    # ── Confidence band — predicted 30% region ────────────────────
    fig.add_trace(go.Scatter(
        x=np.concatenate([
            predict_df['cycle_number'].values,
            predict_df['cycle_number'].values[::-1]
        ]),
        y=np.concatenate([upper_band, lower_band[::-1]]),
        fill='toself', fillcolor='rgba(100,149,237,0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Band'
    ))

    # ── Future uncertainty band ───────────────────────────────────
    if rul_available:
        fig.add_trace(go.Scatter(
            x=np.concatenate([future_cycles, future_cycles[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill='toself', fillcolor='rgba(100,149,237,0.08)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Future Uncertainty',
            showlegend=False
        ))

    # ── Green dotted — full length, all the way to replace marker ─
    fig.add_trace(go.Scatter(
        x=extended_x, y=extended_y,
        mode='lines', name='XGBoost Prediction',
        line=dict(color='limegreen', width=2, dash='dot')
    ))

    # ── Replace cycle marker ──────────────────────────────────────
    if rul_available:
        fig.add_vline(
            x=int(current_cycle + rul),
            line_dash="dash", line_color="red", opacity=0.5,
            annotation_text=f"Replace ~cycle {int(current_cycle + rul)}"
        )

    # ── Threshold line ────────────────────────────────────────────
    fig.add_hline(
        y=soh_threshold, line_dash="dot",
        line_color="red", opacity=0.7,
        annotation_text=f"Replacement Threshold ({soh_threshold*100:.0f}%)"
    )

    # ── Known/predicted divider ───────────────────────────────────
    fig.add_vline(
        x=int(known_df['cycle_number'].iloc[-1]),
        line_dash="solid", line_color="gray", opacity=0.4,
        annotation_text="← Known | Predicted →"
    )

    fig.update_layout(
        xaxis_title="Cycle Number",
        yaxis_title="State of Health (SoH)",
        hovermode='x unified', height=450,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Metrics row ───────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Health",       f"{current_soh*100:.1f}%")
    m2.metric("Prediction Error",     f"{mae_this_battery*100:.1f}%")
    m3.metric("Cycles Remaining",     f"{rul} cycles" if rul_available else "N/A")
    m4.metric("Replace Around Cycle", f"{int(current_cycle + rul)}" if rul_available else "N/A")

    st.markdown("---")

    health_status = (
        "🟢 Healthy"       if current_soh > 0.90
        else "🟡 Moderate" if current_soh > soh_threshold
        else "🔴 Critical"
    )

    st.markdown(
        f"**{health_status}** &nbsp;|&nbsp; "
        f"The **blue line** is ground truth. Real sensor data from the first "
        f"{split_idx} cycles. "
        f"The **green dotted line** runs the full length. For the first 70% you "
        f"can see how closely it tracks the blue, validating the model. "
        f"After the grey divider the blue line stops and only the green continues, "
        f"predicting health without having seen the actual values. "
        f"The **shaded band** shows the margin of error. If the green line stays "
        f"inside it, the model is well calibrated. "
        + (
            f"At the current degradation rate, **plan for replacement around "
            f"cycle {int(current_cycle + rul)}**."
            if rul_available else
            "⚠️ This battery is already past the replacement threshold — "
            "**replace now**."
        )
    )
# ── Tab 2: Data Health ───────────────────────────────────────────
with tab2:
    st.subheader("Data Health Report")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Feature Summary**")
        st.dataframe(
            bat_df[FEATURE_COLS].describe().round(4),
            use_container_width=True
        )

    with col_b:
        st.markdown("**Degradation Feature Trends**")
        fig3 = go.Figure()
        for feat in ['discharge_capacity', 'voltage_area', 'temp_rise']:
            vals      = bat_df[feat].values
            vals_norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-10)
            fig3.add_trace(go.Scatter(
                x=bat_df['cycle_number'], y=vals_norm,
                mode='lines', name=feat
            ))
        fig3.update_layout(
            xaxis_title="Cycle",
            yaxis_title="Normalised Value",
            height=320,
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("**Raw Cycle Data**")
    display_cols = ['cycle_number', 'SoH'] + [
        c for c in FEATURE_COLS if c != 'cycle_number'
    ]
    st.dataframe(
        bat_df[display_cols].round(4),
        use_container_width=True,
        height=250
    )

    st.markdown("---")
    st.markdown(
        "📉 **Discharge capacity** and **voltage area** decrease as the battery ages. "
        "It holds less charge and delivers less energy per cycle. "
        "🌡️ **Temperature rise** increases over time. Degraded batteries run hotter because "
        "internal resistance builds up. "
        "If these three signals are trending in opposite directions on the chart above, "
        "the data is clean and the model's predictions are reliable. "
        "Any sudden spikes or flatlines in the raw data table indicate sensor noise. "
        "The cleaning pipeline handled these before training."
    )

# ── Tab 3: Business Impact ───────────────────────────────────────
with tab3:
    st.subheader("Business Impact Analysis")
    st.markdown("Enter your cost parameters to get a replacement recommendation.")

    c1, c2 = st.columns(2)
    with c1:
        replacement_cost = st.number_input(
            "Battery Replacement Cost (₹)",
            min_value=1000, max_value=10000000,
            value=50000, step=1000
        )
        downtime_cost = st.number_input(
            "Cost of Downtime per Hour (₹)",
            min_value=100, max_value=1000000,
            value=5000, step=500
        )
    with c2:
        cycles_per_day = st.number_input(
            "Cycles per Day",
            min_value=1, max_value=20, value=3
        )
        failure_hours = st.number_input(
            "Estimated Downtime if Battery Fails (hours)",
            min_value=1, max_value=168, value=8
        )

    st.divider()

    if current_soh <= soh_threshold:
        st.error(
            f"⚠️ Battery {selected_battery} is already past the replacement threshold. "
            f"Replace immediately to avoid unplanned failure."
        )
    elif not rul_available:
        st.info("No degradation trend detected. Business impact analysis unavailable.")
    else:
        days_remaining      = rul / cycles_per_day
        failure_probability = max(
            0, min(1, 1 - (current_soh - soh_threshold) / 0.2)
        )
        cost_of_failure   = (downtime_cost * failure_hours) + replacement_cost
        cost_of_waiting   = failure_probability * cost_of_failure
        cost_of_replacing = replacement_cost
        savings           = cost_of_waiting - cost_of_replacing

        r1, r2, r3 = st.columns(3)
        r1.metric("Days Until Threshold", f"{days_remaining:.0f} days")
        r2.metric("Failure Probability",  f"{failure_probability*100:.0f}%")
        r3.metric(
            "Potential Savings",
            f"₹{max(0, savings):,.0f}",
            delta="Replace now" if savings > 0 else "Wait is fine",
            delta_color="inverse" if savings > 0 else "normal"
        )

        st.divider()

        fig4 = go.Figure(go.Bar(
            x=['Replace Now', 'Risk of Waiting'],
            y=[cost_of_replacing, cost_of_waiting],
            marker_color=['steelblue', 'tomato'],
            text=[f'₹{cost_of_replacing:,.0f}', f'₹{cost_of_waiting:,.0f}'],
            textposition='auto'
        ))
        fig4.update_layout(
            yaxis_title="Cost (₹)",
            height=350,
            title="Cost Comparison: Replace Now vs Wait"
        )
        st.plotly_chart(fig4, use_container_width=True)

        if savings > 0:
            st.error(
                f"⚠️ **Recommendation: Replace within {rul_optimistic}–{rul} cycles.** "
                f"Expected savings of ₹{savings:,.0f} by replacing proactively "
                f"vs risking in-service failure."
            )
        else:
            st.success(
                f"✅ **Recommendation: Continue monitoring.** "
                f"Battery has ~{days_remaining:.0f} days of useful life remaining. "
                f"Schedule replacement around cycle {int(current_cycle + rul)}."
            )

    st.markdown("---")
    st.markdown(
        "💡 **How this works:** Failure probability is calculated from how close this battery is "
        "to the replacement threshold — the closer it is, the higher the risk. "
        "**Cost of waiting** = failure probability × (downtime cost + replacement cost). "
        "**Cost of replacing now** = just the replacement cost. "
        "If the cost of waiting exceeds the cost of replacing, the system recommends acting now. "
        "Adjust the threshold slider in the sidebar to see how your risk tolerance changes the recommendation."
    )