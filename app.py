import streamlit as st
import pandas as pd
import numpy as np
import io
from model import load_data, DRLTradingModel 
import time # For simulating a professional loading delay

# --- 0. Configuration and Helper Functions ---

# Custom CSS for a professional, dark-mode focused look
def load_css():
    st.markdown("""
    <style>
        /* Main App Background and Layout */
        .main {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        /* Custom Header Styling (Branding) */
        .stApp header {
            background-color: #0c1821; /* Deep navy/charcoal */
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            padding: 10px 0;
            display: none; /* Hide default Streamlit header */
        }
        
        /* Custom Title with an Accent Line */
        h1 {
            color: #4CAF50; /* Primary Green Accent */
            border-bottom: 3px solid #00C853;
            padding-bottom: 10px;
            font-weight: 700;
        }

        /* Metrics Styling - Focus on clarity and data hierarchy */
        [data-testid="stMetric"] {
            background-color: #212F3D; /* Darker blue background for metrics */
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #00C853; /* Light green border */
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.3);
        }
        
        /* Center the Metric Label */
        [data-testid="stMetricLabel"] {
            text-align: center;
            font-size: 1.1em;
            color: #BBDEFB; /* Light blue for labels */
        }
        
        /* Styling for Signal Metrics (BUY/SELL) */
        .stSuccess { background-color: rgba(76, 175, 80, 0.1); border-left: 5px solid #4CAF50; } /* BUY/Profit */
        .stWarning { background-color: rgba(255, 152, 0, 0.1); border-left: 5px solid #FF9800; } /* SELL/Hold (Neutral) */
        .stError { background-color: rgba(244, 67, 54, 0.1); border-left: 5px solid #F44336; } /* Error/Risk Avoided */

    </style>
    """, unsafe_allow_html=True)

# Function to add a branded header
def professional_header():
    col_logo, col_title = st.columns([1, 6])
    with col_logo:
        # Using a simple icon as a logo placeholder
        st.markdown("# 📈", unsafe_allow_html=True) 
    with col_title:
        st.title("DRL Quantum Trade Engine")
        st.caption("Autonomous Policy Simulation & Backtesting Interface")
    st.markdown("---")


# Cache the model instantiation for performance
@st.cache_resource(ttl=3600)
def instantiate_model(data, initial_capital):
    """Initializes the DRL Model with caching."""
    return DRLTradingModel(data.copy(), initial_capital)

# --- 1. Main Application Function ---

def main():
    # FIX: Removed initial_sidebar_state="expanded" to prevent conflict with user's hide preference.
    # The user can now always recover the sidebar via the hamburger menu.
    st.set_page_config(layout="wide", page_title="DRL Quantum Trade Engine")
    
    # Apply custom CSS
    load_css()
    
    # Custom Branded Header
    professional_header()

    # --- Sidebar: Input/Configuration (Phase 1: Data Upload) ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.subheader("Data Source")
        
        uploaded_file = st.file_uploader( 
            "Upload Nifty 50 Snapshot CSV",
            type=['csv'],
            key="data_uploader",
            help="The CSV must contain 'Symbol', 'LTP', '30 d % chng', and '365 d % chng' columns."
        )

    # --- Data Validation Check (Main body warning) ---
    if uploaded_file is None:
        st.warning("The DRL Engine requires stock data to operate. Please upload a CSV file in the sidebar to activate the system parameters.")
        st.stop()
        
    # --- Data Loading (Script continues only if file is uploaded) ---
    with st.spinner("Processing uploaded data..."):
        try:
            uploaded_file_content = uploaded_file.getvalue()
            data = load_data(uploaded_file_content)
        except Exception as e:
            st.error(f"Data Processing Error: Could not load or parse the file. Details: {e}")
            st.stop()
            
    # --- Sidebar: Input/Configuration (Phase 2: Parameters, rendered AFTER data is loaded) ---
    with st.sidebar:
        st.subheader("Trade Parameters")
        
        initial_capital = st.number_input( 
            "Initial Portfolio Capital (₹)",
            min_value=100000,
            max_value=10000000,
            value=1000000,
            step=100000,
            format="%d"
        )

        # Instantiate the cached model
        drl_model = instantiate_model(data, initial_capital)
        available_symbols = sorted(drl_model.df['Symbol'].unique())
        
        # Stock Selection
        selected_symbol = st.selectbox(
            "Select Stock for Autonomous Trading",
            options=available_symbols,
            index=available_symbols.index('RELIANCE') if 'RELIANCE' in available_symbols else 0,
            help="The DRL agent will run its policy against this stock's data."
        )
        
        st.markdown("---")
        
        # --- Execute Button ---
        if st.button("▶️ RUN DRL SIMULATION", type="primary", use_container_width=True):
            st.session_state['run_simulation'] = True
        
        st.markdown(f"**Loaded {len(data)} Stock Records.**")


    # --- 2. Main Content Area: Results Display ---
    
    if st.session_state.get('run_simulation', False):
        st.subheader(f"🧠 DRL Autonomous Decision for **{selected_symbol}**")
        
        # Simulate loading time for professional feel
        progress_text = "Analyzing market state and calculating optimal policy..."
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            time.sleep(0.005)
            my_bar.progress(percent_complete + 1, text=progress_text)
        my_bar.empty() # Clear the progress bar after completion

        # Run the backtest using the imported model
        metrics, action_log = drl_model.run_policy_and_backtest(selected_symbol)
        
        # --- Display Metrics ---
        if metrics:
            
            st.markdown("### Decision Summary")
            
            col_signal, col_score, col_ltp, col_trend = st.columns(4)
            
            with col_signal:
                st.metric(
                    "Autonomous Signal", 
                    metrics['Final_Signal'], 
                    delta_color="off" 
                )
            
            with col_score:
                # Highlight DRL Score based on magnitude
                delta_score = f"{metrics['DRL_Score']:,.4f}"
                score_color = 'off'
                if metrics['DRL_Score'] > 0.1: score_color = 'inverse'
                elif metrics['DRL_Score'] < -0.1: score_color = 'normal'
                
                st.metric(
                    "DRL Confidence Score", 
                    delta_score,
                    delta_color=score_color,
                    help="A composite score combining trend, momentum, and volume. Higher absolute value means higher policy confidence."
                )
            
            with col_ltp:
                st.metric("Current LTP (₹)", f"₹ {metrics['LTP']:,.2f}")
            
            with col_trend:
                # Display 1-Year and 30-Day Trends
                trend_pct = metrics['365d % Chng']
                st.metric(
                    "1-Year Trend %", 
                    f"{trend_pct:,.2f} %",
                    delta=f"{metrics['30d % Chng']:,.2f} % (30-Day)"
                )
                
            st.markdown("---")

            # --- Autonomous Action Log (User Experience: Clear Feedback) ---
            st.markdown("### 📝 Trade Execution & Risk Log")
            
            # Use a container to group the log output
            log_container = st.container(border=True)
            for line in action_log:
                if "**Autonomous BUY Signal**" in line or "**Shares Purchased**" in line or "Profit" in line:
                    log_container.success(f"✅ {line}")
                elif "**Autonomous SELL Signal**" in line or "Risk Managed" in line:
                    log_container.error(f"🛑 {line}")
                else:
                    log_container.info(f"ℹ️ {line}")
            
            st.markdown("---")
            
            # --- Simulated Performance (Professional Metrics) ---
            st.markdown("### 📊 Simulated Performance & Risk")
            
            perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

            with perf_col1:
                st.metric(
                    "Simulated Profit/Loss (₹)",
                    f"₹ {metrics['Simulated_Profit']:,.2f}",
                    delta=f"{metrics['Cumulative_Return'] * 100:.2f} % Return"
                )
            
            with perf_col2:
                st.metric(
                    "Final Portfolio Value (₹)",
                    f"₹ {metrics['Final_Capital']:,.2f}"
                )
            
            with perf_col3:
                # Sharpe Ratio: Key Risk-Adjusted Metric
                st.metric(
                    "Sharpe Ratio",
                    f"{metrics['Sharpe_Ratio']:.4f}",
                    help="Risk-adjusted return: Higher is better (simplified volatility assumed)."
                )
            
            with perf_col4:
                # Download Button for Results (Professional Feature)
                metrics_df = pd.DataFrame([metrics])
                csv = metrics_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Metrics CSV",
                    data=csv,
                    file_name=f'DRL_Metrics_{selected_symbol}.csv',
                    mime='text/csv',
                    key='download-csv',
                    help="Export the key performance metrics for this simulation."
                )

            # --- Technical Deep Dive (Advanced Features) ---
            st.markdown("---")
            with st.expander("🔬 DRL Feature Input and Raw Data"):
                st.markdown("**Raw Features used for DRL Policy Calculation:**")
                
                # Filter the specific stock data and select relevant columns
                raw_data_row = drl_model.df[drl_model.df['Symbol'] == selected_symbol].iloc[0]
                
                features = {
                    'LTP': raw_data_row['LTP'],
                    '30 d % chng': raw_data_row['30 d % chng'],
                    '365 d % chng': raw_data_row['365 d % chng'],
                    'Volume (lacs)': raw_data_row['Volume (lacs)'],
                    'Volume_Ratio': raw_data_row['Volume_Ratio'],
                    'Momentum_Score': raw_data_row['Momentum_Score'],
                }
                st.json(features)
                
                st.markdown("**Full Processed Data Snippet:**")
                st.dataframe(drl_model.df.head(), use_container_width=True)

        else:
            st.error(f"Simulation Failed: Could not retrieve valid data or run the policy for {selected_symbol}")
            
    else:
        st.info("The DRL Engine is on standby. Configure parameters in the sidebar and click RUN.")

if __name__ == "__main__":
    # Initialize session state for running the simulation
    if 'run_simulation' not in st.session_state:
        st.session_state['run_simulation'] = False
        
    main()
