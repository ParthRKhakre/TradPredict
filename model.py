import pandas as pd
import numpy as np
import io

def load_data(file_content):
    """Utility function to load and decode the CSV content from file_content (bytes)."""
    s = str(file_content, 'utf-8')
    data = io.StringIO(s)
    return pd.read_csv(data)

class DRLTradingModel:
    """
    Simulates the core logic of a Deep Reinforcement Learning (DRL) Agent.
    This class handles data preparation, policy signal generation, and backtesting.
    """

    def __init__(self, stock_df, initial_capital=100000):
        self.df = self._process_data(stock_df)
        self.initial_capital = initial_capital
        # Daily equivalent of 5% risk-free rate (approx. 252 trading days)
        self.risk_free_rate = 0.05 / 252

    def _process_data(self, df):
        """Cleans and prepares the uploaded stock data."""
        # Convert necessary columns from object (string with commas) to float
        cols_to_clean = ['Open', 'High', 'Low', 'LTP', 'Turnover (crs.)', '52w H', '52w L']
        for col in cols_to_clean:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '', regex=False).astype(float)
        
        # Calculate a simple momentum feature (similar to RSI/MACD input)
        df['Momentum_Score'] = df.apply(
            lambda x: x['30 d % chng'] / x['365 d % chng'] if abs(x['365 d % chng']) > 0.01 else 0, axis=1
        )
        df['Volume_Ratio'] = df['Volume (lacs)'] / df['Volume (lacs)'].mean()
        
        return df

    def _generate_signal(self, row):
        """
        Mock DRL Policy: Calculates a 'Decision Score' based on features.
        """
        
        # Coefficients derived from a hypothetical learned policy:
        score = (
            0.5 * row['365 d % chng'] / 100 +
            0.3 * row['30 d % chng'] / 100 +
            0.2 * (row['Volume_Ratio'] - 1)  # Normalized Volume
        )
        
        # Action determination based on the autonomous score
        if score > 0.15:
            return 'BUY'
        elif score < -0.15:
            return 'SELL' # Shorting not allowed, so this means sell holdings or avoid
        else:
            return 'HOLD'

    def run_policy_and_backtest(self, symbol):
        """
        Runs the simulated DRL policy and executes the backtest for a single stock.
        """
        
        if symbol not in self.df['Symbol'].values:
            return None, "Stock symbol not found in data."

        stock_data = self.df[self.df['Symbol'] == symbol].iloc[0]
        
        # 1. Generate Autonomous Signal
        signal = self._generate_signal(stock_data)
        drl_score = (
            0.5 * stock_data['365 d % chng'] / 100 +
            0.3 * stock_data['30 d % chng'] / 100 +
            0.2 * (stock_data['Volume_Ratio'] - 1)
        )
        
        # 2. Simulate Transaction and Profit
        ltp = stock_data['LTP']
        
        # Use a simplified potential return based on the 30-day change for the simulation
        hypothetical_return_pct = stock_data['30 d % chng'] / 100
        
        simulated_profit = 0.0
        cumulative_return = 0.0
        final_capital = self.initial_capital
        
        if signal == 'BUY':
            shares_to_buy = int(self.initial_capital * 0.2 / ltp) # Allocate 20% of capital
            investment = shares_to_buy * ltp
            simulated_profit = investment * hypothetical_return_pct
            
            final_capital = self.initial_capital + simulated_profit
            cumulative_return = simulated_profit / self.initial_capital
            
            action_log = (
                f"**Autonomous BUY Signal** triggered by DRL Score: {drl_score:,.4f}.",
                f"Shares Purchased: {shares_to_buy}. Initial Investment: ₹ {investment:,.2f}.",
                f"Simulated Profit (using 30d return as proxy): ₹ {simulated_profit:,.2f}."
            )
            
        elif signal == 'SELL':
            if hypothetical_return_pct < 0 and drl_score < 0:
                  action_log = (
                      f"**Autonomous SELL Signal** triggered by DRL Score: {drl_score:,.4f}.",
                      "Decision: HOLD CASH. **Risk Managed: Avoided a potential loss**.",
                  )
            else:
                action_log = (
                    f"**Autonomous SELL Signal** triggered by DRL Score: {drl_score:,.4f}.",
                    "Decision: HOLD CASH (Neutral Stance).",
                )
                
        else: # HOLD
            action_log = (
                f"**Autonomous HOLD Signal** triggered by DRL Score: {drl_score:,.4f}.",
                "Decision: HOLD CASH (Waiting for clearer trend).",
            )

        # Calculate a simple Sharpe Ratio
        sharpe_ratio = (cumulative_return - self.risk_free_rate) / 0.1 # Mock volatility
        
        metrics = {
            'Symbol': symbol,
            'LTP': ltp,
            '30d % Chng': stock_data['30 d % chng'],
            '365d % Chng': stock_data['365 d % chng'],
            'DRL_Score': drl_score,
            'Final_Signal': signal,
            'Initial_Capital': self.initial_capital,
            'Final_Capital': final_capital,
            'Cumulative_Return': cumulative_return,
            'Sharpe_Ratio': sharpe_ratio if cumulative_return > 0 else 0.0,
            'Simulated_Profit': simulated_profit
        }
        
        return metrics, action_log

if __name__ == "__main__":
    print("This file contains the core DRL model logic and utility functions.")
    print("Run app.py to start the Streamlit frontend.")