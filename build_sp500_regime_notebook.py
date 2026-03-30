from pathlib import Path
from textwrap import dedent

import nbformat as nbf


def markdown_cell(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip("\n"))


def code_cell(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip("\n"))


abstract_text = (
    "Abstract: Financial markets are complex, non-stationary systems driven not only by historical "
    "price action but also by shifting macroeconomic sentiment and policy uncertainty. Traditional "
    "time-series forecasting models often fail to capture sudden regime shifts induced by external "
    "economic shocks. This study proposes a novel, regime-aware deep learning framework to predict "
    "the S&P 500 index. By fusing traditional market data (OHLCV) with structured macroeconomic "
    "sentiment proxies from the FRED database—specifically the University of Michigan Consumer "
    "Sentiment Index and the News-based Equity Market Volatility Tracker—the model learns to "
    "contextualize price movements within broader economic regimes. Furthermore, the framework "
    "moves beyond standard statistical loss functions by evaluating predictive performance through "
    "financially meaningful metrics, including Directional Accuracy, simulated Sharpe Ratios, and "
    "Maximum Drawdown."
)


cells = [
    markdown_cell(abstract_text),
    code_cell(
        """
        # Runtime bootstrap for Google Colab or local notebook environments.
        import importlib.util
        import subprocess
        import sys

        required_modules = {
            "yfinance": "yfinance",
            "pandas_datareader": "pandas_datareader",
            "fredapi": "fredapi",
            "keras": "keras>=3.0.0",
            "tensorflow": "tensorflow",
            "sklearn": "scikit-learn",
            "seaborn": "seaborn",
        }

        for module_name, package_name in required_modules.items():
            if importlib.util.find_spec(module_name) is None:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package_name])

        print("Dependency check completed.")
        """
    ),
    markdown_cell(
        r"""
        # Section 1 - Environment Setup and Reproducibility

        This section fixes the experiment configuration and random seeds to reduce stochastic variance across runs.  
        The forecasting target is the one-day-ahead log return:

        \[
        r_{t+1} = \log\left(\frac{P_{t+1}}{P_t}\right)
        \]

        A deterministic setup improves comparability between model variants and hyperparameter studies.
        """
    ),
    code_cell(
        """
        # Section 1 code: imports, reproducibility controls, and global settings.
        import os
        import random
        import warnings
        from dataclasses import dataclass

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        os.environ.setdefault("KERAS_BACKEND", "tensorflow")
        import keras
        import yfinance as yf
        from fredapi import Fred
        from keras import callbacks, layers, models, ops
        from pandas_datareader import data as pdr
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        from sklearn.preprocessing import StandardScaler

        warnings.filterwarnings("ignore")
        sns.set_theme(style="whitegrid")

        SEED = 42
        os.environ["PYTHONHASHSEED"] = str(SEED)
        random.seed(SEED)
        np.random.seed(SEED)
        keras.utils.set_random_seed(SEED)


        @dataclass
        class ExperimentConfig:
            market_ticker: str = "^GSPC"
            start_date: str = "2000-01-01"
            end_date: str = "2026-01-01"
            sequence_length: int = 60
            train_ratio: float = 0.70
            validation_ratio: float = 0.15
            annualization_factor: int = 252


        config = ExperimentConfig()
        print(config)
        print("Keras backend:", keras.backend.backend())
        print("Keras version:", keras.__version__)
        """
    ),
    markdown_cell(
        """
        # Section 2 - Multimodal Data Acquisition

        We collect two synchronized information channels:

        - Market channel: S&P 500 OHLCV data from Yahoo Finance (`^GSPC`).
        - Macro-sentiment channel from FRED: `UMCSENT`, `EMVMACROBUS`, `FEDFUNDS`, `DGS10`, `CPIAUCSL`, and `UNRATE`.
        - Authentication mode: the pipeline uses `FRED_API_KEY` when available and falls back to public access otherwise.

        This structure explicitly combines endogenous price dynamics with exogenous macroeconomic and sentiment conditions.
        """
    ),
    code_cell(
        """
        # Section 2 code: download market and macro data.
        market_raw = yf.download(
            config.market_ticker,
            start=config.start_date,
            end=config.end_date,
            auto_adjust=False,
            progress=False,
        )

        if isinstance(market_raw.columns, pd.MultiIndex):
            market_raw.columns = [column_tuple[0] for column_tuple in market_raw.columns]

        market_raw = market_raw[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].dropna()
        market_raw.index = pd.to_datetime(market_raw.index).tz_localize(None)

        fred_series_codes = {
            "consumer_sentiment": "UMCSENT",
            "macro_news_volatility": "EMVMACROBUS",
            "federal_funds_rate": "FEDFUNDS",
            "ten_year_treasury": "DGS10",
            "consumer_price_index": "CPIAUCSL",
            "unemployment_rate": "UNRATE",
        }

        fred_api_key = os.getenv("FRED_API_KEY", "").strip()
        if fred_api_key:
            fred_client = Fred(api_key=fred_api_key)
            macro_series_map = {}
            for feature_name, fred_code in fred_series_codes.items():
                macro_series_map[feature_name] = fred_client.get_series(
                    fred_code,
                    observation_start=config.start_date,
                    observation_end=config.end_date,
                )
            macro_raw = pd.DataFrame(macro_series_map)
            macro_raw.index = pd.to_datetime(macro_raw.index)
            macro_raw = macro_raw.sort_index()
            data_source_label = "FRED authenticated API (fredapi)"
        else:
            macro_raw = pdr.DataReader(
                list(fred_series_codes.values()),
                "fred",
                config.start_date,
                config.end_date,
            )
            macro_raw = macro_raw.rename(columns={value: key for key, value in fred_series_codes.items()})
            macro_raw.index = pd.to_datetime(macro_raw.index)
            data_source_label = "FRED public access (pandas_datareader)"

        print("Market data shape:", market_raw.shape)
        print("Macro data shape:", macro_raw.shape)
        print("FRED source mode:", data_source_label)
        display(market_raw.head())
        display(macro_raw.head())
        """
    ),
    markdown_cell(
        r"""
        # Section 3 - Feature Engineering for Market and Macro Signals

        For the market channel, we engineer trend, momentum, and risk descriptors (moving-average gaps, RSI, MACD, ATR, and realized volatility).  
        For the macro channel, we derive interpretable transforms:

        \[
        \pi^{YoY}_t = 100 \times \left(\frac{CPI_t}{CPI_{t-12}} - 1\right), \quad
        Spread_t = DGS10_t - FEDFUNDS_t
        \]

        This heterogeneous representation is designed to support regime-sensitive forecasting.
        """
    ),
    code_cell(
        """
        # Section 3 code: create market and macro engineered features.
        market_df = market_raw.copy()

        market_df["log_return_1d"] = np.log(market_df["Adj Close"]).diff()
        market_df["return_5d"] = market_df["Adj Close"].pct_change(5)

        market_df["sma_10"] = market_df["Adj Close"].rolling(10).mean()
        market_df["sma_20"] = market_df["Adj Close"].rolling(20).mean()
        market_df["sma_50"] = market_df["Adj Close"].rolling(50).mean()
        market_df["sma_10_gap"] = market_df["Adj Close"] / market_df["sma_10"] - 1.0
        market_df["sma_20_gap"] = market_df["Adj Close"] / market_df["sma_20"] - 1.0
        market_df["sma_50_gap"] = market_df["Adj Close"] / market_df["sma_50"] - 1.0

        delta = market_df["Adj Close"].diff()
        up_moves = delta.clip(lower=0)
        down_moves = -delta.clip(upper=0)
        average_up = up_moves.ewm(alpha=1 / 14, adjust=False).mean()
        average_down = down_moves.ewm(alpha=1 / 14, adjust=False).mean()
        relative_strength = average_up / (average_down + 1e-12)
        market_df["rsi_14"] = 100 - (100 / (1 + relative_strength))

        ema_fast = market_df["Adj Close"].ewm(span=12, adjust=False).mean()
        ema_slow = market_df["Adj Close"].ewm(span=26, adjust=False).mean()
        market_df["macd_line"] = ema_fast - ema_slow
        market_df["macd_signal"] = market_df["macd_line"].ewm(span=9, adjust=False).mean()
        market_df["macd_hist"] = market_df["macd_line"] - market_df["macd_signal"]

        high_low_range = market_df["High"] - market_df["Low"]
        high_prev_close_range = (market_df["High"] - market_df["Close"].shift(1)).abs()
        low_prev_close_range = (market_df["Low"] - market_df["Close"].shift(1)).abs()
        true_range = pd.concat(
            [high_low_range, high_prev_close_range, low_prev_close_range],
            axis=1,
        ).max(axis=1)
        market_df["atr_14"] = true_range.rolling(14).mean()

        market_df["realized_vol_21"] = market_df["log_return_1d"].rolling(21).std() * np.sqrt(252)
        market_df["target_return"] = market_df["log_return_1d"].shift(-1)

        macro_df = macro_raw.copy()
        macro_df["inflation_yoy"] = macro_df["consumer_price_index"].pct_change(12) * 100.0
        macro_df["term_spread"] = macro_df["ten_year_treasury"] - macro_df["federal_funds_rate"]
        macro_df["macro_sentiment_spread"] = macro_df["consumer_sentiment"] - macro_df["macro_news_volatility"]

        market_feature_columns = [
            "log_return_1d",
            "return_5d",
            "sma_10_gap",
            "sma_20_gap",
            "sma_50_gap",
            "rsi_14",
            "macd_line",
            "macd_signal",
            "macd_hist",
            "atr_14",
            "realized_vol_21",
            "Volume",
        ]

        macro_feature_columns = [
            "consumer_sentiment",
            "macro_news_volatility",
            "federal_funds_rate",
            "ten_year_treasury",
            "unemployment_rate",
            "inflation_yoy",
            "term_spread",
            "macro_sentiment_spread",
        ]

        print("Number of engineered market features:", len(market_feature_columns))
        print("Number of engineered macro features:", len(macro_feature_columns))
        """
    ),
    markdown_cell(
        r"""
        # Section 4 - Temporal Alignment Without Look-Ahead Bias

        Macroeconomic data arrive at mixed frequencies and are not updated at every market close.  
        We align macro data to the trading calendar via forward fill and then apply a one-business-day lag:

        \[
        \tilde{m}_t = m_{\max(\tau \le t-1)}
        \]

        This conservative convention prevents leakage from same-day macro updates that may not have been observable at decision time.
        """
    ),
    code_cell(
        """
        # Section 4 code: align macro data to daily market dates with a conservative lag.
        macro_daily = macro_df.reindex(market_df.index).ffill()
        macro_daily_lagged = macro_daily.shift(1)

        aligned_df = pd.concat(
            [
                market_df[["Adj Close", "target_return"] + market_feature_columns],
                macro_daily_lagged[macro_feature_columns],
            ],
            axis=1,
        ).dropna()

        aligned_df = aligned_df.rename(columns={"Adj Close": "adj_close"})
        aligned_df.index.name = "date"

        print("Aligned dataset shape:", aligned_df.shape)
        display(aligned_df.head())
        display(aligned_df.tail())
        """
    ),
    markdown_cell(
        r"""
        # Section 5 - Time-Aware Splitting, Scaling, and 3D Sequence Construction

        We use chronological splitting (train, validation, test) to preserve causality.  
        Feature normalization is fit only on the training window and then applied forward in time.

        For sequence length \(L\), each sample is:

        \[
        X_t^{market} \in \mathbb{R}^{L \times d_m}, \quad
        X_t^{macro} \in \mathbb{R}^{L \times d_c}, \quad
        y_t = r_{t+1}
        \]

        The output is a pair of synchronized 3D tensors for multimodal learning.
        """
    ),
    code_cell(
        """
        # Section 5 code: chronological split, train-only scaling, and sequence creation.
        total_rows = len(aligned_df)
        train_end_index = int(total_rows * config.train_ratio)
        validation_end_index = int(total_rows * (config.train_ratio + config.validation_ratio))

        train_df = aligned_df.iloc[:train_end_index].copy()
        validation_df = aligned_df.iloc[train_end_index:validation_end_index].copy()
        test_df = aligned_df.iloc[validation_end_index:].copy()

        for split_name, split_frame in {
            "train": train_df,
            "validation": validation_df,
            "test": test_df,
        }.items():
            if len(split_frame) <= config.sequence_length:
                raise ValueError(
                    f"Insufficient rows in the {split_name} split for sequence length {config.sequence_length}."
                )

        market_scaler = StandardScaler()
        macro_scaler = StandardScaler()

        train_df[market_feature_columns] = market_scaler.fit_transform(train_df[market_feature_columns])
        validation_df[market_feature_columns] = market_scaler.transform(validation_df[market_feature_columns])
        test_df[market_feature_columns] = market_scaler.transform(test_df[market_feature_columns])

        train_df[macro_feature_columns] = macro_scaler.fit_transform(train_df[macro_feature_columns])
        validation_df[macro_feature_columns] = macro_scaler.transform(validation_df[macro_feature_columns])
        test_df[macro_feature_columns] = macro_scaler.transform(test_df[macro_feature_columns])


        def create_multimodal_sequences(input_frame: pd.DataFrame, sequence_length: int):
            market_tensors = []
            macro_tensors = []
            targets = []
            target_dates = []

            for end_row in range(sequence_length, len(input_frame)):
                start_row = end_row - sequence_length
                window_frame = input_frame.iloc[start_row:end_row]
                target_row = input_frame.iloc[end_row]

                market_tensors.append(window_frame[market_feature_columns].values)
                macro_tensors.append(window_frame[macro_feature_columns].values)
                targets.append(target_row["target_return"])
                target_dates.append(input_frame.index[end_row])

            return (
                np.asarray(market_tensors, dtype=np.float32),
                np.asarray(macro_tensors, dtype=np.float32),
                np.asarray(targets, dtype=np.float32).reshape(-1, 1),
                pd.DatetimeIndex(target_dates),
            )


        X_market_train, X_macro_train, y_train, train_dates = create_multimodal_sequences(
            train_df, config.sequence_length
        )
        X_market_validation, X_macro_validation, y_validation, validation_dates = create_multimodal_sequences(
            validation_df, config.sequence_length
        )
        X_market_test, X_macro_test, y_test, test_dates = create_multimodal_sequences(
            test_df, config.sequence_length
        )

        print("Train tensors:", X_market_train.shape, X_macro_train.shape, y_train.shape)
        print("Validation tensors:", X_market_validation.shape, X_macro_validation.shape, y_validation.shape)
        print("Test tensors:", X_market_test.shape, X_macro_test.shape, y_test.shape)
        """
    ),
    markdown_cell(
        r"""
        # Section 6 - Regime-Aware Hybrid Deep Learning Architecture

        The network has two branches:

        - Market branch: `Conv1D -> LSTM -> Temporal Attention`
        - Macro branch: `LSTM -> Temporal Attention`

        A regime gate is learned from macro context and used to modulate market context before fusion:

        \[
        g = \sigma(W_g z^{macro} + b_g), \quad
        \hat{z}^{market} = g \odot z^{market}
        \]

        The fused representation is mapped to a one-step-ahead return prediction.
        """
    ),
    code_cell(
        """
        # Section 6 code: define a dual-branch, regime-aware neural architecture.
        def temporal_attention_pooling(sequence_tensor, block_name: str):
            attention_scores = layers.Dense(
                1,
                activation="tanh",
                name=f"{block_name}_attention_scores",
            )(sequence_tensor)
            attention_weights = layers.Softmax(axis=1, name=f"{block_name}_attention_weights")(attention_scores)
            weighted_sequence = layers.Multiply(name=f"{block_name}_weighted_sequence")(
                [sequence_tensor, attention_weights]
            )
            context_vector = layers.Lambda(
                lambda tensor: ops.sum(tensor, axis=1),
                name=f"{block_name}_context_vector",
            )(weighted_sequence)
            return context_vector


        def build_regime_aware_model(
            sequence_length: int,
            market_feature_count: int,
            macro_feature_count: int,
        ) -> models.Model:
            market_input = layers.Input(
                shape=(sequence_length, market_feature_count),
                name="market_input",
            )
            market_branch = layers.Conv1D(
                filters=32,
                kernel_size=3,
                padding="causal",
                activation="relu",
                name="market_conv1d",
            )(market_input)
            market_branch = layers.Dropout(0.20, name="market_dropout_1")(market_branch)
            market_branch = layers.LSTM(
                units=64,
                return_sequences=True,
                name="market_lstm",
            )(market_branch)
            market_branch = layers.Dropout(0.20, name="market_dropout_2")(market_branch)
            market_context = temporal_attention_pooling(market_branch, block_name="market")

            macro_input = layers.Input(
                shape=(sequence_length, macro_feature_count),
                name="macro_input",
            )
            macro_branch = layers.LSTM(
                units=32,
                return_sequences=True,
                name="macro_lstm",
            )(macro_input)
            macro_branch = layers.Dropout(0.20, name="macro_dropout")(macro_branch)
            macro_context = temporal_attention_pooling(macro_branch, block_name="macro")

            market_projection = layers.Dense(64, activation="tanh", name="market_projection")(market_context)
            regime_gate = layers.Dense(64, activation="sigmoid", name="regime_gate")(macro_context)
            gated_market = layers.Multiply(name="gated_market_context")([market_projection, regime_gate])

            fusion_tensor = layers.Concatenate(name="fusion_layer")(
                [gated_market, macro_context, regime_gate]
            )
            fusion_tensor = layers.Dense(64, activation="relu", name="fusion_dense")(fusion_tensor)
            fusion_tensor = layers.Dropout(0.30, name="fusion_dropout")(fusion_tensor)

            output = layers.Dense(1, name="predicted_next_day_return")(fusion_tensor)

            model = models.Model(inputs=[market_input, macro_input], outputs=output, name="RegimeAwareFusionModel")
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                loss="mse",
                metrics=[keras.metrics.RootMeanSquaredError(name="rmse"), "mae"],
            )
            return model


        model = build_regime_aware_model(
            sequence_length=config.sequence_length,
            market_feature_count=len(market_feature_columns),
            macro_feature_count=len(macro_feature_columns),
        )

        model.summary()
        """
    ),
    markdown_cell(
        """
        # Section 7 - Training Protocol and Regularization

        We optimize mean squared error and monitor validation loss.  
        Regularization controls include:

        - Dropout across feature extractors and fusion head.
        - Early stopping with best-weight restoration.
        - Learning-rate reduction on validation plateaus.

        This protocol is intended to stabilize training under noisy, non-stationary financial regimes.
        """
    ),
    code_cell(
        """
        # Section 7 code: train the model with robust regularization callbacks.
        training_callbacks = [
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=15,
                restore_best_weights=True,
                verbose=1,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=7,
                min_lr=1e-5,
                verbose=1,
            ),
        ]

        history = model.fit(
            [X_market_train, X_macro_train],
            y_train,
            validation_data=([X_market_validation, X_macro_validation], y_validation),
            epochs=200,
            batch_size=32,
            callbacks=training_callbacks,
            verbose=1,
        )

        history_frame = pd.DataFrame(history.history)
        display(history_frame.tail())

        figure, axes = plt.subplots(1, 2, figsize=(14, 4))
        history_frame[["loss", "val_loss"]].plot(ax=axes[0], title="Training and Validation Loss")
        history_frame[["rmse", "val_rmse"]].plot(ax=axes[1], title="Training and Validation RMSE")
        plt.tight_layout()
        plt.show()

        y_pred_test = model.predict([X_market_test, X_macro_test], verbose=0).reshape(-1)
        y_true_test = y_test.reshape(-1)

        test_rmse = float(np.sqrt(mean_squared_error(y_true_test, y_pred_test)))
        test_mae = float(mean_absolute_error(y_true_test, y_pred_test))

        print(f"Test RMSE (log-return scale): {test_rmse:.6f}")
        print(f"Test MAE  (log-return scale): {test_mae:.6f}")
        """
    ),
    markdown_cell(
        r"""
        # Section 8 - Financially Relevant Evaluation

        Beyond point forecast error, we evaluate trading relevance.

        Directional Accuracy:

        \[
        DA = \frac{1}{N}\sum_{t=1}^{N} \mathbf{1}\left[\operatorname{sign}(\hat{r}_{t+1}) = \operatorname{sign}(r_{t+1})\right]
        \]

        Trading rule: go long if \(\hat{r}_{t+1} > 0\), otherwise allocate to cash.  
        We report cumulative return, annualized Sharpe ratio, and maximum drawdown, benchmarked against buy-and-hold.
        """
    ),
    code_cell(
        """
        # Section 8 code: compute directional and strategy-based performance metrics.
        def annualized_sharpe_ratio(simple_returns: pd.Series, annualization_factor: int = 252) -> float:
            volatility = simple_returns.std(ddof=0)
            if volatility == 0 or np.isnan(volatility):
                return np.nan
            return float(np.sqrt(annualization_factor) * simple_returns.mean() / volatility)


        def compute_max_drawdown(simple_returns: pd.Series) -> float:
            equity_curve = (1 + simple_returns).cumprod()
            rolling_peak = equity_curve.cummax()
            drawdown_series = equity_curve / rolling_peak - 1.0
            return float(drawdown_series.min())


        evaluation_df = pd.DataFrame(
            {
                "realized_log_return": y_true_test,
                "predicted_log_return": y_pred_test,
            },
            index=test_dates,
        )

        evaluation_df["realized_simple_return"] = np.exp(evaluation_df["realized_log_return"]) - 1.0
        evaluation_df["predicted_simple_return"] = np.exp(evaluation_df["predicted_log_return"]) - 1.0
        evaluation_df["signal_long_only"] = (evaluation_df["predicted_log_return"] > 0).astype(int)
        evaluation_df["strategy_simple_return"] = (
            evaluation_df["signal_long_only"] * evaluation_df["realized_simple_return"]
        )
        evaluation_df["buy_hold_simple_return"] = evaluation_df["realized_simple_return"]

        directional_accuracy = float(
            (
                np.sign(evaluation_df["predicted_log_return"])
                == np.sign(evaluation_df["realized_log_return"])
            ).mean()
        )

        strategy_cumulative_return = float((1 + evaluation_df["strategy_simple_return"]).prod() - 1.0)
        buy_hold_cumulative_return = float((1 + evaluation_df["buy_hold_simple_return"]).prod() - 1.0)
        strategy_sharpe = annualized_sharpe_ratio(
            evaluation_df["strategy_simple_return"], config.annualization_factor
        )
        buy_hold_sharpe = annualized_sharpe_ratio(
            evaluation_df["buy_hold_simple_return"], config.annualization_factor
        )
        strategy_max_drawdown = compute_max_drawdown(evaluation_df["strategy_simple_return"])
        buy_hold_max_drawdown = compute_max_drawdown(evaluation_df["buy_hold_simple_return"])

        metrics_table = pd.DataFrame(
            [
                {"Metric": "Test RMSE (log returns)", "Strategy": test_rmse, "Buy and Hold": np.nan},
                {"Metric": "Test MAE (log returns)", "Strategy": test_mae, "Buy and Hold": np.nan},
                {"Metric": "Directional Accuracy", "Strategy": directional_accuracy, "Buy and Hold": np.nan},
                {
                    "Metric": "Cumulative Return",
                    "Strategy": strategy_cumulative_return,
                    "Buy and Hold": buy_hold_cumulative_return,
                },
                {
                    "Metric": "Annualized Sharpe Ratio",
                    "Strategy": strategy_sharpe,
                    "Buy and Hold": buy_hold_sharpe,
                },
                {
                    "Metric": "Maximum Drawdown",
                    "Strategy": strategy_max_drawdown,
                    "Buy and Hold": buy_hold_max_drawdown,
                },
            ]
        )

        display(metrics_table)

        equity_curve_df = pd.DataFrame(index=evaluation_df.index)
        equity_curve_df["Regime-Aware Strategy"] = (1 + evaluation_df["strategy_simple_return"]).cumprod()
        equity_curve_df["Buy and Hold"] = (1 + evaluation_df["buy_hold_simple_return"]).cumprod()

        plt.figure(figsize=(12, 5))
        plt.plot(equity_curve_df.index, equity_curve_df["Regime-Aware Strategy"], label="Regime-Aware Strategy")
        plt.plot(equity_curve_df.index, equity_curve_df["Buy and Hold"], label="Buy and Hold", alpha=0.8)
        plt.title("Out-of-Sample Equity Curves")
        plt.ylabel("Growth of $1")
        plt.legend()
        plt.tight_layout()
        plt.show()
        """
    ),
    markdown_cell(
        """
        # Section 9 - Results Packaging and Reproducible Outputs

        This final section exports model artifacts for manuscript integration and reproducibility archives.  
        The exported tables can be directly reused in results sections, appendices, and robustness comparisons.
        """
    ),
    code_cell(
        """
        # Section 9 code: export artifacts and print concise summary statistics.
        metrics_table.to_csv("sp500_regime_aware_metrics.csv", index=False)
        evaluation_df.to_csv("sp500_regime_aware_test_results.csv")
        model.save("sp500_regime_aware_model.keras")

        print("Saved files:")
        print("- sp500_regime_aware_metrics.csv")
        print("- sp500_regime_aware_test_results.csv")
        print("- sp500_regime_aware_model.keras")

        summary_series = pd.Series(
            {
                "test_rmse_log_return": test_rmse,
                "test_mae_log_return": test_mae,
                "directional_accuracy": directional_accuracy,
                "strategy_cumulative_return": strategy_cumulative_return,
                "buy_hold_cumulative_return": buy_hold_cumulative_return,
                "strategy_sharpe": strategy_sharpe,
                "buy_hold_sharpe": buy_hold_sharpe,
                "strategy_max_drawdown": strategy_max_drawdown,
                "buy_hold_max_drawdown": buy_hold_max_drawdown,
            }
        )
        display(summary_series.to_frame(name="value"))
        """
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
        "colab": {
            "provenance": [],
            "name": "sp500_regime_aware_fusion_colab.ipynb",
        },
    },
)

output_path = Path("sp500_regime_aware_fusion_colab.ipynb")
with output_path.open("w", encoding="utf-8") as file_handle:
    nbf.write(notebook, file_handle)

nbf.validate(notebook)

print(f"Notebook created: {output_path.resolve()}")
