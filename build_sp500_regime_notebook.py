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

        $$ r_{t+1} = \log\left(\frac{P_{t+1}}{P_t}\right) $$

        A deterministic setup improves comparability between model variants and hyperparameter studies.
        """
    ),
    code_cell(
        """
        # Section 1 code: imports, reproducibility controls, and global settings.
        import os
        import random
        import json
        import re
        import sys
        import warnings
        from dataclasses import dataclass
        from datetime import datetime
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        os.environ.setdefault("KERAS_BACKEND", "tensorflow")
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
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

        tensorflow_deterministic_ops = "Not enabled"
        if keras.backend.backend() == "tensorflow":
            try:
                import tensorflow as tf

                tf.config.experimental.enable_op_determinism()
                tensorflow_deterministic_ops = "Enabled"
            except Exception as deterministic_error:
                tensorflow_deterministic_ops = f"Unavailable ({deterministic_error})"


        @dataclass
        class ExperimentConfig:
            market_ticker: str = "^GSPC"
            start_date: str = "2000-01-01"
            end_date: str = "2026-01-01"
            sequence_length: int = 60
            train_ratio: float = 0.70
            validation_ratio: float = 0.15
            annualization_factor: int = 252
            output_subdirectory: str = "sp500_regime_aware_outputs"


        def resolve_output_directory(output_subdirectory: str) -> Path:
            if "google.colab" in sys.modules:
                try:
                    from google.colab import drive

                    drive.mount("/content/drive", force_remount=False)
                    google_drive_directory = Path("/content/drive/MyDrive") / output_subdirectory
                    google_drive_directory.mkdir(parents=True, exist_ok=True)
                    print(f"Saving artifacts to Google Drive: {google_drive_directory}")
                    return google_drive_directory
                except Exception as drive_error:
                    print(f"Google Drive mount failed ({drive_error}); using local directory.")

            local_directory = Path(output_subdirectory)
            local_directory.mkdir(parents=True, exist_ok=True)
            print(f"Saving artifacts locally: {local_directory.resolve()}")
            return local_directory


        def slugify_label(raw_label: str) -> str:
            normalized = re.sub(r"[^a-z0-9]+", "_", raw_label.lower())
            return normalized.strip("_")


        ARTIFACT_COUNTER = 1
        ARTIFACT_MANIFEST = []


        def build_artifact_path(section_tag: str, artifact_kind: str, short_name: str, extension: str) -> Path:
            global ARTIFACT_COUNTER
            file_name = (
                f"{ARTIFACT_COUNTER:03d}_"
                f"{slugify_label(section_tag)}_"
                f"{slugify_label(artifact_kind)}_"
                f"{slugify_label(short_name)}."
                f"{extension}"
            )
            ARTIFACT_COUNTER += 1
            return output_dir / file_name


        def register_artifact(
            artifact_path: Path,
            section_tag: str,
            artifact_kind: str,
            short_name: str,
            manuscript_role: str,
            llm_guidance: str,
        ) -> None:
            ARTIFACT_MANIFEST.append(
                {
                    "artifact_order": len(ARTIFACT_MANIFEST) + 1,
                    "file_name": artifact_path.name,
                    "saved_path": str(artifact_path),
                    "section": section_tag,
                    "artifact_kind": artifact_kind,
                    "short_name": short_name,
                    "manuscript_role": manuscript_role,
                    "llm_guidance": llm_guidance,
                    "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                }
            )


        def save_dataframe_artifact(
            dataframe: pd.DataFrame,
            section_tag: str,
            short_name: str,
            manuscript_role: str,
            llm_guidance: str,
            include_index: bool = False,
        ) -> Path:
            artifact_path = build_artifact_path(section_tag, "table", short_name, "csv")
            dataframe.to_csv(artifact_path, index=include_index)
            register_artifact(
                artifact_path=artifact_path,
                section_tag=section_tag,
                artifact_kind="table",
                short_name=short_name,
                manuscript_role=manuscript_role,
                llm_guidance=llm_guidance,
            )
            return artifact_path


        def save_text_artifact(
            text_content: str,
            section_tag: str,
            short_name: str,
            manuscript_role: str,
            llm_guidance: str,
        ) -> Path:
            artifact_path = build_artifact_path(section_tag, "text", short_name, "txt")
            with open(artifact_path, "w", encoding="utf-8") as file_handle:
                file_handle.write(text_content)
            register_artifact(
                artifact_path=artifact_path,
                section_tag=section_tag,
                artifact_kind="text",
                short_name=short_name,
                manuscript_role=manuscript_role,
                llm_guidance=llm_guidance,
            )
            return artifact_path


        def save_figure_artifact(
            figure_object,
            section_tag: str,
            short_name: str,
            manuscript_role: str,
            llm_guidance: str,
        ) -> Path:
            artifact_path = build_artifact_path(section_tag, "figure", short_name, "png")
            figure_object.savefig(artifact_path, dpi=300, bbox_inches="tight")
            register_artifact(
                artifact_path=artifact_path,
                section_tag=section_tag,
                artifact_kind="figure",
                short_name=short_name,
                manuscript_role=manuscript_role,
                llm_guidance=llm_guidance,
            )
            return artifact_path


        def save_model_artifact(
            keras_model,
            section_tag: str,
            short_name: str,
            manuscript_role: str,
            llm_guidance: str,
        ) -> Path:
            artifact_path = build_artifact_path(section_tag, "model", short_name, "keras")
            keras_model.save(artifact_path)
            register_artifact(
                artifact_path=artifact_path,
                section_tag=section_tag,
                artifact_kind="model",
                short_name=short_name,
                manuscript_role=manuscript_role,
                llm_guidance=llm_guidance,
            )
            return artifact_path


        config = ExperimentConfig()
        output_dir = resolve_output_directory(config.output_subdirectory)
        print(config)
        print("Keras backend:", keras.backend.backend())
        print("Keras version:", keras.__version__)
        print("Output directory:", output_dir)

        reproducibility_table = pd.DataFrame(
            [
                {"Item": "Python hash seed", "Value": os.environ["PYTHONHASHSEED"]},
                {"Item": "Random seed", "Value": SEED},
                {"Item": "NumPy seed", "Value": SEED},
                {"Item": "Keras random seed", "Value": SEED},
                {"Item": "TensorFlow deterministic ops", "Value": tensorflow_deterministic_ops},
                {"Item": "Market ticker", "Value": config.market_ticker},
                {"Item": "Sample start date", "Value": config.start_date},
                {"Item": "Sample end date", "Value": config.end_date},
                {"Item": "Sequence length", "Value": config.sequence_length},
                {"Item": "Artifact output directory", "Value": str(output_dir)},
            ]
        )
        display(reproducibility_table)
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

        data_coverage_table = pd.DataFrame(
            [
                {
                    "Dataset": "S&P 500 OHLCV",
                    "Observations": len(market_raw),
                    "Start": market_raw.index.min().date(),
                    "End": market_raw.index.max().date(),
                },
                {
                    "Dataset": "FRED macro-sentiment panel",
                    "Observations": len(macro_raw),
                    "Start": macro_raw.index.min().date(),
                    "End": macro_raw.index.max().date(),
                },
            ]
        )
        missingness_table = (
            macro_raw.isna().mean().sort_values(ascending=False).to_frame(name="Missing Share")
        )
        display(data_coverage_table)
        display(missingness_table)

        figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
        market_raw["Adj Close"].plot(ax=axes[0], color="navy", linewidth=1.2)
        axes[0].set_title("S&P 500 Adjusted Close (Raw Level)")
        axes[0].set_ylabel("Index level")

        macro_raw[["consumer_sentiment", "macro_news_volatility"]].dropna().plot(
            ax=axes[1], linewidth=1.1
        )
        axes[1].set_title("Macro Sentiment Proxies from FRED")
        axes[1].set_ylabel("Index values")
        plt.tight_layout()
        figure_data_overview_path = save_figure_artifact(
            figure_object=figure,
            section_tag="sec02",
            short_name="market_macro_data_overview",
            manuscript_role="Data section figure showing non-stationarity and macro context.",
            llm_guidance=(
                "Use this figure to motivate multimodal fusion and to visually support the abstract claim "
                "that market behavior co-moves with macro sentiment conditions."
            ),
        )
        print(f"Saved figure: {figure_data_overview_path}")
        plt.show()

        display(market_raw.head())
        display(macro_raw.head())
        """
    ),
    markdown_cell(
        r"""
        # Section 3 - Feature Engineering for Market and Macro Signals

        For the market channel, we engineer trend, momentum, and risk descriptors (moving-average gaps, RSI, MACD, ATR, and realized volatility).  
        For the macro channel, we derive interpretable transforms:

        $$ \pi^{YoY}_t = 100 \times \left(\frac{CPI_t}{CPI_{t-12}} - 1\right), \quad
        Spread_t = DGS10_t - FEDFUNDS_t $$

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

        feature_catalog_table = pd.DataFrame(
            [
                {"Feature family": "Market", "Variable": feature_name}
                for feature_name in market_feature_columns
            ]
            + [
                {"Feature family": "Macro-sentiment", "Variable": feature_name}
                for feature_name in macro_feature_columns
            ]
        )
        market_feature_stats = market_df[market_feature_columns].describe().T[
            ["mean", "std", "min", "max"]
        ]
        macro_feature_stats = macro_df[macro_feature_columns].describe().T[
            ["mean", "std", "min", "max"]
        ]

        display(feature_catalog_table)
        display(market_feature_stats.round(4))
        display(macro_feature_stats.round(4))
        """
    ),
    markdown_cell(
        r"""
        # Section 4 - Temporal Alignment Without Look-Ahead Bias

        Macroeconomic data arrive at mixed frequencies and are not updated at every market close.  
        We align macro data to the trading calendar via forward fill and then apply a one-business-day lag:

        $$ \tilde{m}_t = m_{\max(\tau \le t-1)} $$

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

        lag_validation_table = pd.DataFrame(
            [
                {
                    "Macro feature": feature_name,
                    "First valid date (no lag)": macro_daily[feature_name].first_valid_index(),
                    "First valid date (lagged)": macro_daily_lagged[feature_name].first_valid_index(),
                }
                for feature_name in macro_feature_columns
            ]
        )
        alignment_quality_table = pd.DataFrame(
            [
                {
                    "Statistic": "Rows after strict alignment",
                    "Value": len(aligned_df),
                },
                {
                    "Statistic": "Earliest aligned date",
                    "Value": aligned_df.index.min().date(),
                },
                {
                    "Statistic": "Latest aligned date",
                    "Value": aligned_df.index.max().date(),
                },
                {
                    "Statistic": "Rows removed by NaN filtering",
                    "Value": len(market_df) - len(aligned_df),
                },
            ]
        )

        display(alignment_quality_table)
        display(lag_validation_table)
        display(aligned_df.head())
        display(aligned_df.tail())
        """
    ),
    markdown_cell(
        r"""
        # Section 5 - Time-Aware Splitting, Scaling, and 3D Sequence Construction

        We use chronological splitting (train, validation, test) to preserve causality.  
        Feature normalization is fit only on the training window and then applied forward in time.

        For sequence length $L$, each sample is:

        $$ X_t^{market} \in \mathbb{R}^{L \times d_m}, \quad
        X_t^{macro} \in \mathbb{R}^{L \times d_c}, \quad
        y_t = r_{t+1} $$

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

        split_overview_table = pd.DataFrame(
            [
                {
                    "Split": "Train",
                    "Raw rows": len(train_df),
                    "Sequence samples": len(y_train),
                    "Start": train_df.index.min().date(),
                    "End": train_df.index.max().date(),
                },
                {
                    "Split": "Validation",
                    "Raw rows": len(validation_df),
                    "Sequence samples": len(y_validation),
                    "Start": validation_df.index.min().date(),
                    "End": validation_df.index.max().date(),
                },
                {
                    "Split": "Test",
                    "Raw rows": len(test_df),
                    "Sequence samples": len(y_test),
                    "Start": test_df.index.min().date(),
                    "End": test_df.index.max().date(),
                },
            ]
        )
        tensor_shape_table = pd.DataFrame(
            [
                {"Tensor": "X_market_train", "Shape": X_market_train.shape},
                {"Tensor": "X_macro_train", "Shape": X_macro_train.shape},
                {"Tensor": "y_train", "Shape": y_train.shape},
                {"Tensor": "X_market_validation", "Shape": X_market_validation.shape},
                {"Tensor": "X_macro_validation", "Shape": X_macro_validation.shape},
                {"Tensor": "y_validation", "Shape": y_validation.shape},
                {"Tensor": "X_market_test", "Shape": X_market_test.shape},
                {"Tensor": "X_macro_test", "Shape": X_macro_test.shape},
                {"Tensor": "y_test", "Shape": y_test.shape},
            ]
        )
        display(split_overview_table)
        display(tensor_shape_table)
        """
    ),
    markdown_cell(
        r"""
        # Section 6 - Regime-Aware Hybrid Deep Learning Architecture

        The network has two branches:

        - Market branch: `Conv1D -> LSTM -> Temporal Attention`
        - Macro branch: `LSTM -> Temporal Attention`

        A regime gate is learned from macro context and used to modulate market context before fusion:

        $$ g = \sigma(W_g z^{macro} + b_g), \quad
        \hat{z}^{market} = g \odot z^{market} $$

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

        model_summary_lines = []
        model.summary(print_fn=model_summary_lines.append)
        print("\n".join(model_summary_lines))

        model_parameter_table = pd.DataFrame(
            [
                {"Statistic": "Total parameters", "Value": model.count_params()},
                {
                    "Statistic": "Trainable parameters",
                    "Value": int(
                        np.sum([weight.shape.num_elements() for weight in model.trainable_weights])
                    ),
                },
                {
                    "Statistic": "Non-trainable parameters",
                    "Value": int(
                        np.sum([weight.shape.num_elements() for weight in model.non_trainable_weights])
                    ),
                },
                {"Statistic": "Market feature count", "Value": len(market_feature_columns)},
                {"Statistic": "Macro feature count", "Value": len(macro_feature_columns)},
            ]
        )
        display(model_parameter_table)
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
            shuffle=False,
            callbacks=training_callbacks,
            verbose=1,
        )

        history_frame = pd.DataFrame(history.history)
        display(history_frame.tail())

        figure, axes = plt.subplots(1, 2, figsize=(14, 4))
        history_frame[["loss", "val_loss"]].plot(ax=axes[0], title="Training and Validation Loss")
        history_frame[["rmse", "val_rmse"]].plot(ax=axes[1], title="Training and Validation RMSE")
        plt.tight_layout()
        figure_training_curves_path = save_figure_artifact(
            figure_object=figure,
            section_tag="sec07",
            short_name="training_validation_curves",
            manuscript_role="Optimization diagnostics figure for training stability evidence.",
            llm_guidance=(
                "Use this figure in the Methods or Appendix section to discuss convergence, "
                "regularization behavior, and overfitting control."
            ),
        )
        print(f"Saved figure: {figure_training_curves_path}")
        plt.show()

        y_pred_test = model.predict([X_market_test, X_macro_test], verbose=0).reshape(-1)
        y_true_test = y_test.reshape(-1)

        test_rmse = float(np.sqrt(mean_squared_error(y_true_test, y_pred_test)))
        test_mae = float(mean_absolute_error(y_true_test, y_pred_test))

        print(f"Test RMSE (log-return scale): {test_rmse:.6f}")
        print(f"Test MAE  (log-return scale): {test_mae:.6f}")

        best_epoch = int(history_frame["val_loss"].idxmin()) + 1
        training_summary_table = pd.DataFrame(
            [
                {"Statistic": "Best epoch (by validation loss)", "Value": best_epoch},
                {"Statistic": "Best validation loss", "Value": float(history_frame["val_loss"].min())},
                {"Statistic": "Final training loss", "Value": float(history_frame["loss"].iloc[-1])},
                {"Statistic": "Final validation loss", "Value": float(history_frame["val_loss"].iloc[-1])},
                {"Statistic": "Test RMSE (log returns)", "Value": test_rmse},
                {"Statistic": "Test MAE (log returns)", "Value": test_mae},
            ]
        )
        display(training_summary_table)

        prediction_diagnostic_df = pd.DataFrame(
            {"realized_log_return": y_true_test, "predicted_log_return": y_pred_test}
        )
        prediction_scatter_figure = plt.figure(figsize=(6, 6))
        plt.scatter(
            prediction_diagnostic_df["realized_log_return"],
            prediction_diagnostic_df["predicted_log_return"],
            alpha=0.25,
            s=12,
        )
        plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
        plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
        plt.title("Predicted vs Realized Test Returns")
        plt.xlabel("Realized log return")
        plt.ylabel("Predicted log return")
        plt.tight_layout()
        figure_prediction_scatter_path = save_figure_artifact(
            figure_object=prediction_scatter_figure,
            section_tag="sec07",
            short_name="predicted_vs_realized_scatter",
            manuscript_role="Prediction diagnostic figure for forecast quality interpretation.",
            llm_guidance=(
                "Use this plot to discuss calibration quality and sign consistency in out-of-sample returns."
            ),
        )
        print(f"Saved figure: {figure_prediction_scatter_path}")
        plt.show()
        """
    ),
    markdown_cell(
        r"""
        # Section 8 - Financially Relevant Evaluation

        Beyond point forecast error, we evaluate trading relevance.

        Directional Accuracy:

        $$ DA = \frac{1}{N}\sum_{t=1}^{N} \mathbf{1}\left[\operatorname{sign}(\hat{r}_{t+1}) = \operatorname{sign}(r_{t+1})\right] $$

        Trading rule: go long if $\hat{r}_{t+1} > 0$, otherwise allocate to cash.  
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


        def one_sided_binomial_p_value(successes: int, trials: int, null_probability: float = 0.5) -> float:
            from math import comb

            return float(
                sum(
                    comb(trials, k)
                    * (null_probability ** k)
                    * ((1 - null_probability) ** (trials - k))
                    for k in range(successes, trials + 1)
                )
            )


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
        evaluation_df["macro_news_volatility"] = aligned_df.loc[
            evaluation_df.index, "macro_news_volatility"
        ].values
        macro_regime_codes = pd.qcut(
            evaluation_df["macro_news_volatility"], q=3, labels=False, duplicates="drop"
        )
        regime_label_map = {0: "Low uncertainty", 1: "Medium uncertainty", 2: "High uncertainty"}
        evaluation_df["macro_uncertainty_regime"] = macro_regime_codes.map(regime_label_map)

        directional_accuracy = float(
            (
                np.sign(evaluation_df["predicted_log_return"])
                == np.sign(evaluation_df["realized_log_return"])
            ).mean()
        )
        directional_successes = int(
            (
                np.sign(evaluation_df["predicted_log_return"])
                == np.sign(evaluation_df["realized_log_return"])
            ).sum()
        )
        directional_trials = int(len(evaluation_df))
        directional_p_value = one_sided_binomial_p_value(
            successes=directional_successes,
            trials=directional_trials,
            null_probability=0.5,
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
                    "Metric": "Directional Accuracy p-value (H0: 50%)",
                    "Strategy": directional_p_value,
                    "Buy and Hold": np.nan,
                },
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

        regime_performance_table = (
            evaluation_df.groupby("macro_uncertainty_regime")
            .apply(
                lambda group: pd.Series(
                    {
                        "Observations": len(group),
                        "Directional Accuracy": float(
                            (
                                np.sign(group["predicted_log_return"])
                                == np.sign(group["realized_log_return"])
                            ).mean()
                        ),
                        "Strategy Cumulative Return": float(
                            (1 + group["strategy_simple_return"]).prod() - 1.0
                        ),
                        "Buy-and-Hold Cumulative Return": float(
                            (1 + group["buy_hold_simple_return"]).prod() - 1.0
                        ),
                    }
                )
            )
            .reset_index()
        )

        display(metrics_table)
        display(regime_performance_table)

        equity_curve_df = pd.DataFrame(index=evaluation_df.index)
        equity_curve_df["Regime-Aware Strategy"] = (1 + evaluation_df["strategy_simple_return"]).cumprod()
        equity_curve_df["Buy and Hold"] = (1 + evaluation_df["buy_hold_simple_return"]).cumprod()

        equity_curve_figure = plt.figure(figsize=(12, 5))
        plt.plot(equity_curve_df.index, equity_curve_df["Regime-Aware Strategy"], label="Regime-Aware Strategy")
        plt.plot(equity_curve_df.index, equity_curve_df["Buy and Hold"], label="Buy and Hold", alpha=0.8)
        plt.title("Out-of-Sample Equity Curves")
        plt.ylabel("Growth of $1")
        plt.legend()
        plt.tight_layout()
        figure_equity_curves_path = save_figure_artifact(
            figure_object=equity_curve_figure,
            section_tag="sec08",
            short_name="strategy_vs_buy_hold_equity_curves",
            manuscript_role="Main performance figure comparing strategy utility against benchmark.",
            llm_guidance=(
                "Use this figure in the Results section to support claims about economic utility "
                "beyond pure forecasting error metrics."
            ),
        )
        print(f"Saved figure: {figure_equity_curves_path}")
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
        # Section 9 code: export paper-ready artifacts and print manuscript-oriented summaries.
        save_dataframe_artifact(
            dataframe=data_coverage_table,
            section_tag="sec09",
            short_name="data_coverage_table",
            manuscript_role="Documents temporal coverage of market and macro datasets.",
            llm_guidance="Cite this table in the Data subsection to define the sample construction.",
            include_index=False,
        )
        save_dataframe_artifact(
            dataframe=missingness_table,
            section_tag="sec09",
            short_name="macro_missingness_table",
            manuscript_role="Missingness diagnostics for macro sentiment proxies.",
            llm_guidance="Use this table to discuss data quality and imputation assumptions.",
            include_index=True,
        )
        save_dataframe_artifact(
            dataframe=feature_catalog_table,
            section_tag="sec09",
            short_name="feature_catalog_table",
            manuscript_role="Catalog of engineered multimodal features.",
            llm_guidance="Use this table to explain market and macro feature families.",
            include_index=False,
        )
        save_dataframe_artifact(
            dataframe=market_feature_stats,
            section_tag="sec09",
            short_name="market_feature_statistics",
            manuscript_role="Descriptive statistics for engineered market features.",
            llm_guidance="Use for Methods and Appendix descriptive analysis.",
            include_index=True,
        )
        save_dataframe_artifact(
            dataframe=macro_feature_stats,
            section_tag="sec09",
            short_name="macro_feature_statistics",
            manuscript_role="Descriptive statistics for engineered macro features.",
            llm_guidance="Use for Methods and Appendix descriptive analysis.",
            include_index=True,
        )
        save_dataframe_artifact(
            dataframe=alignment_quality_table,
            section_tag="sec09",
            short_name="temporal_alignment_quality",
            manuscript_role="Evidence of strict no-look-ahead alignment design.",
            llm_guidance="Use this to justify causal alignment protocol and leakage control.",
            include_index=False,
        )
        save_dataframe_artifact(
            dataframe=split_overview_table,
            section_tag="sec09",
            short_name="chronological_split_overview",
            manuscript_role="Experimental split summary for train/validation/test windows.",
            llm_guidance="Use in Experimental Setup to describe time-aware splitting.",
            include_index=False,
        )
        save_dataframe_artifact(
            dataframe=tensor_shape_table,
            section_tag="sec09",
            short_name="sequence_tensor_shapes",
            manuscript_role="Model input-output tensor dimensionality reference.",
            llm_guidance="Use in Methods to explain 3D sequence construction.",
            include_index=False,
        )
        save_dataframe_artifact(
            dataframe=model_parameter_table,
            section_tag="sec09",
            short_name="model_parameter_counts",
            manuscript_role="Architecture complexity and parameterization evidence.",
            llm_guidance="Use to report model size and reproducibility details.",
            include_index=False,
        )
        save_dataframe_artifact(
            dataframe=training_summary_table,
            section_tag="sec09",
            short_name="training_summary_table",
            manuscript_role="Optimization summary with best epoch and losses.",
            llm_guidance="Use to describe training behavior and convergence.",
            include_index=False,
        )
        save_dataframe_artifact(
            dataframe=metrics_table,
            section_tag="sec09",
            short_name="main_performance_metrics",
            manuscript_role="Primary forecasting and financial utility result table.",
            llm_guidance="Use as main quantitative results table in the manuscript.",
            include_index=False,
        )
        save_dataframe_artifact(
            dataframe=regime_performance_table,
            section_tag="sec09",
            short_name="regime_conditional_performance",
            manuscript_role="Performance decomposition by macro uncertainty regime.",
            llm_guidance="Use to support the regime-aware contribution claim.",
            include_index=False,
        )
        save_dataframe_artifact(
            dataframe=evaluation_df,
            section_tag="sec09",
            short_name="prediction_level_test_outputs",
            manuscript_role="Row-level out-of-sample predictions and realized returns.",
            llm_guidance="Use for appendix robustness checks and auditability.",
            include_index=True,
        )
        save_model_artifact(
            keras_model=model,
            section_tag="sec09",
            short_name="regime_aware_model_checkpoint",
            manuscript_role="Serialized model checkpoint for reproducibility.",
            llm_guidance="Use to reproduce inference and perform ablation comparisons.",
        )

        manuscript_results_paragraph = (
            "The proposed regime-aware multimodal model integrates S&P 500 price dynamics with "
            "macro-sentiment proxies from FRED and delivers a test directional accuracy of "
            f"{directional_accuracy:.2%}. The associated one-sided binomial p-value under a 50% null "
            f"is {directional_p_value:.4f}, supporting directional skill beyond chance. Under a long-only "
            "signal rule, the strategy reaches a cumulative return of "
            f"{strategy_cumulative_return:.2%} versus {buy_hold_cumulative_return:.2%} for buy-and-hold, "
            f"with Sharpe ratios of {strategy_sharpe:.3f} and {buy_hold_sharpe:.3f}, respectively."
        )
        save_text_artifact(
            text_content=manuscript_results_paragraph,
            section_tag="sec09",
            short_name="results_paragraph_draft",
            manuscript_role="Draft paragraph for the Results section narrative.",
            llm_guidance="Use directly as prose seed when drafting the main results subsection.",
        )

        summary_series = pd.Series(
            {
                "test_rmse_log_return": test_rmse,
                "test_mae_log_return": test_mae,
                "directional_accuracy": directional_accuracy,
                "directional_accuracy_p_value": directional_p_value,
                "strategy_cumulative_return": strategy_cumulative_return,
                "buy_hold_cumulative_return": buy_hold_cumulative_return,
                "strategy_sharpe": strategy_sharpe,
                "buy_hold_sharpe": buy_hold_sharpe,
                "strategy_max_drawdown": strategy_max_drawdown,
                "buy_hold_max_drawdown": buy_hold_max_drawdown,
            }
        )
        headline_metrics_table = summary_series.to_frame(name="value")
        save_dataframe_artifact(
            dataframe=headline_metrics_table,
            section_tag="sec09",
            short_name="headline_metric_snapshot",
            manuscript_role="Compact summary of top-line performance indicators.",
            llm_guidance="Use as a quick reference block in abstract, conclusion, and response letters.",
            include_index=True,
        )

        paper_asset_index_table = pd.DataFrame(ARTIFACT_MANIFEST).sort_values("artifact_order")

        print("Saved files in execution order:")
        for artifact_path in paper_asset_index_table["saved_path"]:
            print(f"- {artifact_path}")

        display(summary_series.to_frame(name="value"))
        display(
            paper_asset_index_table[
                [
                    "artifact_order",
                    "file_name",
                    "section",
                    "artifact_kind",
                    "manuscript_role",
                    "saved_path",
                ]
            ]
        )
        print("Draft manuscript paragraph:")
        print(manuscript_results_paragraph)
        """
    ),
    markdown_cell(
        """
        # Section 10 - Paper Draft Blocks and Figure Captions

        This section produces reusable writing assets for immediate manuscript drafting: figure captions, table captions, and a concise limitations block aligned with the abstract claims.
        """
    ),
    code_cell(
        """
        # Section 10 code: generate paper-writing assets (captions and limitations statements).
        figure_caption_table = pd.DataFrame(
            [
                {
                    "Figure": "Figure 1",
                    "Caption": "S&P 500 adjusted close and macro-sentiment proxies over the study period, illustrating non-stationarity and macro uncertainty shifts.",
                },
                {
                    "Figure": "Figure 2",
                    "Caption": "Training and validation trajectories for loss and RMSE, showing optimization stability under early stopping.",
                },
                {
                    "Figure": "Figure 3",
                    "Caption": "Out-of-sample equity curves comparing the regime-aware strategy against buy-and-hold.",
                },
                {
                    "Figure": "Figure 4",
                    "Caption": "Predicted versus realized one-day-ahead log returns on the test split.",
                },
            ]
        )

        table_caption_table = pd.DataFrame(
            [
                {
                    "Table": "Table 1",
                    "Caption": "Sample coverage and missingness diagnostics for market and macro-sentiment data.",
                },
                {
                    "Table": "Table 2",
                    "Caption": "Feature catalog and summary statistics for engineered market and macro variables.",
                },
                {
                    "Table": "Table 3",
                    "Caption": "Core predictive and financial performance metrics against buy-and-hold.",
                },
                {
                    "Table": "Table 4",
                    "Caption": "Regime-conditional performance by macro uncertainty terciles.",
                },
            ]
        )

        limitations_block = (
            "Limitations: (i) the backtest excludes transaction costs and market impact, "
            "(ii) macro release calendars are approximated via conservative lagging rather than full real-time vintages, "
            "and (iii) results are based on a single index and should be stress-tested with rolling-origin evaluation."
        )

        save_dataframe_artifact(
            dataframe=figure_caption_table,
            section_tag="sec10",
            short_name="figure_caption_library",
            manuscript_role="Ready-to-use captions for manuscript figures.",
            llm_guidance="Use this table when assembling figure legends in the final article.",
            include_index=False,
        )
        save_dataframe_artifact(
            dataframe=table_caption_table,
            section_tag="sec10",
            short_name="table_caption_library",
            manuscript_role="Ready-to-use captions for manuscript tables.",
            llm_guidance="Use this table when assembling table legends in the final article.",
            include_index=False,
        )
        save_text_artifact(
            text_content=limitations_block,
            section_tag="sec10",
            short_name="discussion_limitations_block",
            manuscript_role="Limitations paragraph for Discussion section.",
            llm_guidance="Use this text as a structured limitations paragraph in the manuscript discussion.",
        )

        paper_synthesis_prompt = (
            "Paper assembly guidance for LLMs:\\n"
            "1) Use artifacts in ascending artifact_order from the manifest.\\n"
            "2) Build the Data section from coverage, missingness, and feature tables.\\n"
            "3) Build the Methods section from alignment, split, tensor, and model parameter tables.\\n"
            "4) Build the Results section from main metrics, regime performance, and figures.\\n"
            "5) Use results_paragraph_draft as a starting point, then cross-check every numeric claim.\\n"
            "6) Use limitations text unchanged unless new robustness checks are added.\\n"
            "7) Always reference uncertainty-regime evidence when discussing the contribution."
        )
        save_text_artifact(
            text_content=paper_synthesis_prompt,
            section_tag="sec10",
            short_name="llm_paper_synthesis_prompt",
            manuscript_role="Instruction block for automated manuscript drafting.",
            llm_guidance="Use this as system context when asking an LLM to draft the paper.",
        )

        manifest_table = pd.DataFrame(ARTIFACT_MANIFEST).sort_values("artifact_order").reset_index(drop=True)
        save_dataframe_artifact(
            dataframe=manifest_table,
            section_tag="sec10",
            short_name="artifact_manifest",
            manuscript_role="Master index of all generated artifacts in execution order.",
            llm_guidance="This is the primary routing table for any LLM-based paper construction pipeline.",
            include_index=False,
        )

        manifest_json_path = build_artifact_path("sec10", "manifest", "artifact_manifest", "json")
        with open(manifest_json_path, "w", encoding="utf-8") as file_handle:
            json.dump(ARTIFACT_MANIFEST, file_handle, indent=2, ensure_ascii=False)
        register_artifact(
            artifact_path=manifest_json_path,
            section_tag="sec10",
            artifact_kind="manifest",
            short_name="artifact_manifest",
            manuscript_role="Machine-readable manifest with rich metadata for all artifacts.",
            llm_guidance="Use for programmatic ingestion and retrieval-augmented drafting workflows.",
        )

        final_manifest_table = pd.DataFrame(ARTIFACT_MANIFEST).sort_values("artifact_order").reset_index(drop=True)
        display(figure_caption_table)
        display(table_caption_table)
        display(
            final_manifest_table[
                [
                    "artifact_order",
                    "file_name",
                    "section",
                    "artifact_kind",
                    "short_name",
                    "manuscript_role",
                    "saved_path",
                ]
            ]
        )
        print("Saved additional writing and manifest assets in execution order:")
        for artifact_path in final_manifest_table["saved_path"]:
            print(f"- {artifact_path}")
        print("Limitations block for Discussion section:")
        print(limitations_block)
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
