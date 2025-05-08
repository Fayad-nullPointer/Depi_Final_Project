import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import timesfm
from statsmodels.tsa.arima.model import ARIMA

# Set page configuration
st.set_page_config(
    page_title="Store Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

# App title and description
st.title("📊 Store Sales Time Series Forecasting")
st.markdown("""
This application analyzes store sales data and provides forecasting using multiple models.
Use the sidebar to navigate through different sections of the app.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
pages = ["Data Overview", "Exploratory Data Analysis", "Basic Time Series Analysis", 
         "Time Series Forecasting with TimesFM", "LSTM Forecasting", "ARIMA Forecasting"]
selected_page = st.sidebar.radio("Go to", pages)

# Add data source section to the sidebar
st.sidebar.header("📂 Data Source")

# Function to check if file exists in any of the common locations - this is NOT cached
def find_data_file():
    possible_paths = [
        "Rossmann Stores Data.csv",
        "data/Rossmann Stores Data.csv",
        "Rossmann_Stores_Data.csv"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                return pd.read_csv(path), path
            except Exception:
                continue
    return None, None

# Updated process_data_from_file to handle NaT values
# Function to process data from file - only contains data processing, no widgets
# @st.cache_data
def process_data_from_file(file_content):
    df = file_content.copy()
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True, errors='coerce')
    # Drop rows with invalid dates
    df = df.dropna(subset=['Date'])
    return df

# Function to create sample data for demo - only contains data generation, no widgets
def create_sample_data():
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    stores = np.random.randint(1, 11, size=100)
    sales = np.random.randint(2000, 15000, size=100)
    
    df = pd.DataFrame({
        'Date': dates,
        'Store': stores,
        'Sales': sales,
        'Customers': np.random.randint(100, 1000, size=100),
        'Open': 1,
        'Promo': np.random.randint(0, 2, size=100),
        'StateHoliday': 0,
        'SchoolHoliday': np.random.randint(0, 2, size=100)
    })
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# Centralize data upload logic to ensure data is uploaded only once
# Updated load_data to separate widget commands from cached logic

def load_data():
    # Try to find and load the dataset from common locations
    found_data, found_path = find_data_file()
    if found_data is not None:
        st.sidebar.success(f"✅ Data loaded from {found_path}")
        return process_data_from_file(found_data)

    # If data not found, offer upload option
    st.sidebar.info("📤 Please upload Rossmann Store Data")
    uploaded_file = st.sidebar.file_uploader(
        "Drag and drop CSV file here",
        type=["csv"],
        help="Upload the Rossmann Stores Data CSV file"
    )

    use_sample_data = st.sidebar.button("Use sample data (for demo only)")

    if uploaded_file is not None:
        return handle_uploaded_file(uploaded_file)
    elif use_sample_data:
        st.sidebar.warning("⚠️ Using sample data. For accurate analysis, upload the actual dataset.")
        return create_sample_data()
    else:
        st.error("❌ Please upload the Rossmann Stores dataset to continue")
        st.info("You can drag and drop the CSV file in the sidebar or use the sample data option")
        st.stop()

def handle_uploaded_file(uploaded_file):
    try:
        file_content = pd.read_csv(uploaded_file)
        df = process_data_from_file(file_content)

        # Basic validation of required columns
        required_columns = ['Date', 'Store', 'Sales']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            st.sidebar.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
            st.stop()

        # Option to save file for future use
        if st.sidebar.checkbox("Save file for future use", value=True):
            try:
                df.to_csv("Rossmann Stores Data.csv", index=False)
                st.sidebar.success("✅ File saved for future use")
            except Exception as e:
                st.sidebar.warning(f"⚠️ Could not save file: {str(e)}")

        st.sidebar.success("✅ Data loaded successfully")
        return df
    except Exception as e:
        st.sidebar.error(f"❌ Error loading data: {str(e)}")
        st.stop()

# Load data once
df = load_data()

# Check if we have data
if df is None:
    st.error("❌ Could not load or generate data")
    st.stop()

# Function for data preprocessing - only contains data processing, no widgets
def preprocess_data(df):
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True, errors='coerce')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek + 1
    return df

# Preprocess the data
df_processed = preprocess_data(df)

# Function to prepare data for TimesFM model - only contains data processing, no widgets
def prepare_timesfm_data(df):
    Dropped = ['Store', 'DayOfWeek', 'Customers', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday']
    new_df = df.drop(Dropped, axis=1)
    new_df = new_df.rename(columns={"Date": "ds"})
    new_df = new_df.reset_index(drop=True)
    
    # Sort values to ensure consistent ordering
    new_df = new_df.sort_values(by=["ds"]).reset_index(drop=True)
    
    # Create group-wise counters that reset per date
    new_df["unique_id"] = new_df.groupby("ds").cumcount() + 1
    
    # Convert the counter to T1, T2, ...
    new_df["unique_id"] = new_df["unique_id"].apply(lambda x: f"T{x}")
    
    return new_df

# ======== Data Overview Page =========
if selected_page == "Data Overview":
    st.header("Data Overview")
    
    # Display dataset general info
    st.subheader("Dataset Sample")
    st.dataframe(df.head())
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Shape")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        
    with col2:
        st.subheader("Columns")
        st.write(df.columns.tolist())
    
    # Display missing values
    st.subheader("Missing Values")
    st.write(df.isnull().sum())
    
    # Display basic statistics
    st.subheader("Summary Statistics")
    st.write(df.describe())
    
    # Display information about data types
    st.subheader("Data Types")
    buffer = pd.DataFrame({'Data Type': df.dtypes})
    st.write(buffer)

# ======== EDA Page =========
elif selected_page == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Biggest Sales Amount")
        max_sales_value = df['Sales'].max()
        st.metric("Maximum Sales", f"€{max_sales_value:,.2f}")
        
        date_of_max = df.loc[df['Sales'] == max_sales_value, 'Date'].iloc[0]
        st.write(f"Date of Maximum Sales: {date_of_max.strftime('%Y-%m-%d')}")
    
    with col2:
        st.subheader("Store with Highest Total Sales")
        store_sales = df.groupby("Store")["Sales"].sum()
        max_store_id = store_sales.idxmax()
        max_sales_amount = store_sales.max()
        st.metric("Store ID", max_store_id)
        st.metric("Total Sales", f"€{max_sales_amount:,.2f}")
    
    # Year selection for specific analysis
    st.subheader("Analysis by Year")
    year_options = sorted(df['Date'].dt.year.unique().tolist())
    selected_year = st.selectbox("Select Year", year_options)
    
    df_year = df[df['Date'].dt.year == selected_year]
    store_sales_year = df_year.groupby("Store")["Sales"].sum()
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Store with most sales in selected year
        st.subheader(f"Top Store in {selected_year}")
        max_store_id_year = store_sales_year.idxmax()
        max_sales_amount_year = store_sales_year.max()
        st.metric("Store ID", max_store_id_year)
        st.metric("Total Sales", f"€{max_sales_amount_year:,.2f}")
    
    with col4:
        # Days of week analysis
        st.subheader(f"Sales by Day of Week in {selected_year}")
        df_year['DayName'] = df_year['Date'].dt.day_name()
        day_sales = df_year.groupby("DayName")["Sales"].sum()
        
        # Sort days correctly
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_sales = day_sales.reindex(days_order)
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=day_sales.index, y=day_sales.values, palette="Blues_r", ax=ax)
        ax.set_xlabel("Day of the Week")
        ax.set_ylabel("Total Sales Amount")
        ax.set_title(f"Total Sales by Day of the Week in {selected_year}")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    # Promotional and Holiday Analysis
    st.subheader("Impact of Promotions and Holidays")
    
    col5, col6 = st.columns(2)
    
    with col5:
        # Promo effect
        promo_sales = df.groupby("Promo")["Sales"].sum()
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=promo_sales.index, y=promo_sales.values, palette=["red", "green"], ax=ax)
        ax.set_xlabel("Promo Applied (0 = No, 1 = Yes)")
        ax.set_ylabel("Total Sales Amount")
        ax.set_title("Total Sales Amount with and without Promo")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No Promo", "With Promo"])
        st.pyplot(fig)
    
    with col6:
        # School holiday effect
        holiday_sales = df.groupby("SchoolHoliday")["Sales"].sum()
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=holiday_sales.index, y=holiday_sales.values, palette=["red", "green"], ax=ax)
        ax.set_xlabel("School Holiday (0 = No, 1 = Yes)")
        ax.set_ylabel("Total Sales Amount")
        ax.set_title("Total Sales During School Holidays vs. Normal Days")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No Holiday", "School Holiday"])
        st.pyplot(fig)

# ======== Basic Time Series Analysis =========
elif selected_page == "Basic Time Series Analysis":
    st.header("Basic Time Series Analysis")
    
    # Add store selection to Basic Time Series Analysis
    st.sidebar.subheader("Filter Options")
    store_options = sorted(df['Store'].unique().tolist())
    selected_store = st.sidebar.selectbox("Select Store", store_options)

    # Filter data for selected store
    store_data = df[df['Store'] == selected_store]

    # Aggregate daily sales for the store
    daily_sales = store_data.groupby('Date')['Sales'].sum().reset_index()
    daily_sales = daily_sales.sort_values('Date')
    
    # Display time series plot
    st.subheader(f"Daily Sales for Store {selected_store}")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(daily_sales['Date'], daily_sales['Sales'])
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.set_title(f"Daily Sales for Store {selected_store}")
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Seasonal decomposition
    st.subheader("Weekly Differencing")
    
    # Set date as index
    store_ts = daily_sales.set_index('Date')
    
    # Create 7-day difference
    diff_7 = store_ts['Sales'].diff(7)
    
    # Plot original and differenced series
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    store_ts.plot(ax=axs[0], legend=False, marker=".")
    store_ts.shift(7).plot(ax=axs[0], grid=True, legend=False, linestyle=":")
    axs[0].set_title(f"Original Time Series for Store {selected_store}")
    
    diff_7.plot(ax=axs[1], grid=True, marker=".")
    axs[1].set_title(f"7-day Differenced Time Series")
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Simple statistics
    st.subheader("Statistics")
    
    mae_naive = diff_7.abs().mean()
    st.metric("Mean Absolute Error (Naive Weekly Forecast)", f"{mae_naive:.2f}")

# ======== TimesFM Forecasting =========
elif selected_page == "Time Series Forecasting with TimesFM":
    st.header("Time Series Forecasting with TimesFM")
    
    st.warning("""
    This section requires the TimesFM package to be installed.
    If it's not installed, please run: `pip install timesfm`
    
    This model requires significant computational resources and 
    may take a while to run.
    """)
    
    # Choose whether to run TimesFM (since it's resource-intensive)
    run_timesfm = st.checkbox("Run TimesFM Model", value=False)
    
    if run_timesfm:
        try:
            import timesfm
            
            # Prepare data
            new_df = prepare_timesfm_data(df)
            
            # Display the processed data
            st.subheader("Processed Data for TimesFM")
            st.dataframe(new_df.head())
            
            # Store selection
            store_ids = sorted([int(uid.replace('T', '')) for uid in new_df['unique_id'].unique()])
            selected_store = st.selectbox("Select Store ID", store_ids)
            store_filter = f'T{selected_store}'
            
            store_data = new_df[new_df['unique_id'] == store_filter]
            
            # Select training/test split
            st.subheader("Training Parameters")
            test_size = st.slider("Test Size (%)", 10, 50, 20)
            
            # Split data
            split_idx = int(len(store_data) * (1 - test_size/100))
            train_df = store_data[:split_idx]
            test_df = store_data[split_idx:]
            
            # Initialize model
            st.text("Initializing TimesFM model...")
            
            tfm = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend="cpu",
                    per_core_batch_size=32,
                    horizon_len=128,
                    input_patch_len=32,
                    output_patch_len=128,
                    num_layers=50,
                    model_dims=1280,
                    use_positional_embedding=False,
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
            )
            
            # Train and forecast
            with st.spinner("Generating forecast..."):
                forecast_df = tfm.forecast_on_df(
                    inputs=train_df,
                    freq="D",  # Daily
                    value_name="Sales",
                    num_jobs=-1,
                )
            
            # Display forecast results
            st.subheader("Forecast Results")
            
            # Get forecast values
            forecast_values = forecast_df['timesfm'].values
            test_values = test_df['Sales'].values[:len(forecast_values)]
            
            # Calculate metrics
            mae = mean_absolute_error(forecast_values, test_values)
            rmse = np.sqrt(mean_squared_error(forecast_values, test_values))
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("MAE", f"{mae:.2f}")
            with col2:
                st.metric("RMSE", f"{rmse:.2f}")
            
            # Plot results
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(train_df['ds'].values, train_df['Sales'].values, label='Training Data')
            
            forecast_dates = pd.date_range(
                start=train_df['ds'].iloc[-1] + pd.Timedelta(days=1),
                periods=len(forecast_values),
                freq='D'
            )
            
            ax.plot(forecast_dates, forecast_values, label='Forecast', color='red')
            ax.plot(test_df['ds'].values[:len(test_values)], test_values, label='Actual', color='green')
            
            ax.set_xlabel('Date')
            ax.set_ylabel('Sales')
            ax.setTitle(f'TimesFM Forecast for Store {selected_store}')
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            st.pyplot(fig)
            
        except ImportError:
            st.error("TimesFM package is not installed. Please run: pip install timesfm")
    else:
        st.info("Select the checkbox to run TimesFM model")

# ======== LSTM Forecasting =========
elif selected_page == "LSTM Forecasting":
    st.header("LSTM Forecasting")
    
    st.warning("""
    This section requires TensorFlow to be installed.
    If it's not installed, please run: `pip install tensorflow`
    
    LSTM models require significant computational resources.
    """)
    
    run_lstm = st.checkbox("Run LSTM Model", value=False)
    
    if run_lstm:
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense
            
            # Store selection
            store_options = sorted(df['Store'].unique().tolist())
            selected_store = st.selectbox("Select Store", store_options)
            
            # Filter data for selected store
            store_data = df[df['Store'] == selected_store]
            
            # Prepare time series data
            store_ts = store_data.sort_values('Date').set_index('Date')['Sales']
            
            # Parameters for LSTM
            st.subheader("LSTM Parameters")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                seq_length = st.slider("Sequence Length", 5, 50, 30)
            with col2:
                test_size = st.slider("Test Size (%)", 10, 50, 20)
            with col3:
                epochs = st.slider("Training Epochs", 5, 50, 20)
            
            # Convert to numpy array and normalize
            sales_data = store_ts.values.reshape(-1, 1)
            
            # Normalize data
            scaler = MinMaxScaler()
            scaled_sales = scaler.fit_transform(sales_data)
            
            # Create sequences for LSTM
            def create_sequences(data, seq_length):
                X, y = [], []
                for i in range(len(data) - seq_length):
                    X.append(data[i:i+seq_length])
                    y.append(data[i+seq_length])
                return np.array(X), np.array(y)
            
            # Create sequences
            X, y = create_sequences(scaled_sales, seq_length)
            X = X.reshape((X.shape[0], seq_length, 1))
            
            # Split data
            split_idx = int(X.shape[0] * (1 - test_size/100))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Build LSTM model
            model = Sequential()
            model.add(LSTM(64, activation='relu', input_shape=(seq_length, 1)))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
            
            # Train model
            with st.spinner(f"Training LSTM model for {epochs} epochs..."):
                model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
            
            # Make predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # Inverse transform to get actual values
            train_pred_rescaled = scaler.inverse_transform(train_pred)
            test_pred_rescaled = scaler.inverse_transform(test_pred)
            y_train_rescaled = scaler.inverse_transform(y_train)
            y_test_rescaled = scaler.inverse_transform(y_test)
            
            # Calculate metrics
            train_mae = mean_absolute_error(y_train_rescaled, train_pred_rescaled)
            train_rmse = np.sqrt(mean_squared_error(y_train_rescaled, train_pred_rescaled))
            test_mae = mean_absolute_error(y_test_rescaled, test_pred_rescaled)
            test_rmse = np.sqrt(mean_squared_error(y_test_rescaled, test_pred_rescaled))
            
            # Display metrics
            st.subheader("Model Performance")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training MAE", f"{train_mae:.2f}")
                st.metric("Training RMSE", f"{train_rmse:.2f}")
            with col2:
                st.metric("Testing MAE", f"{test_mae:.2f}")
                st.metric("Testing RMSE", f"{test_rmse:.2f}")
            
            # Forecast next day
            last_seq = scaled_sales[-seq_length:]
            last_seq = last_seq.reshape((1, seq_length, 1))
            predicted_scaled = model.predict(last_seq)
            predicted_value = scaler.inverse_transform(predicted_scaled)
            
            st.subheader("Next Day Forecast")
            st.metric("Predicted Sales", f"€{predicted_value[0][0]:.2f}")
            
            # Plot results
            # Combine indices
            train_idx = store_ts.index[seq_length:seq_length+len(train_pred)]
            test_idx = store_ts.index[seq_length+len(train_pred):seq_length+len(train_pred)+len(test_pred)]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(train_idx, y_train_rescaled, label='Training Actual')
            ax.plot(train_idx, train_pred_rescaled, label='Training Predictions')
            ax.plot(test_idx, y_test_rescaled, label='Testing Actual')
            ax.plot(test_idx, test_pred_rescaled, label='Testing Predictions')
            
            ax.set_xlabel('Date')
            ax.set_ylabel('Sales')
            ax.set_title(f'LSTM Model - Store {selected_store}')
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            st.pyplot(fig)
            
        except ImportError:
            st.error("TensorFlow is not installed. Please run: pip install tensorflow")
    else:
        st.info("Select the checkbox to run LSTM model")

# ======== ARIMA Forecasting =========
elif selected_page == "ARIMA Forecasting":
    st.header("ARIMA Forecasting")
    
    # Store selection
    store_options = sorted(df['Store'].unique().tolist())
    selected_store = st.selectbox("Select Store", store_options)
    
    # Time period selection
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", 
                                min_value=min_date.date(),
                                max_value=max_date.date(),
                                value=min_date.date())
    with col2:
        end_date = st.date_input("End Date", 
                              min_value=min_date.date(),
                              max_value=max_date.date(),
                              value=max_date.date())
    
    # ARIMA parameters
    st.subheader("ARIMA Parameters")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        p = st.slider("p (Auto-Regressive)", 0, 5, 1)
    with col2:
        d = st.slider("d (Integration/Differencing)", 0, 2, 0) 
    with col3:
        q = st.slider("q (Moving Average)", 0, 5, 0)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        P = st.slider("Seasonal P", 0, 2, 0)
    with col2:
        D = st.slider("Seasonal D", 0, 2, 1)
    with col3:
        Q = st.slider("Seasonal Q", 0, 2, 1)
    
    col1, col2 = st.columns(2)
    with col1:
        s = st.slider("s (Seasonal Period)", 1, 30, 7)
    with col2:
        forecast_days = st.slider("Forecast Days", 1, 30, 7)
    
    # Filter data for selected store and timeframe
    store_data = df[(df['Store'] == selected_store) & 
                    (df['Date'] >= pd.Timestamp(start_date)) & 
                    (df['Date'] <= pd.Timestamp(end_date))]
    
    # Set up time series
    store_ts = store_data.set_index('Date')['Sales']
    
    # Fit ARIMA model
    if st.button("Fit ARIMA Model"):
        try:
            with st.spinner("Fitting ARIMA model..."):
                # Convert to regular time series
                store_ts = store_ts.asfreq('D')
                
                # Fit ARIMA model
                model = ARIMA(store_ts, 
                             order=(p, d, q), 
                             seasonal_order=(P, D, Q, s))
                model_fit = model.fit()
                
                # Generate forecast
                forecast = model_fit.forecast(steps=forecast_days)
                forecast_dates = pd.date_range(start=store_ts.index[-1] + pd.Timedelta(days=1), 
                                              periods=forecast_days)
                forecast.index = forecast_dates
                
                # Show model summary
                st.subheader("Model Summary")
                st.text(str(model_fit.summary()))
                
                # Plot results
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(store_ts.index, store_ts.values, label='Historical Data')
                ax.plot(forecast.index, forecast.values, label='Forecast', color='red')
                
                ax.set_xlabel('Date')
                ax.set_ylabel('Sales')
                ax.set_title(f'ARIMA Forecast for Store {selected_store}')
                ax.legend()
                ax.grid(True)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Display forecast values
                st.subheader("Forecast Values")
                forecast_df = pd.DataFrame({
                    'Date': forecast.index,
                    'Forecasted Sales': forecast.values
                })
                st.dataframe(forecast_df)
                
                # Add forecasted values display in ARIMA Forecasting section
                st.subheader("Forecasted Sales Amounts")
                st.dataframe(forecast_df)
                
        except Exception as e:
            st.error(f"Error fitting ARIMA model: {e}")
            st.info("Try different parameters or a different data range.")

# Add a footer with information
st.markdown("---")
st.markdown("© 2025 Store Sales Forecasting App | Created as a Final Project for DEPI BY Ahmed Fayad")



