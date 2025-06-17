Expert Time Series Analysis Prompt for Memecoin Data
You are an expert data scientist specializing in time series analysis and deep learning. You have a dataset of 10,000 memecoins stored as individual parquet files.

Dataset Specifications:
File Format: Each memecoin has its own parquet file
Data Coverage: ONLY the first 24 hours after token launch (minute-by-minute)
Columns: Exactly 2 columns - "price" and "datetime" (timestamp)
Temporal Range: Tokens launched between 2022-2025 (3+ year span)
No Metadata: No volume, liquidity, holders, or any other data - ONLY PRICE ACTION

Critical Considerations:
Market Evolution: A token launched in 2022 faced very different market conditions than one in 2025
Pure Price Action: All analysis must be derived from price movements alone
Launch Date Matters: Extract and consider the launch date from timestamps as a critical feature

Your Task: Comprehensive Time Series Analysis Pipeline
Analysis Modes Required:
Single Token Analysis: Deep dive into individual token behavior
Multi-Token Comparison: Compare selected tokens (2-100)
Full Dataset Analysis: Patterns across all 10,000 tokens

Phase 0: Data Quality Assessment and Cleaning
Data Quality Checks
Verify exactly 2 columns: "price" and "datetime"
Check timestamp continuity (should have ~1,440 points for 24 hours)
Identify gaps in minute-by-minute data
Detect zero or negative prices
Find duplicate timestamps
Find dead token : token with price that stops changing
After how many hours after the launch the "dead tokens" died
Verify chronological order
Extract launch date/time from first timestamp
Launch Context Extraction:
python
# Critical: Extract contextual information
- launch_year = datetime.year
- launch_month = datetime.month
- launch_day_of_week = datetime.dayofweek
- launch_hour = datetime.hour
- is_weekend = launch_day_of_week in [5, 6]
- market_era = categorize_by_year(launch_year)  # "2022_bear", "2023_recovery", "2024_bull", etc.

Data Cleaning Strategy
Missing Minutes:
Single gaps: Linear interpolation between adjacent prices
2-5 minute gaps: Polynomial interpolation (order 2)
5-10 minute gaps: Forward fill then linear interpolation
10 minute gaps: Flag segment, consider excluding

Price Anomalies:
Calculate returns: returns = price.pct_change()
Flag returns > 1000% or < -90% in single minute
Flag extremes as token with issues :  > 10k% from a minute to the next or < 99.9% 
For outliers: Replace with median of surrounding 5 minutes
Keep anomaly count as data quality metric

Phase 1: Feature Engineering (Price-Only Features)
Since we ONLY have price data, engineer creative features:
Basic Price Metrics:
python
- initial_price = price[0]
- final_price = price[-1]
- max_price = price.max()
- min_price = price.min()
- normalized_prices = price / initial_price  # All analysis on normalized
Temporal Price Features:
python
- time_to_peak = argmax(price)
- time_to_bottom = argmin(price)
- time_above_launch = sum(price > initial_price)
- peak_duration = consecutive_minutes_near_peak(price, threshold=0.9)
Price Movement Patterns:
python
- returns = price.pct_change()
- volatility_5m = returns.rolling(5).std()
- volatility_30m = returns.rolling(30).std()
- max_drawdown = calculate_max_drawdown(price)
- recovery_time = time_to_recover_from_drawdown()
Price Action Characteristics:
python
- pump_velocity = max_price_increase_per_minute()
- dump_velocity = max_price_decrease_per_minute()
- price_efficiency = distance_traveled / displacement
- fractal_dimension = estimate_price_roughness()
- momentum_shifts = count_trend_changes()
Market Context Features (from launch timestamp):
python
- launch_year_encoded = one_hot_encode(launch_year)
- crypto_season = map_to_market_cycle(launch_date)
- time_since_btc_halving = days_from_nearest_halving(launch_date)

Phase 2: Analysis Approaches by Mode
Mode 1: Single Token Analysis
python
def analyze_single_token(parquet_file):
    # Load and clean data
    df = pd.read_parquet(parquet_file)
    
    # Extract all features
    features = extract_price_features(df)
    
    # Generate visualizations:
    # - Price chart with key moments marked
    # - Return distribution
    # - Volatility evolution
    # - Momentum indicators
    
    # Identify pattern type (pump & dump, steady growth, etc.)
    pattern = classify_price_pattern(features)
    
    return detailed_report
Mode 2: Multi-Token Comparison (2-100 tokens)
python
def compare_tokens(parquet_files_list):
    # Load all tokens
    tokens_data = [load_and_process(f) for f in parquet_files_list]
    
    # Align by launch time (minute 0 = launch for all)
    aligned_data = align_by_launch_time(tokens_data)
    
    # Group by launch era for fair comparison
    grouped = group_by_market_conditions(tokens_data)
    
    # Comparative analysis:
    # - Overlay normalized price charts
    # - Compare volatility profiles
    # - Rank by various metrics
    # - Identify common patterns
    
    return comparison_matrix
Mode 3: Full Dataset Analysis (10,000 tokens)
python
def analyze_all_tokens(all_parquet_files):
    # Process in batches to manage memory
    batch_size = 100
    
    # Extract features for all tokens
    all_features = []
    for batch in chunks(all_parquet_files, batch_size):
        batch_features = parallel_process(batch)
        all_features.extend(batch_features)
    
    # Create feature matrix
    feature_matrix = pd.DataFrame(all_features)
    
    # Segment by launch period
    era_segments = segment_by_launch_era(feature_matrix)
    
    # Analyze patterns by era
    for era, data in era_segments.items():
        analyze_era_patterns(data)
    
    # Global patterns across all eras
    global_patterns = find_universal_patterns(feature_matrix)
    
    return comprehensive_report

Phase 3: Pattern Recognition and Clustering
Price Pattern Archetypes (using ONLY price):
python
patterns = {
    "explosive_pump": max_gain > 1000% and time_to_peak < 60,
    "steady_climb": consistent_positive_momentum and low_volatility,
    "pump_and_dump": high_peak and final_price < 0.2 * max_price,
    "volatile_survivor": high_volatility but final_price > initial_price,
    "slow_death": gradual_decline with no recovery,
    "instant_fail": never exceeds 1.2x launch price
}
Era-Adjusted Clustering:
Normalize features within each launch year
Account for different market conditions
Use temporal-aware clustering
Phase 4: Deep Learning with Temporal Context
Input Preparation:
python
# For each token
input_features = {
    'price_sequence': normalized_prices[:N],  # First N minutes
    'launch_context': [year, month, day_of_week, hour],
    'market_era': era_encoding
}
Architecture for Price-Only Data:
python
class PriceOnlyPredictor(nn.Module):
    def __init__(self):
        # Price sequence encoder
        self.price_lstm = nn.LSTM(1, 128, 2)
        
        # Context encoder
        self.context_embed = nn.Linear(context_dim, 64)
        
        # Attention mechanism for critical moments
        self.attention = nn.MultiheadAttention(128, 8)
        
        # Prediction heads
        self.trajectory_head = nn.Linear(192, future_steps)
        self.outcome_head = nn.Linear(192, n_classes)

Training Considerations:
Temporal Split: Train on 2022-2023, validate on 2024, test on 2025 (this may need adjustments because we have a lot more token from 2025 than the years before)
Era Stratification: Ensure each era is represented in train/val/test
Data Augmentation: Add noise, time-shift, amplitude scaling
Phase 5: Trading Strategy Development
Entry Signals (from price alone):
python
entry_conditions = {
    'momentum_breakout': price > 1.5x and accelerating,
    'volatility_expansion': volatility > 2x baseline,
    'pattern_recognition': matches successful archetype
}
Risk Management:
Position sizing based on volatility
Stop-loss levels from price action
Take-profit targets from historical patterns
Special Considerations for Price-Only Analysis:
Volume Proxy: Use price volatility as volume proxy
python
volume_proxy = abs(returns) * price  # Larger moves = higher "volume"
Liquidity Proxy: Price efficiency as liquidity indicator
python
liquidity_proxy = 1 / (sum(abs(returns)) / abs(final_return))
Market Regime Detection: From price action alone
python
regime = detect_regime(volatility_pattern, momentum_pattern)
Spread it can be up to 20% for some memecoin with very low liquidity 
gas fees but negligeable on Solana

Output Requirements:
For Single Token:
Complete price action report
Pattern classification
Key moments identified
Risk metrics
For Multi-Token:
Comparative visualizations
Relative performance metrics
Pattern similarity scores
Launch context impact
For All Tokens:
Statistical distributions by era
Success rate evolution over time
Universal patterns vs era-specific
Predictive model performance by market condition
Remember: We're building a system that can work with ONLY price data, but must be smart about extracting maximum information from price movements and their temporal context.

Phase 0: Data Quality Assessment and Cleaning
Data Quality Checks
Check for missing timestamps (gaps in minute-by-minute data)
Identify zero or negative prices (invalid data)
Detect duplicate timestamps
Find extreme price jumps (>1000% in 1 minute) that might indicate data errors
Calculate completeness percentage for each series
Identify series that are too short (<60 minutes of data)
Data Cleaning Strategy
Missing Values Imputation:
For single missing points: Use linear interpolation
For gaps 2-5 minutes: Use polynomial interpolation (order 2)
For gaps >5 minutes: Forward fill followed by linear interpolation
If >10% data missing: Flag series as low quality
Outlier Treatment:
Use Interquartile Range (IQR) method on returns, not prices
For extreme outliers (>10x IQR): Replace with local median (5-minute window)
Keep track of all modifications in a separate column
Zero/Negative Prices:
If isolated: Replace with average of neighboring values
If consecutive: Mark entire period as invalid
If >5% of series: Exclude from analysis
Data Validation
Ensure all timestamps are properly ordered
Verify price continuity (no impossible jumps)
Create quality score for each series (0-1) based on completeness and modifications
Phase 1: Feature Engineering (Per Coin)
Extract time-invariant features from each 24-hour series:

Price-based features:
Initial price, final price, max price, min price
Time to peak price (in minutes)
Maximum gain percentage from launch
Final return percentage
Number of times price doubled
Volatility features:
Standard deviation of returns (5-min, 30-min, full period)
Average True Range (ATR) proxy
Largest single-minute move (up and down)
Volatility clustering coefficient
Momentum features:
Consecutive positive/negative minutes (max streak)
Number of momentum reversals
Average momentum over different windows
Acceleration metrics (momentum of momentum)
Pattern features:
Time spent above initial price
Number of local peaks/troughs
Hurst exponent (trending vs mean-reverting)
Autocorrelation at various lags
Microstructure features:
Price efficiency ratio
Average bid-ask spread estimate (Roll's measure)
Kyle's lambda (price impact estimate)
Phase 2: Clustering/Segmentation
Group coins by similar behavior patterns:

Preprocessing:
Normalize all features using StandardScaler
Apply PCA if features > 50 to reduce dimensionality
Handle outliers in feature space
Clustering approach:
Use multiple algorithms: K-means, DBSCAN, Gaussian Mixture Models
Determine optimal clusters using silhouette score and elbow method
Expected archetypes: "pump & dump", "steady growth", "instant death", "volatile survivor"
Validation:
Visualize clusters using t-SNE or UMAP
Analyze feature importance per cluster
Ensure clusters are interpretable and actionable
Phase 3: Panel Data Modeling
Structure data for cross-sectional time series analysis:

Data Structure:
Columns: [coin_id, minutes_since_launch, price, returns, cluster_label, features...]
Alignment:
All series aligned by launch time (t=0 at launch)
NOT by calendar time
Normalize prices by initial price (price_t / price_0)
Panel Models:
Fixed effects by cluster
Random effects for individual coins
Dynamic panel models for momentum effects
Phase 4: Sequential Deep Learning Models
Design models for real-time prediction:

Problem Formulation:
Input: First N minutes of normalized price data (N = 10, 30, 60)
Output: Either next M minutes prediction OR final outcome classification
Architecture Recommendations:
python
Model Architecture Options:
1. LSTM with attention mechanism
2. Transformer (better for capturing "critical moments")
3. CNN-LSTM hybrid for local pattern detection
4. Temporal Convolutional Networks (TCN)
Training Strategy:
Train/Val/Test split by launch DATE (not random) to avoid temporal leakage
Use time-based cross-validation
Augment data with sliding windows
Multi-task Learning:
Task 1: Predict price trajectory (regression)
Task 2: Classify final outcome (pump & dump vs sustainable)
Share lower layers, separate heads for each task
Phase 5: Ensemble and Production Pipeline
Ensemble Strategy:
Cluster-specific models for each archetype
Global model for general patterns
Weighted average based on cluster confidence
Real-time Implementation:
Streaming architecture for new data
Model updates with sliding window
Confidence intervals for predictions

Critical Guidelines:
DO:
Treat each coin as individual time series with common patterns
Align all series by "minutes since launch"
Use robust methods for outliers and missing data
Validate on future coins (temporal validation)
Consider market regime when interpreting results
DON'T:
Don't concatenate all series into one long series
Don't use calendar time for alignment
Don't average prices across coins (different scales)
Don't assume stationarity
Don't ignore the survival bias in successful coins
Expected Deliverables:
Data Quality Report: Statistics on missing data, outliers, and cleaning performed
Feature Importance Analysis: Which features best predict success
Cluster Profiles: Characteristics of each memecoin archetype
Model Performance: Metrics for both regression and classification tasks
Trading Strategy: Based on model predictions, when to enter/exit positions
Performance Metrics:
For price prediction: RMSE, MAE, directional accuracy
For classification: Precision, Recall, F1, ROC-AUC
For trading: Sharpe ratio, maximum drawdown, win rate
Remember: The goal is to identify patterns in the first N minutes that predict success/failure, enabling profitable trading decisions in real-time.

