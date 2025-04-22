# prophet_streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

# Forecasting models
from prophet import Prophet
from prophet.make_holidays import make_holidays_df

# Try to import NeuralProphet
try:
    from neuralprophet import NeuralProphet
    neural_available = True
except ImportError:
    neural_available = False

st.set_page_config(page_title='Forecast Explorer', layout='wide')
st.title('📈 Time Series Forecast Explorer')

# Sidebar: Data Upload
st.sidebar.header('Upload Your Data')
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    # Parse dates
    try:
        df['ds'] = pd.to_datetime(df['ds'])
    except Exception as e:
        st.error(f"Failed to parse 'ds' as datetime: {e}")
        st.stop()

    st.subheader("Preview Data")
    st.dataframe(df.head())
    st.markdown('---')

    # Sidebar: Model Selection
    st.sidebar.header('Forecast Settings')
    model_options = ['Prophet']
    if neural_available:
        model_options.append('NeuralProphet')
    model_choice = st.sidebar.selectbox('Select Forecast Model', model_options)

    # Target series
    y_cols = [c for c in ['y','y1','y2'] if c in df.columns]
    target = st.sidebar.selectbox('Target Series', y_cols)
    # Forecast horizon
    periods = st.sidebar.number_input('Forecast Horizon (days)', min_value=1, max_value=365*5, value=365)
    # Seasonality toggles
    yearly = st.sidebar.checkbox('Yearly Seasonality', True)
    weekly = st.sidebar.checkbox('Weekly Seasonality', True)
    daily = st.sidebar.checkbox('Daily Seasonality', False)
    # Changepoint settings
    st.sidebar.header('Changepoint Settings')
    cp_scale = st.sidebar.number_input('Changepoint Prior Scale', 0.001, 1.0, 0.05, 0.01)
    n_changepoints = st.sidebar.number_input('Number of Automatic Changepoints', 0, 100, 25, 1)
    custom_cps = st.sidebar.text_input('Custom Changepoints (YYYY-MM-DD, comma-separated)', '')
    # Holiday effects
    include_holidays = st.sidebar.checkbox('Include US Holiday Effects', True)
    # Regressor
    use_y2 = False
    if 'y2' in df.columns and target != 'y2':
        use_y2 = st.sidebar.checkbox(f'Use y2 as Regressor for {target}?', False)
    # Backtesting
    do_backtest = st.sidebar.checkbox('Enable Backtesting', False)
    backtest_days = 0
    if do_backtest:
        max_bt = len(df) - 1
        backtest_days = st.sidebar.number_input('Backtest Days', 1, max_bt, min(90, max_bt))

    # Build holiday DataFrame
    holidays = None
    if include_holidays:
        years = list(range(df['ds'].dt.year.min(), df['ds'].dt.year.max() + 5))
        holidays = make_holidays_df(year_list=years, country='US')
        extra = []
        def monday_after(d): return d + timedelta(days=(7-d.weekday())) if d.weekday()>=5 else None
        bd = pd.to_datetime([f'{yr}-12-26' for yr in years])
        extra.append(pd.DataFrame({'holiday':'Boxing Day','ds':bd}))
        def ext(kw): return pd.to_datetime(holidays[holidays['holiday'].str.contains(kw, case=False)]['ds'])
        ny, ch, th, ind = ext('New Year'), ext('Christmas'), ext('Thanksgiving'), ext('Independence Day')
        extra.append(pd.DataFrame({'holiday':'Day After New Year','ds':ny+timedelta(days=1)}))
        for name, arr in [('Mon After NY', [monday_after(d) for d in ny if monday_after(d)]),
                          ('Black Friday',[d+timedelta(days=1) for d in th]),
                          ('Mon After Thanksgiving',[monday_after(d) for d in th if monday_after(d)]),
                          ('Mon After Christmas',[monday_after(d) for d in ch if monday_after(d)]),
                          ('Mon After Independence',[monday_after(d) for d in ind if monday_after(d)])]:
            if any(arr): extra.append(pd.DataFrame({'holiday':name,'ds':arr}))
        holidays = pd.concat([holidays]+extra, ignore_index=True)

    # Prepare modeling DataFrame
    df_model = df[['ds', target]].rename(columns={target:'y'})
    if use_y2:
        df_model['y2'] = df['y2'].fillna(method='ffill').fillna(method='bfill')

    # Parse custom changepoints
    changepoints = None
    if custom_cps:
        try:
            changepoints = [pd.to_datetime(dt.strip()) for dt in custom_cps.split(',')]
        except:
            st.warning('Ignoring invalid custom changepoints')

    # Prophet fit function
    def fit_prophet(data):
        m = Prophet(
            yearly_seasonality=yearly,
            weekly_seasonality=weekly,
            daily_seasonality=daily,
            changepoint_prior_scale=cp_scale,
            n_changepoints=(None if changepoints else n_changepoints),
            changepoints=changepoints,
            holidays=holidays)
        if use_y2: m.add_regressor('y2')
        m.fit(data)
        return m

    # NeuralProphet fit function
    def fit_neural(data):
        if not neural_available:
            st.error('NeuralProphet is not installed in this environment.')
            return None
        m = NeuralProphet(
            yearly_seasonality=yearly,
            weekly_seasonality=weekly,
            daily_seasonality=daily,
            n_changepoints=n_changepoints,
            changepoints=changepoints)
        if use_y2: m.add_lagged_regressor(names='y2', lags=0)
        metrics = m.fit(data, freq='D')
        return m

    # Choose and fit model
    if model_choice == 'Prophet':
        m = fit_prophet(df_model)
    else:
        m = fit_neural(df_model)
        if m is None:
            st.stop()

    # Backtesting if enabled
    if do_backtest and backtest_days > 0:
        train = df_model[:-backtest_days]; test = df_model[-backtest_days:]
        if model_choice == 'Prophet':
            m_bt = fit_prophet(train)
            future_bt = m_bt.make_future_dataframe(periods=backtest_days, include_history=False)
        else:
            m_bt = fit_neural(train)
            future_bt = m_bt.make_future_dataframe(train, periods=backtest_days)
        if use_y2:
            future_bt = future_bt.merge(df[['ds','y2']], on='ds', how='left')
            future_bt['y2'] = future_bt['y2'].fillna(method='ffill').fillna(method='bfill')
        fc_bt = m_bt.predict(future_bt)
        pred = fc_bt['yhat1'] if model_choice == 'NeuralProphet' else fc_bt['yhat']
        res = test.set_index('ds').assign(yhat=pred.values)
        res['residual'] = res['y'] - res['yhat']
        st.subheader('Backtest Metrics')
        st.write(f"MAE: {res['residual'].abs().mean():.2f}, RMSE: {np.sqrt((res['residual']**2).mean()):.2f}")
        # Rolling residuals
        window = min(7, backtest_days)
        res['roll_res'] = res['residual'].rolling(window, min_periods=1).mean()
        fig, ax = plt.subplots(); ax.plot(res.index, res['roll_res']); ax.set_title(f'{window}-Day Rolling Mean Residual'); st.pyplot(fig)
        st.markdown('---')

    # Forecast future
    if model_choice == 'Prophet':
        future = m.make_future_dataframe(periods=periods)
    else:
        future = m.make_future_dataframe(df_model, periods=periods)
    if use_y2:
        future = future.merge(df[['ds','y2']], on='ds', how='left')
        future['y2'] = future['y2'].fillna(method='ffill').fillna(method='bfill')
    forecast = m.predict(future)

        # Plot results
    st.subheader(f"Forecast ({model_choice}) for {target}")
    if model_choice == 'Prophet':
        fig = m.plot(forecast)
        st.pyplot(fig)
    else:
        fig = m.plot(forecast)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader('Forecast Components')
    if model_choice == 'Prophet':
        comp_fig = m.plot_components(forecast)
        st.pyplot(comp_fig)
    else:
        comp_fig = m.plot_components(forecast)
        st.plotly_chart(comp_fig, use_container_width=True)

    # Download
    pred_col = 'yhat1' if model_choice == 'NeuralProphet' else 'yhat'
    out = forecast[['ds', pred_col]].rename(columns={pred_col: target})
    st.download_button('Download Forecast', data=out.to_csv(index=False),
                       file_name=f'forecast_{model_choice}_{target}.csv', mime='text/csv')('Download Forecast', data=out.to_csv(index=False),
                       file_name=f'forecast_{model_choice}_{target}.csv', mime='text/csv')
else:
    st.info('Please upload a CSV to begin.')

st.markdown('---')
st.caption('Supports Prophet & NeuralProphet (if installed), changepoints, holiday effects, backtesting, rolling diagnostics')
