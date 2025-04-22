# prophet_streamlit_app.py

import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.make_holidays import make_holidays_df
import matplotlib.pyplot as plt
from datetime import timedelta
import numpy as np

st.set_page_config(page_title='Prophet Forecast Explorer', layout='wide')
st.title('📈 Prophet Forecast Explorer')

# Sidebar: data upload
st.sidebar.header('Upload Your Time Series Data')
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    # parse ds
    try:
        df['ds'] = pd.to_datetime(df['ds'])
    except Exception as e:
        st.error(f"Failed to parse 'ds' column as datetime: {e}")
        st.stop()

    st.subheader("Raw Uploaded Data")
    st.dataframe(df.head())
    st.markdown("---")

    # Sidebar: model settings
    st.sidebar.header('Prophet Settings')
    # target series
    y_cols = [col for col in ['y', 'y1', 'y2'] if col in df.columns]
    selected_y = st.sidebar.selectbox('Select Target Series', y_cols)
    # forecast horizon
    periods = st.sidebar.number_input('Forecast Horizon (days)', min_value=1, max_value=365*5, value=365)
    # seasonality toggles
    yearly = st.sidebar.checkbox('Yearly Seasonality', value=True)
    weekly = st.sidebar.checkbox('Weekly Seasonality', value=True)
    daily = st.sidebar.checkbox('Daily Seasonality', value=False)
    # holiday effects
    include_holidays = st.sidebar.checkbox('Include US Holidays + Spillover', value=True)
    # regressors
    use_y2 = False
    if 'y2' in df.columns and selected_y != 'y2':
        use_y2 = st.sidebar.checkbox(f'Use y2 as Regressor for {selected_y}?', value=False)
    # backtesting settings
    do_backtest = st.sidebar.checkbox('Enable Backtesting', value=False)
    backtest_days = 0
    if do_backtest:
        max_bt = len(df) - 1
        backtest_days = st.sidebar.number_input('Backtest Period (days)', min_value=1, max_value=max_bt, value=min(90, max_bt))

    # Prepare holiday dataframe
    holidays = None
    if include_holidays:
        years = list(range(df['ds'].dt.year.min(), df['ds'].dt.year.max() + 5))
        holidays = make_holidays_df(year_list=years, country='US')
        extra = []
        def monday_after(d):
            return d + timedelta(days=(7 - d.weekday())) if d.weekday() in [4,5,6] else None
        # Boxing Day
        bd = pd.to_datetime([f'{yr}-12-26' for yr in years])
        extra.append(pd.DataFrame({'holiday':'Boxing Day','ds':bd}))
        # extract by keyword
        def ext(kw):
            return pd.to_datetime(holidays[holidays['holiday'].str.contains(kw, case=False)]['ds'])
        ny = ext('New Year'); ch = ext('Christmas'); th = ext('Thanksgiving'); ind = ext('Independence Day')
        # add spillovers
        extra.append(pd.DataFrame({'holiday':'Day After New Year','ds':ny + timedelta(days=1)}))
        mny = [monday_after(d) for d in ny if monday_after(d)];
        if mny: extra.append(pd.DataFrame({'holiday':'Monday After New Year','ds':mny}))
        extra.append(pd.DataFrame({'holiday':'Black Friday','ds':th + timedelta(days=1)}))
        mth = [monday_after(d) for d in th if monday_after(d)];
        if mth: extra.append(pd.DataFrame({'holiday':'Monday After Thanksgiving','ds':mth}))
        mch = [monday_after(d) for d in ch if monday_after(d)];
        if mch: extra.append(pd.DataFrame({'holiday':'Monday After Christmas','ds':mch}))
        mind = [monday_after(d) for d in ind if monday_after(d)];
        if mind: extra.append(pd.DataFrame({'holiday':'Monday After Independence Day','ds':mind}))
        if extra:
            holidays = pd.concat([holidays] + extra, ignore_index=True)

    # Build modeling dataframe
    df_model = df[['ds', selected_y]].rename(columns={selected_y:'y'})
    if use_y2:
        df_model['y2'] = df['y2'].fillna(method='ffill').fillna(method='bfill')

    # Backtesting
    if do_backtest and backtest_days > 0:
        train = df_model.iloc[:-backtest_days]
        test = df_model.iloc[-backtest_days:]
        m_bt = Prophet(yearly_seasonality=yearly, weekly_seasonality=weekly,
                       daily_seasonality=daily, holidays=holidays)
        if use_y2: m_bt.add_regressor('y2')
        m_bt.fit(train)
        future_bt = m_bt.make_future_dataframe(periods=backtest_days, include_history=False)
        if use_y2:
            future_bt = future_bt.merge(df[['ds','y2']], on='ds', how='left')
            future_bt['y2'] = future_bt['y2'].fillna(method='ffill').fillna(method='bfill')
        fc_bt = m_bt.predict(future_bt)
        res = test.set_index('ds').join(fc_bt.set_index('ds')[['yhat']])
        res['residual'] = res['y'] - res['yhat']
        mae = res['residual'].abs().mean()
        rmse = np.sqrt((res['residual']**2).mean())
        st.subheader('Backtest Performance')
        st.write(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        fig1, ax1 = plt.subplots()
        ax1.plot(res.index, res['residual'])
        ax1.set_title('Residuals Over Backtest')
        st.pyplot(fig1)
        fig2, ax2 = plt.subplots()
        ax2.hist(res['residual'], bins=30)
        ax2.set_title('Residuals Distribution')
        st.pyplot(fig2)
        # Residual analysis tables
        res_analysis = res.copy()
        res_analysis['day_of_week'] = res_analysis.index.day_name()
        if include_holidays and holidays is not None:
            hol_map = holidays.set_index('ds')['holiday'].to_dict()
            res_analysis['holiday'] = res_analysis.index.map(lambda d: hol_map.get(d, ''))
        high_errors = res_analysis.reindex(res_analysis['residual'].abs().sort_values(ascending=False).index)
        st.subheader('Top High-Error Dates')
        st.table(high_errors[['y','yhat','residual','day_of_week','holiday']].head(10))
        err_by_dow = res_analysis.groupby('day_of_week')['residual'].apply(lambda x: x.abs().mean()).sort_values(ascending=False)
        st.subheader('Mean Absolute Error by Day of Week')
        st.bar_chart(err_by_dow)
        st.markdown('---')

    # Full model fit and forecast
    m = Prophet(yearly_seasonality=yearly, weekly_seasonality=weekly,
                daily_seasonality=daily, holidays=holidays)
    if use_y2: m.add_regressor('y2')
    m.fit(df_model)
    future = m.make_future_dataframe(periods=periods)
    if use_y2:
        future = future.merge(df[['ds','y2']], on='ds', how='left')
        future['y2'] = future['y2'].fillna(method='ffill').fillna(method='bfill')
    forecast = m.predict(future)

    # Display forecasts
    st.subheader(f"Forecast Plot ({selected_y})")
    st.pyplot(m.plot(forecast))
    st.subheader(f"Forecast Components ({selected_y})")
    st.pyplot(m.plot_components(forecast))

    # Download
    st.download_button('Download Forecast', data=forecast.to_csv(index=False),
                       file_name=f'forecast_{selected_y}.csv', mime='text/csv')
else:
    st.info('👈 Upload a CSV file to get started!')

st.markdown('---')
st.caption('Built with Prophet, Streamlit, plus backtesting and residual diagnostics')
