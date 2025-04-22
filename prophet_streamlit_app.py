# prophet_streamlit_app.py

import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.make_holidays import make_holidays_df
import matplotlib.pyplot as plt
from datetime import timedelta

st.set_page_config(page_title='Prophet Forecast Explorer', layout='wide')
st.title('📈 Prophet Forecast Explorer')

st.sidebar.header('Upload Your Time Series Data')

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Ensure 'ds' column is parsed as datetime safely
    try:
        df['ds'] = pd.to_datetime(df['ds'])
    except Exception as e:
        st.error(f"Failed to parse 'ds' column as datetime: {e}")

    st.subheader("Raw Uploaded Data")
    st.dataframe(df.head())

    st.markdown("---")

    st.sidebar.header('Prophet Settings')

    try:
        if 'ds' not in df.columns or not any(col in df.columns for col in ['y', 'y1', 'y2']):
            st.error("CSV must have columns: 'ds' (date) and at least one of 'y', 'y1', or 'y2'")
        else:
            # Choose target series
            y_options = [col for col in ['y', 'y1', 'y2'] if col in df.columns]
            selected_y = st.sidebar.selectbox('Select Target Series', y_options)

            # Forecast settings
            periods_input = st.sidebar.number_input(
                'Forecast Horizon (days)', min_value=1, max_value=365*5, value=365)
            yearly_seasonality = st.sidebar.checkbox('Yearly Seasonality', value=True)
            weekly_seasonality = st.sidebar.checkbox('Weekly Seasonality', value=True)
            daily_seasonality = st.sidebar.checkbox('Daily Seasonality', value=False)
            include_holidays = st.sidebar.checkbox('Include US Holidays + Spillover Effects', value=True)

            # Enable regressor
            use_regressor = False
            if 'y2' in df.columns and selected_y != 'y2':
                use_regressor = st.sidebar.checkbox(
                    f'Use y2 as Regressor for Forecasting {selected_y}?', value=False)

            # Prepare holidays
            holidays = None
            if include_holidays:
                start_year = df['ds'].dt.year.min()
                end_year = df['ds'].dt.year.max()
                holidays = make_holidays_df(
                    year_list=list(range(start_year, end_year + 5)), country='US')

                extra_holidays = []
                def monday_after(date):
                    if date.weekday() in [4, 5, 6]:  # Fri/Sat/Sun
                        return date + timedelta(days=(7 - date.weekday()))
                    return None

                # Boxing Day
                years = range(start_year, end_year + 5)
                boxing_days = pd.to_datetime([f'{yr}-12-26' for yr in years])
                extra_holidays.append(pd.DataFrame({'holiday': 'Boxing Day', 'ds': boxing_days}))

                # Extract standard holidays robustly
                def extract_dates(keyword):
                    return pd.to_datetime(
                        holidays[holidays['holiday'].str.contains(keyword, case=False)]['ds']
                    )
                new_years = extract_dates("New Year")
                christmas = extract_dates("Christmas")
                thanksgiving = extract_dates("Thanksgiving")
                independence = extract_dates("Independence Day")

                # Day after New Year
                extra_holidays.append(pd.DataFrame({
                    'holiday': 'Day After New Year', 'ds': new_years + timedelta(days=1)}))
                # Monday after New Year
                mons = [monday_after(d) for d in new_years if monday_after(d)]
                if mons:
                    extra_holidays.append(pd.DataFrame({'holiday': 'Monday After New Year', 'ds': mons}))
                # Black Friday (day after Thanksgiving)
                extra_holidays.append(pd.DataFrame({
                    'holiday': 'Black Friday', 'ds': thanksgiving + timedelta(days=1)}))
                # Monday after Thanksgiving
                mons = [monday_after(d) for d in thanksgiving if monday_after(d)]
                if mons:
                    extra_holidays.append(pd.DataFrame({'holiday': 'Monday After Thanksgiving', 'ds': mons}))
                # Monday after Christmas
                mons = [monday_after(d) for d in christmas if monday_after(d)]
                if mons:
                    extra_holidays.append(pd.DataFrame({'holiday': 'Monday After Christmas', 'ds': mons}))
                # Monday after Independence Day
                mons = [monday_after(d) for d in independence if monday_after(d)]
                if mons:
                    extra_holidays.append(pd.DataFrame({'holiday': 'Monday After Independence Day', 'ds': mons}))

                # Combine holidays
                if extra_holidays:
                    holidays = pd.concat([holidays] + extra_holidays)

            # Build DataFrame for Prophet
            prophet_df = df[['ds', selected_y]].rename(columns={selected_y: 'y'})
            if use_regressor:
                prophet_df['y2'] = df['y2'].fillna(method='ffill').fillna(method='bfill')

            # Initialize model
            m = Prophet(
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=daily_seasonality,
                holidays=holidays)
            if use_regressor:
                m.add_regressor('y2')

            # Fit
            m.fit(prophet_df)

            # Forecast
            future = m.make_future_dataframe(periods=periods_input)
            if use_regressor:
                future = future.merge(
                    df[['ds', 'y2']], on='ds', how='left')
                future['y2'] = future['y2'].fillna(method='ffill').fillna(method='bfill')

            forecast = m.predict(future)

            # Plot results
            st.subheader(f"Forecast Plot ({selected_y})")
            st.pyplot(m.plot(forecast))
            st.subheader(f"Forecast Components ({selected_y})")
            st.pyplot(m.plot_components(forecast))

            if use_regressor:
                st.subheader("Regressor Influence")
                beta_samples = m.params['beta'][0]
                coef = beta_samples.mean()
                std = beta_samples.std()
                st.write(f"Coefficient for y2: {coef:.4f} ± {std:.4f}")
                fig, ax = plt.subplots()
                ax.bar(['y2'], [coef], yerr=[std], capsize=5)
                ax.axhline(0, color='black', linewidth=0.8)
                ax.set_ylabel('Coefficient')
                ax.set_title('External Regressor Effect')
                st.pyplot(fig)

            # Download
            st.download_button(
                label=f"Download Forecast for {selected_y}",
                data=forecast.to_csv(index=False),
                file_name=f'forecast_{selected_y}.csv',
                mime='text/csv')
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info('👈 Upload a CSV file to get started!')

st.markdown("---")
st.caption("Built with Prophet, Streamlit, and Enhanced Holiday Modeling for ER Forecasting")
