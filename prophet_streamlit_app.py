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

    st.subheader("Raw Uploaded Data")
    st.dataframe(df.head())

    st.markdown("---")

    st.sidebar.header('Prophet Settings')

    # Prophet expects columns 'ds' (date) and 'y' (value)
    try:
        if 'ds' not in df.columns or 'y' not in df.columns:
            st.error("CSV must have columns: 'ds' (date) and 'y' (numeric value)")
        else:
            # Allow setting parameters
            periods_input = st.sidebar.number_input('Forecast Horizon (days)', min_value=1, max_value=365*5, value=365)
            yearly_seasonality = st.sidebar.checkbox('Yearly Seasonality', value=True)
            weekly_seasonality = st.sidebar.checkbox('Weekly Seasonality', value=True)
            daily_seasonality = st.sidebar.checkbox('Daily Seasonality', value=False)
            include_holidays = st.sidebar.checkbox('Include US Holidays + Spillover Effects', value=True)

            # Setup holidays if requested
            holidays = None
            if include_holidays:
                # Built-in US holidays
                start_year = pd.to_datetime(df['ds'].min()).year
                end_year = pd.to_datetime(df['ds'].max()).year
                holidays = make_holidays_df(year_list=list(range(start_year, end_year + 5)), country='US')

                # Custom lagged holidays
                extra_holidays = []

                # Day after New Year's
                new_years = pd.to_datetime(holidays[holidays['holiday'] == "New Year's Day"]['ds'])
                day_after_new_years = new_years + timedelta(days=1)
                extra_holidays.append(pd.DataFrame({'holiday': 'Day After New Year', 'ds': day_after_new_years}))

                # Black Friday (Day after Thanksgiving)
                thanksgiving = pd.to_datetime(holidays[holidays['holiday'] == 'Thanksgiving']['ds'])
                black_friday = thanksgiving + timedelta(days=1)
                extra_holidays.append(pd.DataFrame({'holiday': 'Black Friday', 'ds': black_friday}))

                # Monday after July 4th (only if Independence Day falls on Friday or Weekend)
                independence = pd.to_datetime(holidays[holidays['holiday'] == 'Independence Day']['ds'])
                for day in independence:
                    if day.weekday() in [5, 6]:  # Saturday or Sunday
                        spillover = day + timedelta(days=(7 - day.weekday()))  # Next Monday
                        extra_holidays.append(pd.DataFrame({'holiday': 'Monday After Independence Day', 'ds': [spillover]}))

                # Combine all holidays
                if extra_holidays:
                    extra_holidays_df = pd.concat(extra_holidays)
                    holidays = pd.concat([holidays, extra_holidays_df])

            # Fit model
            m = Prophet(
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=daily_seasonality,
                holidays=holidays
            )
            m.fit(df)

            # Future dataframe
            future = m.make_future_dataframe(periods=periods_input)
            forecast = m.predict(future)

            st.subheader("Forecast Plot")
            fig1 = m.plot(forecast)
            st.pyplot(fig1)

            st.subheader("Forecast Components")
            fig2 = m.plot_components(forecast)
            st.pyplot(fig2)

            # Option to download forecast
            csv = forecast.to_csv(index=False)
            st.download_button(
                label="Download Forecast as CSV",
                data=csv,
                file_name='forecast.csv',
                mime='text/csv'
            )

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info('👈 Upload a CSV file to get started!')

st.markdown("---")
st.caption("Built with Prophet, Streamlit, and US Holiday Modeling | Customize freely!")
