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

    try:
        if 'ds' not in df.columns or 'y' not in df.columns:
            st.error("CSV must have columns: 'ds' (date) and 'y' (numeric value)")
        else:
            periods_input = st.sidebar.number_input('Forecast Horizon (days)', min_value=1, max_value=365*5, value=365)
            yearly_seasonality = st.sidebar.checkbox('Yearly Seasonality', value=True)
            weekly_seasonality = st.sidebar.checkbox('Weekly Seasonality', value=True)
            daily_seasonality = st.sidebar.checkbox('Daily Seasonality', value=False)
            include_holidays = st.sidebar.checkbox('Include US Holidays + Spillover Effects', value=True)

            holidays = None
            if include_holidays:
                start_year = pd.to_datetime(df['ds'].min()).year
                end_year = pd.to_datetime(df['ds'].max()).year
                holidays = make_holidays_df(year_list=list(range(start_year, end_year + 5)), country='US')

                extra_holidays = []

                # Helper to create Monday after a holiday
                def monday_after(date):
                    if date.weekday() in [4, 5, 6]:  # Friday, Saturday, Sunday
                        return date + timedelta(days=(7 - date.weekday()))
                    return None

                # Boxing Day
                years = range(start_year, end_year + 5)
                boxing_days = pd.to_datetime([f'{year}-12-26' for year in years])
                extra_holidays.append(pd.DataFrame({'holiday': 'Boxing Day', 'ds': boxing_days}))

                # Extract holidays
                new_years = pd.to_datetime(holidays[holidays['holiday'] == "New Year's Day"]['ds'])
                christmas = pd.to_datetime(holidays[holidays['holiday'] == 'Christmas Day']['ds'])
                thanksgiving = pd.to_datetime(holidays[holidays['holiday'] == 'Thanksgiving']['ds'])
                independence = pd.to_datetime(holidays[holidays['holiday'] == 'Independence Day']['ds'])

                # Day after New Year's
                day_after_new_years = new_years + timedelta(days=1)
                extra_holidays.append(pd.DataFrame({'holiday': 'Day After New Year', 'ds': day_after_new_years}))

                # Monday after New Year's
                monday_new_years = [monday_after(d) for d in new_years if monday_after(d) is not None]
                if monday_new_years:
                    extra_holidays.append(pd.DataFrame({'holiday': 'Monday After New Year', 'ds': monday_new_years}))

                # Black Friday
                black_friday = thanksgiving + timedelta(days=1)
                extra_holidays.append(pd.DataFrame({'holiday': 'Black Friday', 'ds': black_friday}))

                # Monday after Thanksgiving
                monday_thanksgiving = [monday_after(d) for d in thanksgiving if monday_after(d) is not None]
                if monday_thanksgiving:
                    extra_holidays.append(pd.DataFrame({'holiday': 'Monday After Thanksgiving', 'ds': monday_thanksgiving}))

                # Monday after Christmas
                monday_christmas = [monday_after(d) for d in christmas if monday_after(d) is not None]
                if monday_christmas:
                    extra_holidays.append(pd.DataFrame({'holiday': 'Monday After Christmas', 'ds': monday_christmas}))

                # Monday after Independence Day
                monday_independence = [monday_after(d) for d in independence if monday_after(d) is not None]
                if monday_independence:
                    extra_holidays.append(pd.DataFrame({'holiday': 'Monday After Independence Day', 'ds': monday_independence}))

                # Combine holidays
                if extra_holidays:
                    extra_holidays_df = pd.concat(extra_holidays)
                    holidays = pd.concat([holidays, extra_holidays_df])

            m = Prophet(
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=daily_seasonality,
                holidays=holidays
            )
            m.fit(df)

            future = m.make_future_dataframe(periods=periods_input)
            forecast = m.predict(future)

            st.subheader("Forecast Plot")
            fig1 = m.plot(forecast)
            st.pyplot(fig1)

            st.subheader("Forecast Components")
            fig2 = m.plot_components(forecast)
            st.pyplot(fig2)

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
st.caption("Built with Prophet, Streamlit, and Enhanced Holiday Modeling for ER Forecasting")
