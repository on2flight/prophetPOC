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
        if 'ds' not in df.columns or not any(col in df.columns for col in ['y', 'y1', 'y2']):
            st.error("CSV must have columns: 'ds' (date) and at least one of 'y', 'y1', or 'y2'")
        else:
            y_options = [col for col in ['y', 'y1', 'y2'] if col in df.columns]
            selected_y = st.sidebar.selectbox('Select Target Series', y_options)

            periods_input = st.sidebar.number_input('Forecast Horizon (days)', min_value=1, max_value=365*5, value=365)
            yearly_seasonality = st.sidebar.checkbox('Yearly Seasonality', value=True)
            weekly_seasonality = st.sidebar.checkbox('Weekly Seasonality', value=True)
            daily_seasonality = st.sidebar.checkbox('Daily Seasonality', value=False)
            include_holidays = st.sidebar.checkbox('Include US Holidays + Spillover Effects', value=True)

            use_regressor = False
            if 'y2' in df.columns and selected_y != 'y2':
                use_regressor = st.sidebar.checkbox(f'Use y2 as Regressor for Forecasting {selected_y}?', value=False)

            holidays = None
            if include_holidays:
                start_year = pd.to_datetime(df['ds'].min()).year
                end_year = pd.to_datetime(df['ds'].max()).year
                holidays = make_holidays_df(year_list=list(range(start_year, end_year + 5)), country='US')

                extra_holidays = []

                def monday_after(date):
                    if date.weekday() in [4, 5, 6]:
                        return date + timedelta(days=(7 - date.weekday()))
                    return None

                years = range(start_year, end_year + 5)
                boxing_days = pd.to_datetime([f'{year}-12-26' for year in years])
                extra_holidays.append(pd.DataFrame({'holiday': 'Boxing Day', 'ds': boxing_days}))

                new_years = pd.to_datetime(holidays[holidays['holiday'] == "New Year's Day"]['ds'])
                christmas = pd.to_datetime(holidays[holidays['holiday'] == 'Christmas Day']['ds'])
                thanksgiving = pd.to_datetime(holidays[holidays['holiday'] == 'Thanksgiving']['ds'])
                independence = pd.to_datetime(holidays[holidays['holiday'] == 'Independence Day']['ds'])

                day_after_new_years = new_years + timedelta(days=1)
                extra_holidays.append(pd.DataFrame({'holiday': 'Day After New Year', 'ds': day_after_new_years}))

                monday_new_years = [monday_after(d) for d in new_years if monday_after(d) is not None]
                if monday_new_years:
                    extra_holidays.append(pd.DataFrame({'holiday': 'Monday After New Year', 'ds': monday_new_years}))

                black_friday = thanksgiving + timedelta(days=1)
                extra_holidays.append(pd.DataFrame({'holiday': 'Black Friday', 'ds': black_friday}))

                monday_thanksgiving = [monday_after(d) for d in thanksgiving if monday_after(d) is not None]
                if monday_thanksgiving:
                    extra_holidays.append(pd.DataFrame({'holiday': 'Monday After Thanksgiving', 'ds': monday_thanksgiving}))

                monday_christmas = [monday_after(d) for d in christmas if monday_after(d) is not None]
                if monday_christmas:
                    extra_holidays.append(pd.DataFrame({'holiday': 'Monday After Christmas', 'ds': monday_christmas}))

                monday_independence = [monday_after(d) for d in independence if monday_after(d) is not None]
                if monday_independence:
                    extra_holidays.append(pd.DataFrame({'holiday': 'Monday After Independence Day', 'ds': monday_independence}))

                if extra_holidays:
                    extra_holidays_df = pd.concat(extra_holidays)
                    holidays = pd.concat([holidays, extra_holidays_df])

            prophet_df = df[['ds', selected_y]].rename(columns={selected_y: 'y'})
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])

            if use_regressor:
                prophet_df['y2'] = df['y2']
                prophet_df['y2'].fillna(method='ffill', inplace=True)
                prophet_df['y2'].fillna(method='bfill', inplace=True)

            m = Prophet(
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=daily_seasonality,
                holidays=holidays
            )

            if use_regressor:
                m.add_regressor('y2')

            m.fit(prophet_df)

            future = m.make_future_dataframe(periods=periods_input)

            if use_regressor:
                future = future.merge(df[['ds', 'y2']], on='ds', how='left')
                future['y2'].fillna(method='ffill', inplace=True)
                future['y2'].fillna(method='bfill', inplace=True)

            forecast = m.predict(future)

            st.subheader(f"Forecast Plot ({selected_y})")
            fig1 = m.plot(forecast)
            st.pyplot(fig1)

            st.subheader(f"Forecast Components ({selected_y})")
            fig2 = m.plot_components(forecast)
            st.pyplot(fig2)

            if use_regressor:
                st.subheader("Regressor Coefficient with Confidence Interval")
                regressor_coefs = m.params['beta'][0]
                regressor_std = m.params['beta'][1]
                regressor_name = 'y2'
                st.write(f"Coefficient for {regressor_name}: {regressor_coefs:.4f} ± {regressor_std:.4f}")
                fig3, ax3 = plt.subplots()
                ax3.bar([regressor_name], [regressor_coefs], yerr=[regressor_std], capsize=5)
                ax3.axhline(0, color='black', linewidth=0.8)
                ax3.set_ylabel('Coefficient Value')
                ax3.set_title('External Regressor Influence with Error Bars')
                st.pyplot(fig3)

            csv = forecast.to_csv(index=False)
            st.download_button(
                label=f"Download Forecast for {selected_y} as CSV",
                data=csv,
                file_name=f'forecast_{selected_y}.csv',
                mime='text/csv'
            )

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info('👈 Upload a CSV file to get started!')

st.markdown("---")
st.caption("Built with Prophet, Streamlit, and Enhanced Holiday Modeling for ER Forecasting with Multi-Series and Regressor Support")
