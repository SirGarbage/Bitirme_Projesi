import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import warnings

#Hataları filtreler
warnings.filterwarnings('ignore')

# Sayfanın kurulumu
st.set_page_config(page_title="Turkey Population & GDP Forecaster", layout="wide")
st.title("🇹🇷 Future Projection Dashboard")
st.markdown("Estimate population, economic growth, and **sectoral shifts** using Facebook Prophet.")

# Ayarlar
st.sidebar.header("Simülasyon Ayarları")

# İşlenmiş, kullanmaya uygun veriyi bulunduran dosyayı alırız
INPUT_FILE = '.venv/Prophet_Training_Set_Sektorlu_USD.xlsx'

#Yıl seçimi yapılır
years_to_predict = st.sidebar.slider("Tahmin öngörüsü (Yıl)", 1, 30, 5)

# Kriz senaryoları için ek menüw
st.sidebar.subheader("Kriz senaryoları")
enable_crash = st.sidebar.checkbox("Enable 'Sudden Event' Scenario")

if enable_crash:
    crash_type = st.sidebar.radio("Event Type", ["Economic Crash (GDP)", "Population Decline", "Both"])
    crash_year = st.sidebar.number_input("Year of Event", min_value=2024, max_value=2050, value=2026)
    crash_severity = st.sidebar.slider("Severity (% Drop)", 1, 90, 20) / 100
else:
    crash_type = None
    crash_year = None
    crash_severity = 0


#Veriler yüklenir
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_excel(file_path, sheet_name='Iller_Verisi')
        return df
    except Exception as e:
        return None


df = load_data(INPUT_FILE)

if df is None:
    st.error(f"Dosya yüklenemedi")
    st.stop()

#Şehir seçimi yapımı için
city_list = sorted(df['İl'].unique())
selected_city = st.selectbox("Analiz için şehir seçiniz:", city_list)

#Veri şehir için filtrelenir
df_city = df[df['İl'] == selected_city].copy()

#Sektör dağılımı yapılır
all_sector_cols = [col for col in df_city.columns if col.startswith('Pay_')]
main_sector_cols = [col for col in all_sector_cols if col != 'Pay_Imalat']


# Prophet için temel fonksiyonlar
def run_prophet(df_input, target_col, years, crash_on, c_year, c_sev, is_gdp=False):
    df_p = df_input[['ds', target_col]].rename(columns={target_col: 'y'})
    if len(df_p) < 2: return None, None

    m = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
    m.fit(df_p)

    future = m.make_future_dataframe(periods=years, freq='YE')
    forecast = m.predict(future)

    # Kriz senaryolarının uygulanıp uygulanmadığını kontrol eder, uygulanırsa etkilerini uygular
    if crash_on:
        apply_crash = False
        if is_gdp and (crash_type in ["Economic Crash (GDP)", "Both"]):
            apply_crash = True
        elif not is_gdp and (crash_type in ["Population Decline", "Both"]):
            apply_crash = True

        if apply_crash:
            mask = forecast['ds'].dt.year >= c_year
            forecast.loc[mask, ['yhat', 'yhat_lower', 'yhat_upper']] *= (1 - c_sev)

    return m, forecast


# Sektörler büyüyüp küçülürken bazen matematiksel olarak %100 değerinin üstüne çıkıyor, burada normalizasyon ile dağıtım yapıyoruz
def forecast_sector_trends(df_input, sectors, years):

    # Dummy bir model ile gelecek oluşturulur
    m_dummy = Prophet(yearly_seasonality=True)
    m_dummy.fit(df_input[['ds']].assign(y=0))
    future_dates = m_dummy.make_future_dataframe(periods=years, freq='YE')

    forecast_results = pd.DataFrame({'ds': future_dates['ds']})

    # Sektörlere, her biri için ayrı olarak forcast uygulanır
    progress_bar = st.progress(0)

    for i, sector in enumerate(sectors):
        df_s = df_input[['ds', sector]].rename(columns={sector: 'y'})

        # Model koyulur
        m = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
        m.fit(df_s)
        fcst = m.predict(future_dates)

        # Negatif değerler kaldırılır
        forecast_results[sector] = fcst['yhat'].clip(lower=0)

        progress_bar.progress((i + 1) / len(sectors))

    progress_bar.empty()

    forecast_results['Total_Sum'] = forecast_results[sectors].sum(axis=1)

    # Sektörler bölünür, toplamda 100e yuvarlanır/100e göre dağıtılır
    for sector in sectors:
        forecast_results[sector] = (forecast_results[sector] / forecast_results['Total_Sum']) * 100

    return forecast_results


#Burası layout'ın tasarımı için kodları içerir
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Nüfus Tahmini")
    with st.spinner('Nüfus hesaplanıyor. Lütfen bekleyiniz.'):
        m_pop, f_pop = run_prophet(df_city, 'y', years_to_predict, enable_crash, crash_year, crash_severity)

    if f_pop is not None:
        future_val = f_pop.iloc[-1]['yhat']
        growth = ((future_val - f_pop.iloc[-years_to_predict - 1]['yhat']) / f_pop.iloc[-years_to_predict - 1][
            'yhat']) * 100

        st.metric("Est. Population", f"{int(future_val):,}", f"{growth:.2f}% Growth")

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(df_city['ds'], df_city['y'], 'ko', label='History')
        ax1.plot(f_pop['ds'], f_pop['yhat'], 'b-', linewidth=2, label='Forecast')
        ax1.fill_between(f_pop['ds'], f_pop['yhat_lower'], f_pop['yhat_upper'], color='blue', alpha=0.2)
        if enable_crash and crash_type in ["Population Decline", "Both"]:
            ax1.axvline(pd.to_datetime(f'{crash_year}-01-01'), color='red', linestyle='--', label='Crash')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)

with col2:
    st.subheader(f"Ekonomik Tahmin (USD)")

    target_col = 'GSYIH_USD' if 'GSYIH_USD' in df_city.columns else 'GSYIH'

    # Veri kontrol edilir
    if target_col in df_city.columns and df_city[target_col].sum() > 0:

        # Toplam GDP tahmini yapılır
        with st.spinner('GSYIH Hesaplanıyor. Lütfen bekleyiniz.'):
            m_gdp, f_gdp = run_prophet(df_city, target_col, years_to_predict, enable_crash, crash_year, crash_severity,
                                       is_gdp=True)

        if f_gdp is not None:
            current_gdp = f_gdp.iloc[-years_to_predict - 1]['yhat']
            future_gdp = f_gdp.iloc[-1]['yhat']
            gdp_growth = ((future_gdp - current_gdp) / current_gdp) * 100

            # Metrikler gösterilir
            prefix = "$" if target_col == 'GSYIH_USD' else "₺"
            st.metric("Est. GDP", f"{prefix}{int(future_gdp):,}", f"{gdp_growth:.2f}% Growth")

            # Matplot ile plotting işlemi
            fig2, ax2 = plt.subplots(figsize=(10, 4))

            ax2.plot(df_city['ds'], df_city[target_col], 'go', label='History')
            ax2.plot(f_gdp['ds'], f_gdp['yhat'], 'g-', linewidth=2, label='Forecast')
            ax2.fill_between(f_gdp['ds'], f_gdp['yhat_lower'], f_gdp['yhat_upper'], color='green', alpha=0.2)

            if enable_crash and crash_type in ["Economic Crash (GDP)", "Both"]:
                ax2.axvline(pd.to_datetime(f'{crash_year}-01-01'), color='red', linestyle='--', label='Crash')

            ax2.legend()
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)

        # Normalizasyona uğramış sektörlerin gösterimi için olan kısım
        st.write("----")
        st.subheader("Sektör trendleri")

        if main_sector_cols:
            with st.spinner('Gelecek tahmini ve normalizasyon uygulanıyor'):
                df_norm = forecast_sector_trends(df_city, main_sector_cols, years_to_predict)
                future_means = df_norm[main_sector_cols].mean().sort_values(ascending=False)
                top_5_cols = future_means.head(5).index.tolist()
                df_norm['Others'] = 100 - df_norm[top_5_cols].sum(axis=1)
                plot_cols = top_5_cols + ['Others']

            #Matplot ile plotting
            fig3, ax3 = plt.subplots(figsize=(10, 5))

            x = df_norm['ds']
            y_stack = [df_norm[c] for c in plot_cols]
            labels = [c.replace('Pay_', '') for c in plot_cols]

            # Tahmin çizgisi plota çekilir
            last_hist_date = df_city['ds'].max()
            ax3.axvline(last_hist_date, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
            ax3.text(last_hist_date, 5, ' Forecast Start', color='white', fontsize=9, ha='left')
            ax3.stackplot(x, y_stack, labels=labels, alpha=0.85)

            ax3.set_title(f"Tahmini ekonomik kompozisyon")
            ax3.set_ylabel("Share (%)")
            ax3.set_ylim(0, 100)
            ax3.legend(loc='upper left', fontsize='small', framealpha=0.6, bbox_to_anchor=(1, 1))
            st.pyplot(fig3)

        else:
            st.warning("Veri bulunamadı.")
    else:
        st.warning("Bu şehir için GSYIH değeri bulunamadı.")

# --- DATA TABLE ---
with st.expander("Tahmin verisini göster"):
    if f_pop is not None: st.dataframe(f_pop.tail())