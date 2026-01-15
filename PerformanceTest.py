import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import warnings
import logging
import numpy as np

# Suppress warnings for cleaner terminal output
warnings.filterwarnings('ignore')
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)

INPUT_FILE = 'Prophet_Training_Set_Sektorlu.xlsx'

# CV_INITIAL: Eğitim için kaç yıllık veri kullanılacak
# CV_PERIOD: Model ne sıklıkla tahmin yapacak
# CV_HORIZON: Testimiz kaç yıl uzunluğunda olacak
# NOTE: Ensure your dataset is longer than 'initial' + 'horizon'
CV_INITIAL = '3650 days'  # 10 yıllık eğitim verisi kullanacağız
CV_PERIOD = '365 days'  # Orijinal veriler yıl bazlı olduğu için 365 gün verdim, böylelikle her period 1 yıl olacak
CV_HORIZON = '1825 days'  # Accuracy (isabetlilik) için 5 yıllık tahmin yapacağız


def performans_metrik_hesabi(input_path):
    try:
        df = pd.read_excel(input_path, sheet_name='Iller_Verisi')
    except Exception as e:
        print(f"Hata: {e}")
        return

    unique_cities = sorted(df['İl'].unique())

    # Sonuçları bir tabloda toplamak için
    summary_results = []

    for i, city in enumerate(unique_cities):
        df_city = df[df['İl'] == city].sort_values('ds').copy()
        #Popülasyon metriği için
        try:
            df_p = df_city[['ds', 'y']].copy()

            m_pop = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False) #Seasonality fonksiyonu trend yakalamak için kullanılır
                #Elimizdeki veri yıllık olduğu için sadece yearly_seasonality true yaptık
            m_pop.fit(df_p)

            # Cross Validation (Çapraz Doğrulama) uyguluyoruz
            # Burada parallel processes çok çekirdekli işlemlemcileri için işlem hızını artırır
            #Prophet'in dokümantasyonunda çok büyük olmayan veri setleri için parallel="process" kullanmamızı öneriyorlar
            df_cv_population = cross_validation(m_pop, initial=CV_INITIAL, period=CV_PERIOD, horizon=CV_HORIZON,
                                         parallel="processes")

            # Metriklerin hesaplanması
            df_p_metrics = performance_metrics(df_cv_population)

            # Horizon değeri için ortalama RMSE ve MAPE değerleri alınıyor
            pop_rmse = df_p_metrics['rmse'].mean()
            pop_mape = df_p_metrics['mape'].mean()

            print(f" Nüfus için ortalama RMSE (Root Mean Square Error): {pop_rmse:,.0f} | MAPE (Mean Absolute Percentage Error): {pop_mape:.2%}")
        except Exception as e:
            print("Hesaplama sırasında hata oluştu:", e)
            pop_rmse, pop_mape = np.nan, np.nan #Bu kısım veri sızıntısını engellemeye yarrar.
            # Bir şehirde hata olursa, değer NaN olarak atandığı için diğer şehirlerin versini etkilemez

        try:
            if 'GSYIH' in df_city.columns and df_city['GSYIH'].sum() > 0:
                df_g = df_city[['ds', 'GSYIH']].rename(columns={'GSYIH': 'y'}).copy()

                m_gdp = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False) #Aynı şekilde, model yıllık olduğu için sadece yearly değeri true olur
                m_gdp.fit(df_g)

                df_cv_gdp = cross_validation(m_gdp, initial=CV_INITIAL, period=CV_PERIOD, horizon=CV_HORIZON,
                                             parallel="processes")
                df_g_metrics = performance_metrics(df_cv_gdp)

                gdp_rmse = df_g_metrics['rmse'].mean() #Burada RMSE ve MAPE değerleri hesaplanır, ve variablelere atanır, böylelikle çıktıya yazdırabiliriz.
                gdp_mape = df_g_metrics['mape'].mean()

                print(f"GSYİH için RMSE (Root Mean Square Error): {gdp_rmse:,.0f} | MAPE (Mean Absolute Percentage Error): {gdp_mape:.2%}")
                #Atadığımız değerler burada ki print tarafından yazdırılır



        except Exception as e:
            print("Hesaplama sırasında hata oluştu:", e)

        # Sonuçları özet tablosuna ekleriz
        summary_results.append({
            'City': city,
            'Pop_RMSE': pop_rmse, #Kök ortalama kare hatası (Root Mean Square Error) ve Ortalama mutlak hata yüzdesi (Mean Absolute Percentage Error) değerleri burada atanır
            'Pop_MAPE': pop_mape,
            'GDP_RMSE': gdp_rmse,
            'GDP_MAPE': gdp_mape
        })

    #Sonuçları tablo olarak çıktısını yazdırırız
    for res in summary_results:
        p_mape = f"{res['Pop_MAPE']:.2%}" if not np.isnan(res['Pop_MAPE']) else "N/A"
        g_mape = f"{res['GDP_MAPE']:.2%}" if not np.isnan(res['GDP_MAPE']) else "N/A"
        print(f"{res['City']:<20} | {p_mape:<10} | {g_mape:<10}")



if __name__ == "__main__":
   performans_metrik_hesabi(INPUT_FILE)