import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from prophet import Prophet
import warnings
import logging

# Hatalar supresslenir, temiz bir log için ayarlanır
warnings.filterwarnings('ignore')
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)


INPUT_FILE = 'Prophet_Training_Set_Sektorlu.xlsx'
OUTPUT_PDF_POP = 'Rapor_1_Nufus_Tahminleri.pdf'
OUTPUT_PDF_GDP = 'Rapor_2_GSYIH_ve_Sektor_Analizi.pdf'

#Ayarlar
PREDICTION_YEARS = 5


def generate_dual_reports(input_path, out_pop, out_gdp):
    # Veri yüklenir
    try:
        # Oluşturulan veri dosyası okunur
        df = pd.read_excel(input_path, sheet_name='Iller_Verisi')
        print(f"{df.shape}")

        # Sektörler incelenir
        sector_cols = [col for col in df.columns if col.startswith('Pay_')]
        if sector_cols:
            print(f"{len(sector_cols)} adet ekonomik bulundu: {sector_cols}")


    except Exception as e:
        print(f"ERROR: Dosya: '{input_path}' okunamıyor. Çünkü: {e}")
        return

    unique_cities = sorted(df['İl'].unique())
    print(f"{len(unique_cities)} adet il bulundu. Analiz yapılıyor, lütfen bekleyiniz.")

    with PdfPages(out_pop) as pdf_pop, PdfPages(out_gdp) as pdf_gdp:

        for i, city in enumerate(unique_cities):
            print(f"[{i + 1}/{len(unique_cities)}] işlenmekte olan: {city}")

            # Filter data for the specific city
            df_city = df[df['İl'] == city].sort_values('ds').copy()

            if len(df_city) < 2:
                print(f"Yeterli veri olmadığından {city} atlanıyor.")
                continue


           #Nüfus tahmin raporu burada oluşturulur

            try:
                # Veri hazırlanır
                df_p = df_city[['ds', 'y']].copy()

                m_pop = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
                m_pop.fit(df_p)

                future_pop = m_pop.make_future_dataframe(periods=PREDICTION_YEARS, freq='YE')
                forecast_pop = m_pop.predict(future_pop)

                # Bulunan popülasyon görselleştirilir
                fig1 = plt.figure(figsize=(10, 7))

                plt.plot(df_p['ds'], df_p['y'], 'ko', label='Actual Data')
                plt.plot(forecast_pop['ds'], forecast_pop['yhat'], 'b-', linewidth=2, label='Forecast')
                plt.fill_between(forecast_pop['ds'], forecast_pop['yhat_lower'], forecast_pop['yhat_upper'],
                                 color='blue', alpha=0.2)

                plt.title(f"{city} Popülasyon tahmini (Sonraki {PREDICTION_YEARS} yıl için)", fontsize=14,
                          fontweight='bold')
                plt.xlabel("Yıl")
                plt.ylabel("Nüfus")
                plt.grid(True, alpha=0.3)
                plt.legend()

                last_val = forecast_pop['yhat'].iloc[-1]
                plt.figtext(0.5, 0.01, f"Tahmin edilen popülasyon: {future_pop['ds'].iloc[-1].year}: {int(last_val):,}",
                            ha="center", fontsize=10, bbox={"facecolor": "orange", "alpha": 0.2, "pad": 5})

                pdf_pop.savefig(fig1)
                plt.close(fig1)
            except Exception as e:
                print(f"{city} için popülasyon modeli hatası: {e}")

            #Sektör dağılımı ve GPD analizi bu kısımda yapılır.

            try:
                if 'GSYIH' in df_city.columns and df_city['GSYIH'].sum() > 0:
                    df_g = df_city[['ds', 'GSYIH']].rename(columns={'GSYIH': 'y'}).copy()

                    m_gdp = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
                    m_gdp.fit(df_g)

                    future_gdp = m_gdp.make_future_dataframe(periods=PREDICTION_YEARS, freq='YE')
                    forecast_gdp = m_gdp.predict(future_gdp)

                    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [1, 1]})

                    # GDP tahmini için plot oluşturulur
                    ax1.plot(df_g['ds'], df_g['y'], 'go', label='Actual GDP')
                    ax1.plot(forecast_gdp['ds'], forecast_gdp['yhat'], 'g-', linewidth=2, label='GDP Forecast')
                    ax1.fill_between(forecast_gdp['ds'], forecast_gdp['yhat_lower'], forecast_gdp['yhat_upper'],
                                     color='green', alpha=0.2)
                    ax1.set_title(f"{city} - Economic Outlook (GSYIH)", fontsize=12, fontweight='bold')
                    ax1.set_ylabel("GDP Value (TL)")
                    ax1.grid(True, alpha=0.3)
                    ax1.legend()

                    # Sektör dağılımı için görselleştirme burada yapılır
                    if sector_cols:
                        df_sectors = df_city[['ds'] + sector_cols].dropna()

                        if not df_sectors.empty:
                            means = df_sectors[sector_cols].mean().sort_values(ascending=False)
                            top_5 = means.head(5).index.tolist()

                            x = df_sectors['ds']
                            y_stack = [df_sectors[col] for col in top_5]
                            labels = [col.replace('Pay_', '') for col in top_5]

                            ax2.stackplot(x, y_stack, labels=labels, alpha=0.7)
                            ax2.set_title(f"({city}) için en büyük 5 sektör", fontsize=12)
                            ax2.set_ylabel("Sektör Payı (%)")
                            ax2.set_xlabel("Yıl")
                            ax2.legend(loc='upper left', fontsize='small', framealpha=0.5)
                            ax2.grid(True, alpha=0.3)
                        else:
                            ax2.text(0.5, 0.5, "Yetersiz veri", ha='center')
                    else:
                        ax2.text(0.5, 0.5, "Sektör verisi yok", ha='center')

                    plt.tight_layout()
                    pdf_gdp.savefig(fig2)
                    plt.close(fig2)
                else:
                    print(f"{city} için veri yok")

            except Exception as e:
                print(f"{city} için hata: {e}")

    print("\nİşlem tamamlandı")
    print(f"Popülasyon raporu {out_pop} olarak kaydedildi")
    print(f"Ekonomi raporu {out_gdp} olarak kaydedildi")


if __name__ == "__main__":
    generate_dual_reports(INPUT_FILE, OUTPUT_PDF_POP, OUTPUT_PDF_GDP)