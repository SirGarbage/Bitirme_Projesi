import pandas as pd
import os
import re


def prepare_data_for_prophet(input_file, gdp_file, output_file):



    # Adım 1: Uygulama 81 il için olan nüfüus verilerini okumaya başlar.

    if not os.path.exists(input_file):
        print(f"HATA: '{input_file}' bulunamadı.")
        return

    try:
        df_pop = pd.read_csv(input_file, encoding='utf-8-sig')
        # Sütun isimlerindeki olası boşlukları temizle
        df_pop.columns = df_pop.columns.str.strip()
    except Exception as e:
        print(f"HATA: Nüfus dosyası okunamadı: {e}")
        return


    # 2.adım uygulama Ekonomik verileri tutan csv dosyasını okumaya başlar

    df_gdp_clean = pd.DataFrame()

    if os.path.exists(gdp_file):
        print("Ekonomik veriler okunuyor")
        try:
            df_raw = pd.read_csv(gdp_file, sep='|', header=None, engine='python')


            years_row = df_raw.iloc[3] #Bu kısım yıl değerlerine bakar. Bu da hardcoding yapmadan farklı veri dosyları yüklesekde sıkıntı çıkmamasına olanak sağlar
            year_indices = {}

            for col_idx, val in enumerate(years_row):
                if pd.notna(val) and str(val).strip().isdigit():
                    year_indices[int(str(val).strip())] = col_idx

            print(f"   Bulunan Yıllar: {min(year_indices.keys())} - {max(year_indices.keys())}")

            # Şehirleri tarayan döngü burada. N = Şehir Adı + GSYIH Değerleri ve N+1 = Sektör Payları
            # Bu şekilde 2şerli okur
            gdp_data = []
            max_rows = len(df_raw)

            # 5'ten başlayıp dosya sonuna kadar 2'şer atlayarak gidiyoruz, tüm iller için

                # Dosya sonu kontrolü
                if row_idx + 1 >= max_rows:
                    break

                # Şehir adını al
                raw_city = str(df_raw.iloc[row_idx, 0])
                if raw_city == 'nan' or not raw_city.strip():
                    continue

                # Her yıl için veri çekilir
                for year, start_col in year_indices.items():
                    try:
                        # GSYIH Değeri (Şehir satırında, yıl başlangıç sütununda)
                        gdp_val = df_raw.iloc[row_idx, start_col]

                        # Sektör Payları (Bir alt satırda, sırayla dizili)
                        sector_shares = df_raw.iloc[row_idx + 1, start_col + 1: start_col + 12].values

                        # Veri Sözlüğü
                        entry = {
                            'İl_Ham': raw_city,
                            'Yıl': year,
                            'GSYIH': gdp_val,
                            # Sektörleri sırasıyla eşleştiriyoruz
                            'Pay_Tarim': sector_shares[0],
                            'Pay_Sanayi': sector_shares[1],
                            'Pay_Imalat': sector_shares[2],
                            'Pay_Insaat': sector_shares[3],
                            'Pay_Hizmet': sector_shares[4],
                            'Pay_Bilgi': sector_shares[5],
                            'Pay_Finans': sector_shares[6],
                            'Pay_Gayrimenkul': sector_shares[7],
                            'Pay_Mesleki': sector_shares[8],
                            'Pay_Kamu': sector_shares[9],
                            'Pay_Diger': sector_shares[10]
                        }
                        gdp_data.append(entry)

                    except Exception as e:
                        # Yılın sütunu boşsa veya hata varsa atla
                        continue

            df_gdp_clean = pd.DataFrame(gdp_data)

            # Sayısal Dönüşüm (Hata varsa 0 yap)
            cols_to_numeric = ['GSYIH', 'Pay_Tarim', 'Pay_Sanayi', 'Pay_Imalat', 'Pay_Insaat',
                               'Pay_Hizmet', 'Pay_Bilgi', 'Pay_Finans', 'Pay_Gayrimenkul',
                               'Pay_Mesleki', 'Pay_Kamu', 'Pay_Diger']

            for col in cols_to_numeric:
                df_gdp_clean[col] = pd.to_numeric(df_gdp_clean[col], errors='coerce').fillna(0)

            print(f"Toplam {len(df_gdp_clean)} veri noktası oluşturuldu.")

        except Exception as e:
            print(f"GDP dosyası işlenirken sorun oluştu: {e}")
            return
    else:
        print(f"'{gdp_file}' dosyası bulunamadı. Sadece nüfus verisi kullanılacak.")

    # Şehir isimlerini temizleme fonksiyonum. Örneğim Adana-1'i adana yapar
    def clean_city_name(text):
        if pd.isna(text): return ""
        text = str(text).strip()
        # Sondaki tire ve sayıyı sil (-1, -02 vb.)
        text = re.sub(r'-\d+$', '', text)
        # Türkçe karakterleri düzelt ve büyüt
        text = text.replace('i', 'İ').upper()
        return text

    # Her iki tabloya da temizlenmiş isim anahtarı ekle
    df_pop['İl_Key'] = df_pop['İl'].apply(clean_city_name)

    if not df_gdp_clean.empty:
        df_gdp_clean['İl_Key'] = df_gdp_clean['İl_Ham'].apply(clean_city_name)

        # Nüfus tablosundaki her satıra uygun GDP verisini ekle
        df_merged = pd.merge(df_pop,
                             df_gdp_clean.drop(columns=['İl_Ham']),
                             on=['İl_Key', 'Yıl'],
                             how='left')
        fill_cols = ['GSYIH'] + [c for c in df_gdp_clean.columns if c.startswith('Pay_')]
        df_merged[fill_cols] = df_merged.groupby('İl_Key')[fill_cols].ffill().fillna(0)

        df_final = df_merged
    else:
        df_final = df_pop
        # Sütunlar eksik kalmasın diye boş ekle
        df_final['GSYIH'] = 0

    # Prophet için gerekli tarih sütunu
    df_final['ds'] = pd.to_datetime(df_final['Yıl'].astype(str) + '-12-31')
    df_final['y'] = df_final['Toplam_Nüfus']

    # Ülke geneli özet
    agg_dict = {'y': 'sum', 'GSYIH': 'sum'}
    # Sektör payları için ortalama alınır
    if not df_gdp_clean.empty:
        for c in df_gdp_clean.columns:
            if c.startswith('Pay_'): agg_dict[c] = 'mean'

    df_total = df_final.groupby('ds').agg(agg_dict).reset_index()

    # İller Verisi bazlı veri
    base_cols = ['ds', 'y', 'İl', 'Yıl', 'Kategori', 'Erkek', 'Kadın']
    econ_cols = ['GSYIH'] + [c for c in df_final.columns if c.startswith('Pay_')]

    final_cols = [c for c in base_cols + econ_cols if c in df_final.columns]
    df_provinces = df_final[final_cols]

    # Kaydetme kısmı. Burada ki işlem çok karmaşık değil, hazır fonksiyonlar bize yardımcı oluyor
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_total.to_excel(writer, sheet_name='Turkiye_Toplam', index=False)
        df_provinces.to_excel(writer, sheet_name='Iller_Verisi', index=False)




# Son kontrol olaraktan dosya isimlerinin tam eşleştiğinden emin olmak için
input_csv = 'TUIK_Nufus_Verileri_20251121_130158.csv'
gdp_file = 'GayriSafiSektor.csv'
output_excel = 'Prophet_Training_Set_Sektorlu.xlsx'

if __name__ == "__main__":
    prepare_data_for_prophet(input_csv, gdp_file, output_excel)