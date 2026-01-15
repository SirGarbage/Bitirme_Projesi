import pandas as pd


INPUT_FILE = 'Prophet_Training_Set_Sektorlu.xlsx'
OUTPUT_FILE = 'Prophet_Training_Set_Sektorlu_USD.xlsx'

#Burada yıllara göre yaklaşık bir dolar değeri verdim
USD_RATES = {
    2004: 1.42, 2005: 1.34, 2006: 1.43, 2007: 1.30,
    2008: 1.29, 2009: 1.55, 2010: 1.50, 2011: 1.67,
    2012: 1.80, 2013: 1.90, 2014: 2.19, 2015: 2.72,
    2016: 3.02, 2017: 3.65, 2018: 4.81, 2019: 5.67,
    2020: 7.01, 2021: 8.89, 2022: 16.57, 2023: 23.77,
    2024: 31.50
}

def convert_to_usd():
    try:
        df = pd.read_excel(INPUT_FILE, sheet_name='Iller_Verisi')
    except Exception as e:
        print(f"Error: {e}")
        return

    def get_usd_gdp(row):
        year = row['Yıl']
        gdp_tl = row['GSYIH']

        if year in USD_RATES and USD_RATES[year] > 0:
            return gdp_tl / USD_RATES[year]
        return None

    df['GSYIH_USD'] = df.apply(get_usd_gdp, axis=1)

    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Iller_Verisi', index=False)

if __name__ == "__main__":
    convert_to_usd()