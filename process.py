import pandas as pd
import numpy as np

def process_data(data):
    pass

if __name__ == "__main__":
    # 雨量 & 流量資料
    df = pd.read_excel("./data/石門水庫雨量與流量資料.xlsx")
    df.rename(columns={'日期': 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df.set_index('date')

    drop_cols = [f'STATUS.{i}' if i > 0 else 'STATUS' for i in range(8)]
    df = df.drop(drop_cols, axis=1)
    print(df.head())

    # 溫度資料
    with open('./data/石門站溫度資料.txt', encoding='utf-8') as f:
        lines = f.readlines()

    lines = [each.strip().split('\t') for each in lines]
    temperature_df = pd.DataFrame(lines[1:], columns = ['date', 'temperature'])
    temperature_df['date'] = pd.to_datetime(temperature_df['date'], format='%Y%m%d')
    temperature_df.set_index('date')
    print(temperature_df)

    print('-' * 100)
    result = pd.merge(df, temperature_df, on='date')
    print(result)

    result.to_csv('./data/石門水庫統整資料.csv', index=False, encoding='utf-8')
