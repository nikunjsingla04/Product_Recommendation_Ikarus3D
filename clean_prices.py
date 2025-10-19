import pandas as pd, re, numpy as np
p = "embeddings_data/metadata.csv"
df = pd.read_csv(p)
def parse_price(s):
    if pd.isna(s): return np.nan
    s = str(s)
    m = re.search(r'[-+]?\d[\d,]*\.?\d*', s)
    if not m: return np.nan
    try:
        return float(m.group(0).replace(',', ''))
    except:
        return np.nan
df['price_num'] = df['price'].apply(parse_price)
df.to_csv(p, index=False)
print("Wrote embeddings_data/metadata.csv with price_num.")
