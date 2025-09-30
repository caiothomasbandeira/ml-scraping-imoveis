import pandas as pd # Carregar dados df = pd.read_csv('data/dados_limpos.csv') 
        print(' ANÁLISE DA COLUNA FONTE:') 
        print(f'Valores únicos: {df["fonte"].nunique()}') 
        print(f'\nDistribuição:') 
        print(df["fonte"].value_counts()) 
        print(f'\n Preço médio por fonte:') fonte_stats = df.groupby("fonte")["preco"].agg(['count', 'mean', 'std']).round(0) 
        print(fonte_stats) # Correlação entre fonte e preço 
from sklearn.preprocessing 
import LabelEncoder le = LabelEncoder() df['fonte_encoded'] = le.fit_transform(df['fonte']) corr_fonte_preco = df['fonte_encoded'].corr(df['preco']) 
        print(f'\n Correlação fonte-preço: {corr_fonte_preco:.4f}') 
        if abs(corr_fonte_preco) < 0.1: 
        print(' Correlação muito baixa - fonte pode ser removida') 
        else: 
        print(' Correlação significativa - manter fonte')