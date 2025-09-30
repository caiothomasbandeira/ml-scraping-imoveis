#!/usr/bin/env python3 # -*- coding: utf-8 -*- 
    """ REMOÇÃO DA COLUNA TIMESTAMP =========================== Remove a coluna timestamp dos dados limpos, pois ela não contribui para a predição de preços de imóveis. """
 
import pandas as pd 

def remover_timestamp(): 
    """Remove a coluna timestamp dos dados limpos."""
 
        print(" REMOVENDO COLUNA TIMESTAMP") 
        print("=" * 40) # Carregar dados df = pd.read_csv("data/dados_limpos.csv") 
        print(f" ANTES da remoção:") 
        print(f" • Shape: {df.shape}") 
        print(f" • Colunas: {len(df.columns)}") 
        print(f" • Memória: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB") 
        print(f" • Primeiras colunas: {list(df.columns[:5])}") # Verificar se timestamp existe 
        if 'timestamp' not in df.columns: 
        print(" Coluna 'timestamp' não encontrada!") 
        return False # Remover coluna timestamp df_limpo = df.drop('timestamp', axis=1) 
        print(f"\n APÓS remoção:") 
        print(f" • Shape: {df_limpo.shape}") 
        print(f" • Colunas: {len(df_limpo.columns)}") 
        print(f" • Memória: {df_limpo.memory_usage(deep=True).sum() / 1024**2:.2f} MB") 
        print(f" • Economia de memória: {((df.memory_usage(deep=True).sum() - df_limpo.memory_usage(deep=True).sum()) / df.memory_usage(deep=True).sum()) * 100:.1f}%") # Mostrar novas colunas 
        print(f"\n COLUNAS FINAIS:") 
        for i, col in enumerate(df_limpo.columns, 1): 
        print(f" {i}. {col}") # Salvar dados atualizados df_limpo.to_csv("data/dados_limpos.csv", index=False) 
        print(f"\n Dados salvos sem a coluna timestamp!") 
        print(f" Arquivo atualizado: data/dados_limpos.csv") 
        return True 
        if __name__ == "__main__": sucesso = remover_timestamp() 
        if sucesso: 
        print("\n Pronto! Os dados estão otimizados para modelagem.") 
        else: 
        print("\n Erro na remoção da coluna timestamp.")