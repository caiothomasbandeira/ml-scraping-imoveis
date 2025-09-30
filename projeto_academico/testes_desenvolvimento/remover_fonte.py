#!/usr/bin/env python3 # -*- coding: utf-8 -*- 
    """ REMOÇÃO DA COLUNA FONTE ======================= Remove a coluna fonte dos dados limpos, pois ela representa viés de amostragem e não características reais do imóvel que impactam o preço. """
 
import pandas as pd 

def remover_fonte(): 
    """Remove a coluna fonte dos dados limpos."""
 
        print(" REMOVENDO COLUNA FONTE") 
        print("=" * 40) # Carregar dados df = pd.read_csv("data/dados_limpos.csv") 
        print(f" ANTES da remoção:") 
        print(f" • Shape: {df.shape}") 
        print(f" • Colunas: {len(df.columns)}") 
        print(f" • Memória: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB") # Verificar se fonte existe 
        if 'fonte' not in df.columns: 
        print(" Coluna 'fonte' não encontrada!") 
        return False # Mostrar distribuição da fonte antes de remover 
        print(f"\n DISTRIBUIÇÃO DA FONTE (removendo viés):") fonte_dist = df['fonte'].value_counts() 
        for fonte, count in fonte_dist.items(): pct = (count / len(df)) * 100 
        print(f" • {fonte}: {count:,} ({pct:.1f}%)") # Remover coluna fonte df_limpo = df.drop('fonte', axis=1) 
        print(f"\n APÓS remoção:") 
        print(f" • Shape: {df_limpo.shape}") 
        print(f" • Colunas: {len(df_limpo.columns)}") 
        print(f" • Memória: {df_limpo.memory_usage(deep=True).sum() / 1024**2:.2f} MB") 
        print(f" • Economia de memória: {((df.memory_usage(deep=True).sum() - df_limpo.memory_usage(deep=True).sum()) / df.memory_usage(deep=True).sum()) * 100:.1f}%") # Mostrar colunas finais focadas em características do imóvel 
        print(f"\n COLUNAS FINAIS (apenas características relevantes):") 
        for i, col in enumerate(df_limpo.columns, 1): 
        if col == 'preco': tipo = "(TARGET)" elif col in ['banheiros', 'area', 'quartos']: tipo = "(física)" elif col in ['tipo', 'localizacao']: tipo = "(categórica)" elif col in ['preco_por_m2', 'area_por_quarto']: tipo = "(engenheirada)" 
        else: tipo = "" 
        print(f" {i}. {col} {tipo}") # Salvar dados atualizados df_limpo.to_csv("data/dados_limpos.csv", index=False) 
        print(f"\n Dados salvos sem viés de fonte!") 
        print(f" Arquivo atualizado: data/dados_limpos.csv") 
        print(f" Dataset focado apenas em características do imóvel") 
        return True 
        if __name__ == "__main__": sucesso = remover_fonte() 
        if sucesso: 
        print("\n Excelente! Dataset limpo e sem viéses para modelagem robusta.") 
        else: 
        print("\n Erro na remoção da coluna fonte.")