#!/usr/bin/env python3 # -*- coding: utf-8 -*- 
    """ VERIFICADOR DE DUPLICATAS DETALHADO ================================== Script para verificar diferentes tipos de duplicatas no dataset: - Duplicatas exatas - Duplicatas parciais (mesmo preço, área, localização) - Duplicatas por combinações específicas """
 
import pandas as pd 
import numpy as np 

def verificar_duplicatas_detalhado(): 
    """Verifica duplicatas de forma detalhada."""
 
        print("=" * 80) 
        print(" VERIFICAÇÃO DETALHADA DE DUPLICATAS") 
        print("=" * 80) # Carregar dados consolidados 
        try: df = pd.read_csv("data/dados_consolidados_brutos.csv") 
        print(f" Dataset carregado: {len(df):,} registros") 
        except FileNotFoundError: 
        print(" Arquivo não encontrado: data/dados_consolidados_brutos.csv") 
        return 
        print(f" Colunas disponíveis: {list(df.columns)}") # 1. DUPLICATAS COMPLETAS (todas as colunas) 
        print(f"\n" + "="*50) 
        print("1⃣ DUPLICATAS COMPLETAS (todas as colunas)") 
        print("="*50) duplicatas_completas = df.duplicated().sum() 
        print(f"Duplicatas completas encontradas: {duplicatas_completas}") 
        if duplicatas_completas > 0: 
        print("Removendo duplicatas completas...") df_sem_dup_completas = df.drop_duplicates() 
        print(f"Registros após remoção: {len(df_sem_dup_completas):,}") 
        else: df_sem_dup_completas = df.copy() # 2. DUPLICATAS POR CARACTERÍSTICAS PRINCIPAIS 
        print(f"\n" + "="*50) 
        print("2⃣ DUPLICATAS POR CARACTERÍSTICAS PRINCIPAIS") 
        print("="*50) colunas_principais = ['preco', 'area', 'quartos', 'banheiros', 'localizacao', 'tipo'] duplicatas_principais = df[colunas_principais].duplicated().sum() 
        print(f"Duplicatas por características principais: {duplicatas_principais}") 
        if duplicatas_principais > 0: # Mostrar alguns exemplos dup_mask = df[colunas_principais].duplicated(keep=False) exemplos_dup = df[dup_mask].sort_values(colunas_principais).head(10) 
        print(f"\nExemplos de duplicatas (primeiros 10):") 
        print(exemplos_dup[colunas_principais].to_string()) # 3. DUPLICATAS POR PREÇO + ÁREA + LOCALIZAÇÃO 
        print(f"\n" + "="*50) 
        print("3⃣ DUPLICATAS POR PREÇO + ÁREA + LOCALIZAÇÃO") 
        print("="*50) colunas_criticas = ['preco', 'area', 'localizacao'] duplicatas_criticas = df[colunas_criticas].duplicated().sum() 
        print(f"Duplicatas críticas (preço+área+localização): {duplicatas_criticas}") # 4. DUPLICATAS EXATAS DE PREÇO 
        print(f"\n" + "="*50) 
        print("4⃣ ANÁLISE DE PREÇOS IDÊNTICOS") 
        print("="*50) precos_duplicados = df['preco'].duplicated().sum() precos_unicos = df['preco'].nunique() total_registros = len(df) 
        print(f"Registros com preços idênticos: {precos_duplicados:,}") 
        print(f"Preços únicos: {precos_unicos:,}") 
        print(f"Taxa de duplicação de preços: {(precos_duplicados/total_registros)*100:.1f}%") # Top 10 preços mais frequentes top_precos = df['preco'].value_counts().head(10) 
        print(f"\nTop 10 preços mais frequentes:") 
        for preco, freq in top_precos.items(): 
        print(f" R$ {preco:,}: {freq} ocorrências") # 5. DUPLICATAS POR ÁREA 
        print(f"\n" + "="*50) 
        print("5⃣ ANÁLISE DE ÁREAS IDÊNTICAS") 
        print("="*50) areas_duplicadas = df['area'].duplicated().sum() areas_unicas = df['area'].nunique() 
        print(f"Registros com áreas idênticas: {areas_duplicadas:,}") 
        print(f"Áreas únicas: {areas_unicas:,}") 
        print(f"Taxa de duplicação de áreas: {(areas_duplicadas/total_registros)*100:.1f}%") # 6. ANÁLISE POR LOCALIZAÇÃO 
        print(f"\n" + "="*50) 
        print("6⃣ DISTRIBUIÇÃO POR LOCALIZAÇÃO") 
        print("="*50) dist_localizacao = df['localizacao'].value_counts() 
        print("Distribuição por localização:") 
        for local, qtd in dist_localizacao.items(): pct = (qtd/total_registros)*100 
        print(f" {local}: {qtd:,} registros ({pct:.1f}%)") # 7. RECOMENDAÇÕES DE LIMPEZA 
        print(f"\n" + "="*50) 
        print("7⃣ RECOMENDAÇÕES DE LIMPEZA") 
        print("="*50) 
        print("Estratégias recomendadas:") 
        if duplicatas_completas > 0: 
        print(f" Remover {duplicatas_completas} duplicatas completas") 
        if duplicatas_principais > duplicatas_completas: diff = duplicatas_principais - duplicatas_completas 
        print(f" Considerar remover {diff} duplicatas por características principais") 
        if precos_duplicados > 1000: 
        print(f" Investigar {precos_duplicados} registros com preços idênticos") 
        print(" - Podem ser preços 'redondos' comuns (250k, 300k, 500k, etc.)") 
        print(" - Ou dados sintéticos gerados") # 8. CRIAR DATASET LIMPO 
        print(f"\n" + "="*50) 
        print("8⃣ CRIANDO DATASET LIMPO") 
        print("="*50) # Remover duplicatas por características principais df_limpo = df.drop_duplicates(subset=colunas_principais, keep='first') registros_removidos = len(df) - len(df_limpo) 
        print(f"Registros originais: {len(df):,}") 
        print(f"Registros após limpeza: {len(df_limpo):,}") 
        print(f"Registros removidos: {registros_removidos:,}") 
        print(f"Taxa de remoção: {(registros_removidos/len(df))*100:.1f}%") # Salvar dataset limpo arquivo_limpo = "data/dados_consolidados_sem_duplicatas.csv" df_limpo.to_csv(arquivo_limpo, index=False) 
        print(f" Dataset limpo salvo em: {arquivo_limpo}") # 9. ESTATÍSTICAS FINAIS 
        print(f"\n" + "="*50) 
        print("9⃣ ESTATÍSTICAS DO DATASET LIMPO") 
        print("="*50) 
        print(f"Preços únicos: {df_limpo['preco'].nunique():,}") 
        print(f"Áreas únicas: {df_limpo['area'].nunique():,}") 
        print(f"Combinações preço+área únicas: {df_limpo[['preco', 'area']].drop_duplicates().shape[0]:,}") 
        print(f"\nDistribuição final por localização:") dist_final = df_limpo['localizacao'].value_counts() 
        for local, qtd in dist_final.items(): pct = (qtd/len(df_limpo))*100 
        print(f" {local}: {qtd:,} registros ({pct:.1f}%)") 
        print(f"\n" + "="*80) 
        print(" VERIFICAÇÃO DE DUPLICATAS CONCLUÍDA!") 
        print("="*80) 
        return df_limpo 
        if __name__ == "__main__": verificar_duplicatas_detalhado()