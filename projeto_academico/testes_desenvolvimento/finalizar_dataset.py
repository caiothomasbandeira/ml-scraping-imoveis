#!/usr/bin/env python3 # -*- coding: utf-8 -*- 
    """ FINALIZADOR DE DATASET ====================== Script para: 1. Renomear dataset final para dataset_bruto.csv 2. Limpar todos os outros datasets 3. Deixar apenas o dataset_bruto.csv """
 
import pandas as pd 
import os 
import glob 

def finalizar_dataset(): 
    """Finaliza dataset único e limpa arquivos desnecessários."""
 
        print("=" * 70) 
        print(" FINALIZANDO DATASET ÚNICO") 
        print("=" * 70) # 1. VERIFICAR DATASET FINAL arquivo_final = "data/dataset_final_real.csv" 
        if not os.path.exists(arquivo_final): 
        print(f" Arquivo não encontrado: {arquivo_final}") 
        return # Carregar dataset final df = pd.read_csv(arquivo_final) 
        print(f" Dataset final carregado: {len(df):,} registros") # 2. RENOMEAR PARA dataset_bruto.csv arquivo_bruto = "data/dataset_bruto.csv" df.to_csv(arquivo_bruto, index=False) 
        print(f" Dataset renomeado para: dataset_bruto.csv") # 3. LISTAR TODOS OS ARQUIVOS DA PASTA DATA 
        print(f"\n ARQUIVOS ENCONTRADOS NA PASTA DATA:") arquivos_data = glob.glob("data/*") 
        for arquivo in sorted(arquivos_data): nome_arquivo = os.path.basename(arquivo) 
        print(f" • {nome_arquivo}") # 4. REMOVER TODOS OS OUTROS ARQUIVOS 
        print(f"\n REMOVENDO ARQUIVOS DESNECESSÁRIOS...") arquivos_para_remover = [ "data/dataset_final_real.csv", "data/dados_consolidados_unicos.csv", "data/dados_consolidados_brutos*.csv", "data/dados_brutos_raspados*.csv", "data/dados_limpos.csv", "data/relatorio_*.txt", "data/*backup*", "data/*_v2*" ] arquivos_removidos = [] 
        for padrao in arquivos_para_remover: arquivos = glob.glob(padrao) 
        for arquivo in arquivos: 
        if arquivo != arquivo_bruto: # Não remover o dataset_bruto.csv 
        try: os.remove(arquivo) arquivos_removidos.append(os.path.basename(arquivo)) 
        print(f" Removido: {os.path.basename(arquivo)}") 
        except Exception as e: 
        print(f" Erro ao remover {arquivo}: {e}") # 5. VERIFICAR RESULTADO FINAL 
        print(f"\n ARQUIVOS RESTANTES NA PASTA DATA:") arquivos_restantes = glob.glob("data/*") 
        if arquivos_restantes: 
        for arquivo in sorted(arquivos_restantes): nome_arquivo = os.path.basename(arquivo) 
        print(f" {nome_arquivo}") 
        else: 
        print(" Nenhum arquivo encontrado!") # 6. ESTATÍSTICAS FINAIS DO DATASET 
        print(f"\n ESTATÍSTICAS DO DATASET_BRUTO.CSV:") 
        print(f" • Total de registros: {len(df):,}") 
        print(f" • Preços únicos: {df['preco'].nunique():,}") 
        print(f" • Áreas únicas: {df['area'].nunique():,}") 
        print(f" • Localizações: {df['localizacao'].nunique()}") 
        print(f" • Tipos de imóveis: {df['tipo'].nunique()}") 
        print(f" • Fontes de dados: {df['fonte'].nunique()}") 
        print(f"\n RESUMO FINANCEIRO:") 
        print(f" • Preço médio: R$ {df['preco'].mean():,.0f}") 
        print(f" • Preço mediano: R$ {df['preco'].median():,.0f}") 
        print(f" • Preço mínimo: R$ {df['preco'].min():,}") 
        print(f" • Preço máximo: R$ {df['preco'].max():,}") 
        print(f"\n PRINCIPAIS LOCALIZAÇÕES:") top_localizacoes = df['localizacao'].value_counts().head(5) 
        for local, qtd in top_localizacoes.items(): pct = (qtd/len(df))*100 
        print(f" • {local}: {qtd:,} ({pct:.1f}%)") 
        print(f"\n FONTES DE DADOS:") 
        for fonte, qtd in df['fonte'].value_counts().items(): pct = (qtd/len(df))*100 
        print(f" • {fonte}: {qtd:,} ({pct:.1f}%)") 
        print(f"\n" + "=" * 70) 
        print(" DATASET FINALIZADO COM SUCESSO!") 
        print("=" * 70) 
        print(f" Arquivo único: dataset_bruto.csv") 
        print(f" Total: {len(df):,} registros reais únicos") 
        print(f" Outros arquivos removidos: {len(arquivos_removidos)}") 
        print(" Pronto para análise exploratória!") 
        print("=" * 70) 
        return df 
        if __name__ == "__main__": finalizar_dataset()