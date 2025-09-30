#!/usr/bin/env python3 # -*- coding: utf-8 -*- 
    """ EXECUÇÃO DA ANÁLISE DE DADOS ISOLADA ==================================== Script para executar apenas a análise exploratória dos dados consolidados, permitindo ver insights detalhados antes de prosseguir com o modelo. """
 
import pandas as pd 
import sys 
import os # Adicionar pasta src ao path sys.path.append(os.path.join(os.path.dirname(__file__), 'src')) 
from data_analyzer 
import DataAnalyzer 

def main(): 
    """Executa apenas a análise dos dados consolidados."""
 
        print("=" * 70) 
        print("ANÁLISE EXPLORATÓRIA DE DADOS - MODO ISOLADO") 
        print("=" * 70) # Carregar dataset bruto final arquivo_dados = "data/dataset_bruto.csv" 
        if not os.path.exists(arquivo_dados): 
        print(f" Arquivo não encontrado: {arquivo_dados}") 
        print("Execute primeiro a preparação do dataset!") 
        return 
        print(f" Carregando dados de: {arquivo_dados}") 
        try: df_bruto = pd.read_csv(arquivo_dados) 
        print(f" Dataset carregado: {len(df_bruto)} registros") # Mostrar informações básicas do dataset bruto 
        print(f"\n INFORMAÇÕES DO DATASET BRUTO:") 
        print(f" • Total de registros: {len(df_bruto):,}") 
        print(f" • Colunas: {list(df_bruto.columns)}") 
        if 'preco' in df_bruto.columns: 
        print(f" • Faixa de preços: R$ {df_bruto['preco'].min():,} - R$ {df_bruto['preco'].max():,}") 
        if 'fonte' in df_bruto.columns: 
        print(f" • Fontes de dados:") 
        for fonte, qtd in df_bruto['fonte'].value_counts().items(): 
        print(f" - {fonte}: {qtd:,} registros") # Executar análise 
        print(f"\n INICIANDO ANÁLISE EXPLORATÓRIA...") analyzer = DataAnalyzer() # Processar e limpar dados df_limpo = analyzer.processar_dados(df_bruto) # Executar análise exploratória analyzer.analise_exploratoria(df_limpo) 
        print(f"\n" + "=" * 70) 
        print(" ANÁLISE CONCLUÍDA COM SUCESSO!") 
        print("=" * 70) 
        print(f" Registros analisados: {len(df_limpo):,}") 
        print(f" Dados limpos salvos em: data/dados_limpos.csv") 
        print(f" Relatório salvo em: data/relatorio_analise.txt") 
        print("=" * 70) # Perguntar se quer continuar para o modelo 
        print(f"\n PRÓXIMOS PASSOS:") 
        print("Deseja continuar para a modelagem de regressão linear?") continuar = input("Digite 'sim' ou 's' para continuar: ").lower().strip() 
        if continuar in ['sim', 's', 'yes', 'y']: 
        print(" Prosseguindo para modelagem...") # Aqui poderia chamar o próximo módulo 
        return df_limpo 
        else: 
        print(" Análise finalizada. Execute o pipeline completo quando quiser continuar.") 
        return None 
        except Exception as e: 
        print(f" Erro durante a análise: {e}") 
        return None 
        if __name__ == "__main__": main()