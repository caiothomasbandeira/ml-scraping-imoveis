#!/usr/bin/env python3 # -*- coding: utf-8 -*- 
    """ ANALISADOR DE DADOS PARA ATIVIDADE PRÁTICA ========================================== Módulo responsável pela limpeza e análise exploratória dos dados. Conforme solicitado na atividade prática. Funcionalidades: - Limpeza e organização dos dados - Tratamento de valores ausentes e outliers - Análise exploratória dos dados - Estatísticas descritivas """
 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy 
import stats 

class DataAnalyzer: 
    """ Analisador de dados para o projeto de imóveis. Responsável pela limpeza, tratamento e análise exploratória dos dados coletados via web scraping. """
 

def __init__(self): 
    """Inicializa o analisador de dados."""
 self.df_original = None self.df_limpo = None 

def processar_dados(self, df_raw): 
    """ Processa e limpa os dados coletados. Args: df_raw (pd.DataFrame): Dados brutos do web scraping Returns: pd.DataFrame: Dados limpos e processados """
 
        print("Processando e limpando dados...") self.df_original = df_raw.copy() df = df_raw.copy() 
        print(f"Dados originais: {len(df)} registros") # 1. Verificar e tratar valores ausentes 
        print("Verificando valores ausentes...") missing = df.isnull().sum() 
        if missing.sum() > 0: 
        print("Valores ausentes encontrados:") 
        print(missing[missing > 0]) df = df.dropna() 
        else: 
        print("Nenhum valor ausente encontrado") # 2. Validar tipos de dados 
        print("Validando tipos de dados...") df['preco'] = pd.to_numeric(df['preco'], errors='coerce') df['area'] = pd.to_numeric(df['area'], errors='coerce') df['quartos'] = pd.to_numeric(df['quartos'], errors='coerce') df['banheiros'] = pd.to_numeric(df['banheiros'], errors='coerce') # Remover registros com dados inválidos df = df.dropna() # 3. Remover outliers usando IQR 
        print("Removendo outliers...") initial_len = len(df) 
        for col in ['preco', 'area']: Q1 = df[col].quantile(0.25) Q3 = df[col].quantile(0.75) IQR = Q3 - Q1 lower = Q1 - 1.5 * IQR upper = Q3 + 1.5 * IQR df = df[(df[col] >= lower) & (df[col] <= upper)] 
        print(f"Outliers removidos: {initial_len - len(df)} registros") # 4. Criar features derivadas 
        print("Criando features derivadas...") df['preco_por_m2'] = df['preco'] / df['area'] df['area_por_quarto'] = df['area'] / df['quartos'] self.df_limpo = df # Salvar dados limpos df.to_csv('data/dados_limpos.csv', index=False, encoding='utf-8') 
        print(f"Processamento concluído: {len(df)} registros limpos") 
        print("Dados salvos em 'data/dados_limpos.csv'") 
        return df 

def analise_exploratoria(self, df): 
    """ Realiza análise exploratória dos dados. Args: df (pd.DataFrame): Dados limpos para análise """
 
        print("\nIniciando análise exploratória...") # 1. Estatísticas descritivas 
        print("\n" + "="*50) 
        print("ESTATÍSTICAS DESCRITIVAS") 
        print("="*50) stats_desc = df[['preco', 'area', 'quartos', 'banheiros', 'preco_por_m2']].describe() 
        print(stats_desc.round(2)) # 2. Análise por localização 
        print("\n" + "="*50) 
        print("ANÁLISE POR LOCALIZAÇÃO") 
        print("="*50) loc_analysis = df.groupby('localizacao').agg({ 'preco': ['mean', 'median', 'count'], 'area': 'mean', 'preco_por_m2': 'mean' }).round(0) loc_analysis.columns = ['Preco_Medio', 'Preco_Mediano', 'Qtd_Imoveis', 'Area_Media', 'PrecoM2_Medio'] loc_analysis = loc_analysis.sort_values('Preco_Medio', ascending=False) 
        print(loc_analysis) # 3. Análise de correlações 
        print("\n" + "="*50) 
        print("MATRIZ DE CORRELAÇÃO") 
        print("="*50) corr_vars = ['preco', 'area', 'quartos', 'banheiros', 'preco_por_m2'] correlation = df[corr_vars].corr() 
        print(correlation.round(3)) # 4. Insights principais 
        print("\n" + "="*50) 
        print("PRINCIPAIS INSIGHTS") 
        print("="*50) insights = [] # Região mais cara e mais barata mais_cara = loc_analysis.index[0] mais_barata = loc_analysis.index[-1] preco_mais_cara = loc_analysis.loc[mais_cara, 'Preco_Medio'] preco_mais_barata = loc_analysis.loc[mais_barata, 'Preco_Medio'] insights.append(f"1. Região mais cara: {mais_cara} (R$ {preco_mais_cara:,.0f})") insights.append(f"2. Região mais barata: {mais_barata} (R$ {preco_mais_barata:,.0f})") # Correlação preço-área corr_preco_area = correlation.loc['preco', 'area'] insights.append(f"3. Correlação preço-área: {corr_preco_area:.3f}") # Configuração mais comum quartos_comum = df['quartos'].mode().iloc[0] quartos_pct = (df['quartos'] == quartos_comum).mean() * 100 insights.append(f"4. Configuração mais comum: {quartos_comum} quartos ({quartos_pct:.1f}%)") # Preço médio geral preco_medio = df['preco'].mean() insights.append(f"5. Preço médio geral: R$ {preco_medio:,.0f}") 
        for insight in insights: 
        print(insight) # Salvar relatório de análise self._salvar_relatorio_analise(df, loc_analysis, correlation, insights) 
        print("\nAnálise exploratória concluída!") 
        print("Relatório salvo em 'data/relatorio_analise.txt'") 

def _salvar_relatorio_analise(self, df, loc_analysis, correlation, insights): 
    """Salva relatório detalhado da análise."""
 with open('data/relatorio_analise.txt', 'w', encoding='utf-8') as f: f.write("RELATÓRIO DE ANÁLISE EXPLORATÓRIA\n") f.write("=" * 50 + "\n\n") f.write(f"Data da análise: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M:%S')}\n") f.write(f"Total de registros: {len(df)}\n\n") f.write("ESTATÍSTICAS DESCRITIVAS\n") f.write("-" * 30 + "\n") f.write(df[['preco', 'area', 'quartos', 'banheiros']].describe().to_string()) f.write("\n\n") f.write("ANÁLISE POR LOCALIZAÇÃO\n") f.write("-" * 30 + "\n") f.write(loc_analysis.to_string()) f.write("\n\n") f.write("PRINCIPAIS INSIGHTS\n") f.write("-" * 30 + "\n") 
        for insight in insights: f.write(insight + "\n") # Função para teste #
        if __name__ == "__main__": # Teste com dados simulados #data = { #'preco': [400000, 350000, 600000, 250000], #'area': [80, 70, 120, 50], #'quartos': [3, 2, 4, 2], #'banheiros': [2, 1, 3, 1], #'localizacao': ['Asa Norte', 'Ceilândia', 'Asa Sul', 'Taguatinga'], #'tipo': ['Apartamento', 'Apartamento', 'Casa', 'Apartamento'] # } # df_test = pd.DataFrame(data) # analyzer = DataAnalyzer() # clean_data = analyzer.processar_dados(df_test) # analyzer.analise_exploratoria(clean_data) # 
        print("Teste do analisador concluído!")