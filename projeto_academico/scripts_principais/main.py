#!/usr/bin/env python3 # -*- coding: utf-8 -*- 
    """ =============================================================================== ATIVIDADE PRÁTICA: ANÁLISE DE PREÇOS DE IMÓVEIS =============================================================================== Sistema para análise preditiva de preços de imóveis no Distrito Federal usando REGRESSÃO LINEAR conforme solicitado na atividade prática. Pipeline da Atividade: 1. Web Scraping - Coleta de dados de imóveis 2. Análise de Dados - Limpeza e análise exploratória 3. Regressão Linear - Modelo preditivo usando Scikit-learn 4. Visualização de Dados - Gráficos dos resultados Desenvolvido para: Atividade Prática de Machine Learning =============================================================================== """
 
import os 
import sys 
import time 
from datetime 
import datetime # Importar módulos do sistema 
from src.web_scraper 
import WebScraperReal from src.data_analyzer 
import DataAnalyzer 
from src.linear_regression 
import LinearRegressionModel from src.visualizations 
import DataVisualizer 

def main(): 
    """ Função principal que executa todo o pipeline da atividade. """
 
        print("=" * 60) 
        print("ATIVIDADE PRÁTICA: ANÁLISE DE PREÇOS DE IMÓVEIS") 
        print("=" * 60) 
        print(f"Iniciado em: {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}") 
        print(f"Objetivo: Regressão Linear para previsão de preços") 
        print(f"Região: Distrito Federal") 
        print("=" * 60) 
        try: # ETAPA 1: WEB SCRAPING (COLETA BRUTA - SEM TRATAMENTO) 
        print("\n" + "="*20 + " ETAPA 1: WEB SCRAPING BRUTO " + "="*20) scraper = WebScraperReal() raw_data = scraper.coletar_dados(num_registros=1000) # Pelo menos 1000 registros 
        print(f"Dados brutos coletados: {len(raw_data)} registros") # PONTO DE CONTROLE - INPUT PARA CONTINUAR 
        print("\n" + "="*50) 
        print(" DADOS BRUTOS COLETADOS - PONTO DE CONTROLE") 
        print("="*50) 
        print(f" Coletados: {len(raw_data)} registros brutos") 
        print(" Dados salvos em: data/dados_brutos_raspados.csv") print() 
        print("DESEJA CONTINUAR PARA AS PRÓXIMAS ETAPAS?") 
        print("- Digite 'sim' ou 's' para continuar") 
        print("- Digite qualquer outra coisa para parar e esperar") 
        print("="*50) continuar = input(">>> Continuar? (sim/s): ").lower().strip() 
        if continuar not in ['sim', 's', 'yes', 'y']: 
        print("\n EXECUÇÃO PAUSADA!") 
        print("O programa está aguardando sua decisão...") 
        print("Execute novamente quando quiser continuar.") 
        return 
        print("\n CONTINUANDO PARA AS PRÓXIMAS ETAPAS...") time.sleep(2) # ETAPA 2: ANÁLISE DE DADOS 
        print("\n" + "="*20 + " ETAPA 2: ANÁLISE DE DADOS " + "="*20) analyzer = DataAnalyzer() clean_data = analyzer.processar_dados(raw_data) analyzer.analise_exploratoria(clean_data) 
        print(f"Dados processados: {len(clean_data)} registros limpos") # ETAPA 3: REGRESSÃO LINEAR 
        print("\n" + "="*20 + " ETAPA 3: REGRESSÃO LINEAR " + "="*20) model = LinearRegressionModel() results = model.treinar_modelo(clean_data) 
        print(f"Modelo treinado com R² = {results['r2_score']:.4f}") 
        print(f"MSE = {results['mse']:,.0f}") # ETAPA 4: VISUALIZAÇÃO DE DADOS 
        print("\n" + "="*20 + " ETAPA 4: VISUALIZAÇÃO " + "="*20) visualizer = DataVisualizer() visualizer.criar_visualizacoes(clean_data, results) 
        print("Visualizações geradas com sucesso") # RELATÓRIO FINAL 
        print("\n" + "="*25 + " RESULTADOS " + "="*25) 
        print(f"Total de imóveis analisados: {len(clean_data)}") 
        print(f"R² Score: {results['r2_score']:.4f}") 
        print(f"MSE: {results['mse']:,.0f}") 
        print(f"RMSE: R$ {results['rmse']:,.2f}") 
        print("=" * 60) 
        print("ATIVIDADE CONCLUÍDA COM SUCESSO!") 
        print("Todos os arquivos foram salvos na pasta 'data/'") 
        print("=" * 60) 
        except Exception as e: 
        print(f"ERRO durante a execução: {e}") sys.exit(1) 
        if __name__ == "__main__": main()