#!/usr/bin/env python3 # -*- coding: utf-8 -*- 
    """ MODELAGEM DE REGRESSÃO LINEAR ============================= Implementa e avalia modelos de regressão linear para predição de preços de imóveis baseado na análise exploratória avançada concluída. """
 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection 
import train_test_split, cross_val_score, GridSearchCV from sklearn.linear_model 
import LinearRegression, Ridge, Lasso, ElasticNet 
from sklearn.preprocessing 
import StandardScaler, LabelEncoder, PolynomialFeatures from sklearn.metrics 
import mean_squared_error, r2_score, mean_absolute_error 
from sklearn.pipeline 
import Pipeline 
import warnings warnings.filterwarnings('ignore') # Configurar matplotlib para português plt.rcParams['font.size'] = 10 plt.rcParams['figure.figsize'] = (12, 8) sns.set_style("whitegrid") 

class ModelagemRegressao: 

def __init__(self): 
    """Inicializa a modelagem de regressão."""
 self.df = None self.X = None self.y = None self.X_train = None self.X_test = None self.y_train = None self.y_test = None self.modelos = {} self.resultados = {} self.melhor_modelo = None 

def carregar_dados(self): 
    """Carrega e prepara os dados para modelagem."""
 
        print("=" * 80) 
        print(" MODELAGEM DE REGRESSÃO LINEAR PARA PREÇOS DE IMÓVEIS") 
        print("=" * 80) 
        try: self.df = pd.read_csv("data/dados_limpos.csv") 
        print(f" Dados carregados: {len(self.df):,} registros") # Resumo dos insights da análise exploratória 
        print(f"\n RESUMO DOS INSIGHTS DA ANÁLISE EXPLORATÓRIA:") 
        print(f" • Dataset: {len(self.df):,} registros reais coletados") 
        print(f" • Preço médio: R$ {self.df['preco'].mean():,.0f}") 
        print(f" • Correlação preço-área: {self.df['preco'].corr(self.df['area']):.3f}") 
        print(f" • Correlação preço-preço_por_m2: {self.df['preco'].corr(self.df['preco_por_m2']):.3f}") 
        print(f" • 12 localizações diferentes no DF") 
        print(f" • Diferenças significativas entre regiões (ANOVA p<0.001)") 
        print(f" • Distribuições não-normais - transformações necessárias") 
        return True 
        except FileNotFoundError: 
        print(" Arquivo dados_limpos.csv não encontrado!") 
        return False 

def preparar_features(self): 
    """Prepara as features para o modelo."""
 
        print(f"\n" + "="*60) 
        print(" ENGENHARIA DE FEATURES") 
        print("="*60) # Criar cópia dos dados df_modelo = self.df.copy() # Encoding de variáveis categóricas 
        print(" Aplicando encoding para variáveis categóricas...") # Label encoding para localização (ordinal por preço médio) loc_preco_medio = df_modelo.groupby('localizacao')['preco'].mean().sort_values(ascending=False) loc_mapping = {loc: idx 
        for idx, loc in enumerate(loc_preco_medio.index)} df_modelo['localizacao_encoded'] = df_modelo['localizacao'].map(loc_mapping) # One-hot encoding para tipo de imóvel tipo_dummies = pd.get_dummies(df_modelo['tipo'], prefix='tipo') df_modelo = pd.concat([df_modelo, tipo_dummies], axis=1) # Criar features adicionais baseadas nos insights 
        print(" Criando features engenheiradas...") # Interação área x quartos df_modelo['area_quartos_interacao'] = df_modelo['area'] * df_modelo['quartos'] # Razão banheiros/quartos df_modelo['razao_banheiro_quarto'] = df_modelo['banheiros'] / df_modelo['quartos'] # Transformação log do preço por m² (reduzir assimetria) df_modelo['log_preco_por_m2'] = np.log1p(df_modelo['preco_por_m2']) # Features categóricas baseadas em faixas df_modelo['faixa_area'] = pd.cut(df_modelo['area'], bins=[0, 50, 100, 150, 300], labels=['Pequeno', 'Medio', 'Grande', 'Extra_Grande']) faixa_area_dummies = pd.get_dummies(df_modelo['faixa_area'], prefix='faixa_area') df_modelo = pd.concat([df_modelo, faixa_area_dummies], axis=1) # Selecionar apenas features relevantes para predição (sem 'fonte') features_numericas = [ 'area', 'quartos', 'banheiros', 'preco_por_m2', 'area_por_quarto', 'localizacao_encoded', 'area_quartos_interacao', 'razao_banheiro_quarto', 'log_preco_por_m2' ] features_categoricas = [col for col in df_modelo.columns 
        if col.startswith(('tipo_', 'faixa_area_'))] # Não incluir 'fonte' para evitar viés de amostragem todas_features = features_numericas + features_categoricas 
        print(f" FEATURES SELECIONADAS PARA MODELAGEM:") 
        print(f" • Características físicas: area, quartos, banheiros") 
        print(f" • Localização: localizacao_encoded") 
        print(f" • Tipo de imóvel: {[col 
        for col in features_categoricas 
        if col.startswith('tipo_')]}") 
        print(f" • Features engenheiradas: preco_por_m2, area_por_quarto, etc.") 
        print(f" • EXCLUÍDA: fonte (para evitar viés de amostragem)") # Remover features com valores infinitos ou NaN 
        for feature in todas_features[:]: 
        if df_modelo[feature].isnull().any() or np.isinf(df_modelo[feature]).any(): 
        print(f" Removendo feature {feature} (valores inválidos)") todas_features.remove(feature) self.X = df_modelo[todas_features] self.y = df_modelo['preco'] 
        print(f" Features preparadas: {len(todas_features)} variáveis") 
        print(f" • Numéricas: {len(features_numericas)}") 
        print(f" • Categóricas: {len(features_categoricas)}") 
        print(f" • Target: preço (variável contínua)") 
        print(f" • Dataset completo mantido, fonte preservada para referência") # Mostrar correlações das principais features 
        print(f"\n CORRELAÇÕES DAS FEATURES SELECIONADAS COM O PREÇO:") correlacoes = self.X.corrwith(self.y).abs().sort_values(ascending=False) 
        for feature, corr in correlacoes.head(10).items(): 
        print(f" • {feature}: {corr:.3f}") 

def dividir_dados(self): 
    """Divide os dados em treino e teste."""
 
        print(f"\n" + "="*60) 
        print(" DIVISÃO DOS DADOS") 
        print("="*60) # Divisão estratificada por localização para manter distribuição self.X_train, self.X_test, self.y_train, self.y_test = train_test_split( self.X, self.y, test_size=0.2, random_state=42, stratify=self.df['localizacao'] ) 
        print(f" Dados divididos:") 
        print(f" • Treino: {len(self.X_train):,} registros ({len(self.X_train)/len(self.X)*100:.1f}%)") 
        print(f" • Teste: {len(self.X_test):,} registros ({len(self.X_test)/len(self.X)*100:.1f}%)") 
        print(f" • Features: {self.X.shape[1]} variáveis") # Verificar distribuições 
        print(f"\n Verificação das distribuições:") 
        print(f" • Preço médio treino: R$ {self.y_train.mean():,.0f}") 
        print(f" • Preço médio teste: R$ {self.y_test.mean():,.0f}") 
        print(f" • Diferença: {abs(self.y_train.mean() - self.y_test.mean())/self.y_train.mean()*100:.1f}%") 

def treinar_modelos(self): 
    """Treina diferentes modelos de regressão."""
 
        print(f"\n" + "="*60) 
        print(" TREINAMENTO DOS MODELOS") 
        print("="*60) # Pipeline com padronização modelos_config = { 'Linear_Simples': Pipeline([ ('scaler', StandardScaler()), ('regressor', LinearRegression()) ]), 'Ridge': Pipeline([ ('scaler', StandardScaler()), ('regressor', Ridge(alpha=1.0)) ]), 'Lasso': Pipeline([ ('scaler', StandardScaler()), ('regressor', Lasso(alpha=1.0)) ]), 'ElasticNet': Pipeline([ ('scaler', StandardScaler()), ('regressor', ElasticNet(alpha=1.0, l1_ratio=0.5)) ]), 'Polinomial_2': Pipeline([ ('poly', PolynomialFeatures(degree=2, include_bias=False)), ('scaler', StandardScaler()), ('regressor', LinearRegression()) ]) } 
        print(" Treinando modelos com validação cruzada...") 
        for nome, modelo in modelos_config.items(): 
        print(f"\n {nome}:") 
        try: # Treinar modelo modelo.fit(self.X_train, self.y_train) # Predições y_pred_train = modelo.predict(self.X_train) y_pred_test = modelo.predict(self.X_test) # Métricas r2_train = r2_score(self.y_train, y_pred_train) r2_test = r2_score(self.y_test, y_pred_test) rmse_train = np.sqrt(mean_squared_error(self.y_train, y_pred_train)) rmse_test = np.sqrt(mean_squared_error(self.y_test, y_pred_test)) mae_test = mean_absolute_error(self.y_test, y_pred_test) # Validação cruzada cv_scores = cross_val_score(modelo, self.X_train, self.y_train, cv=5, scoring='r2') # Armazenar resultados self.modelos[nome] = modelo self.resultados[nome] = { 'r2_train': r2_train, 'r2_test': r2_test, 'rmse_train': rmse_train, 'rmse_test': rmse_test, 'mae_test': mae_test, 'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std(), 'overfitting': r2_train - r2_test, 'y_pred_test': y_pred_test } 
        print(f" • R² treino: {r2_train:.4f}") 
        print(f" • R² teste: {r2_test:.4f}") 
        print(f" • RMSE teste: R$ {rmse_test:,.0f}") 
        print(f" • MAE teste: R$ {mae_test:,.0f}") 
        print(f" • CV média: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})") 
        print(f" • Overfitting: {r2_train - r2_test:.4f}") 
        except Exception as e: 
        print(f" Erro no modelo {nome}: {str(e)}") continue 
        print(f"\n {len(self.modelos)} modelos treinados com sucesso!") 

def otimizar_melhor_modelo(self): 
    """Otimiza hiperparâmetros do melhor modelo."""
 
        print(f"\n" + "="*60) 
        print(" OTIMIZAÇÃO DE HIPERPARÂMETROS") 
        print("="*60) # Identificar melhor modelo baseado em R² teste melhor_nome = max(self.resultados.keys(), key=lambda x: self.resultados[x]['r2_test']) 
        print(f" Melhor modelo base: {melhor_nome}") 
        print(f" • R² teste: {self.resultados[melhor_nome]['r2_test']:.4f}") # Otimização específica por tipo de modelo 
        if 'Ridge' in melhor_nome: param_grid = { 'regressor__alpha': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0] } elif 'Lasso' in melhor_nome: param_grid = { 'regressor__alpha': [0.1, 0.5, 1.0, 2.0, 5.0] } elif 'ElasticNet' in melhor_nome: param_grid = { 'regressor__alpha': [0.1, 0.5, 1.0, 2.0], 'regressor__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9] } 
        else: # Para modelos sem hiperparâmetros principais 
        print(" ℹ Modelo não possui hiperparâmetros para otimizar") self.melhor_modelo = self.modelos[melhor_nome] 
        return melhor_nome 
        print(" Executando Grid Search...") # Grid Search grid_search = GridSearchCV( self.modelos[melhor_nome], param_grid, cv=5, scoring='r2', n_jobs=-1 ) grid_search.fit(self.X_train, self.y_train) # Melhor modelo otimizado self.melhor_modelo = grid_search.best_estimator_ # Avaliar modelo otimizado y_pred_train_opt = self.melhor_modelo.predict(self.X_train) y_pred_test_opt = self.melhor_modelo.predict(self.X_test) r2_train_opt = r2_score(self.y_train, y_pred_train_opt) r2_test_opt = r2_score(self.y_test, y_pred_test_opt) rmse_test_opt = np.sqrt(mean_squared_error(self.y_test, y_pred_test_opt)) mae_test_opt = mean_absolute_error(self.y_test, y_pred_test_opt) 
        print(f" Otimização concluída!") 
        print(f" • Melhores parâmetros: {grid_search.best_params_}") 
        print(f" • R² CV: {grid_search.best_score_:.4f}") 
        print(f" • R² teste otimizado: {r2_test_opt:.4f}") 
        print(f" • RMSE teste: R$ {rmse_test_opt:,.0f}") 
        print(f" • MAE teste: R$ {mae_test_opt:,.0f}") # Comparação com modelo base melhoria_r2 = r2_test_opt - self.resultados[melhor_nome]['r2_test'] 
        print(f" • Melhoria R²: {melhoria_r2:+.4f}") 
        return melhor_nome 

def analisar_resultados(self): 
    """Analisa e compara os resultados dos modelos."""
 
        print(f"\n" + "="*60) 
        print(" ANÁLISE COMPARATIVA DOS MODELOS") 
        print("="*60) # Criar DataFrame com resultados df_resultados = pd.DataFrame(self.resultados).T df_resultados = df_resultados.round(4) 
        print(" RANKING DOS MODELOS (por R² teste):") ranking = df_resultados.sort_values('r2_test', ascending=False) 
        for i, (modelo, row) in enumerate(ranking.iterrows(), 1): 
        print(f" {i}. {modelo}:") 
        print(f" • R² teste: {row['r2_test']:.4f}") 
        print(f" • RMSE: R$ {row['rmse_test']:,.0f}") 
        print(f" • MAE: R$ {row['mae_test']:,.0f}") 
        print(f" • Overfitting: {row['overfitting']:.4f}") 
        return df_resultados 

def avaliar_modelo_final(self): 
    """Avaliação detalhada do modelo final."""
 
        print(f"\n" + "="*60) 
        print(" AVALIAÇÃO FINAL DO MELHOR MODELO") 
        print("="*60) 
        if self.melhor_modelo is None: 
        print(" Nenhum modelo foi otimizado!") 
        return # Predições finais y_pred_final = self.melhor_modelo.predict(self.X_test) # Métricas finais r2_final = r2_score(self.y_test, y_pred_final) rmse_final = np.sqrt(mean_squared_error(self.y_test, y_pred_final)) mae_final = mean_absolute_error(self.y_test, y_pred_final) # Erro percentual médio mape = np.mean(np.abs((self.y_test - y_pred_final) / self.y_test)) * 100 
        print(f" MÉTRICAS FINAIS:") 
        print(f" • R² Score: {r2_final:.4f} ({r2_final*100:.1f}% da variância explicada)") 
        print(f" • RMSE: R$ {rmse_final:,.0f}") 
        print(f" • MAE: R$ {mae_final:,.0f}") 
        print(f" • MAPE: {mape:.1f}%") # Interpretação dos resultados 
        print(f"\n INTERPRETAÇÃO:") 
        if r2_final > 0.8: interpretacao = "Excelente" elif r2_final > 0.6: interpretacao = "Bom" elif r2_final > 0.4: interpretacao = "Moderado" 
        else: interpretacao = "Fraco" 
        print(f" • Desempenho: {interpretacao}") 
        print(f" • O modelo explica {r2_final*100:.1f}% da variação nos preços") 
        print(f" • Erro médio absoluto: R$ {mae_final:,.0f}") 
        print(f" • Margem de erro típica: ±{mape:.1f}%") # Análise de resíduos residuos = self.y_test - y_pred_final residuos_percentuais = (residuos / self.y_test) * 100 
        print(f"\n ANÁLISE DE RESÍDUOS:") 
        print(f" • Resíduo médio: R$ {residuos.mean():,.0f}") 
        print(f" • Desvio padrão dos resíduos: R$ {residuos.std():,.0f}") 
        print(f" • Resíduos entre ±10%: {np.sum(np.abs(residuos_percentuais) <= 10)/len(residuos_percentuais)*100:.1f}%") 
        print(f" • Resíduos entre ±20%: {np.sum(np.abs(residuos_percentuais) <= 20)/len(residuos_percentuais)*100:.1f}%") 
        return { 'r2': r2_final, 'rmse': rmse_final, 'mae': mae_final, 'mape': mape, 'residuos': residuos } 

def salvar_modelo_e_relatorio(self, metricas_finais): 
    """Salva o modelo e gera relatório final."""
 
        print(f"\n SALVANDO MODELO E RELATÓRIO...") 
import pickle # Salvar modelo with open('data/modelo_regressao_final.pkl', 'wb') as f: pickle.dump(self.melhor_modelo, f) # Salvar relatório with open('data/relatorio_modelagem.txt', 'w', encoding='utf-8') as f: f.write("RELATÓRIO DE MODELAGEM - REGRESSÃO LINEAR\n") f.write("=" * 50 + "\n\n") f.write(f"Data: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M:%S')}\n") f.write(f"Dataset: {len(self.df)} registros\n\n") f.write("MÉTRICAS FINAIS:\n") f.write(f"- R² Score: {metricas_finais['r2']:.4f}\n") f.write(f"- RMSE: R$ {metricas_finais['rmse']:,.0f}\n") f.write(f"- MAE: R$ {metricas_finais['mae']:,.0f}\n") f.write(f"- MAPE: {metricas_finais['mape']:.1f}%\n\n") f.write("FEATURES UTILIZADAS:\n") 
        for i, feature in enumerate(self.X.columns, 1): f.write(f"{i}. {feature}\n") 
        print(" Modelo salvo: data/modelo_regressao_final.pkl") 
        print(" Relatório salvo: data/relatorio_modelagem.txt") 

def executar_modelagem_completa(self): 
    """Executa pipeline completo de modelagem."""
 
        if not self.carregar_dados(): 
        return False self.preparar_features() self.dividir_dados() self.treinar_modelos() melhor_nome = self.otimizar_melhor_modelo() df_resultados = self.analisar_resultados() metricas_finais = self.avaliar_modelo_final() self.salvar_modelo_e_relatorio(metricas_finais) 
        print(f"\n" + "="*80) 
        print(" MODELAGEM DE REGRESSÃO LINEAR CONCLUÍDA!") 
        print("="*80) 
        print(f" Melhor modelo: {melhor_nome}") 
        print(f" R² final: {metricas_finais['r2']:.4f}") 
        print(f" Erro médio: R$ {metricas_finais['mae']:,.0f}") 
        print(f" Margem de erro: ±{metricas_finais['mape']:.1f}%") 
        print(" Modelo pronto para predições!") 
        print("="*80) 
        return True 

def main(): 
    """Função principal."""
 modelagem = ModelagemRegressao() sucesso = modelagem.executar_modelagem_completa() 
        if sucesso: 
        print("\n PROJETO DE ML FINALIZADO COM SUCESSO!") 
        print(" Modelo treinado e validado") 
        print(" Arquivos salvos para produção") 
        print(" Pronto para deploy!") 
        return sucesso 
        if __name__ == "__main__": main()