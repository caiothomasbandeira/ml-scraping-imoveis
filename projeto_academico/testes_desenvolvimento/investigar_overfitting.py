#!/usr/bin/env python3 # -*- coding: utf-8 -*- 
    """ INVESTIGAÇÃO DE OVERFITTING E TESTE DE MODELOS SIMPLES ====================================================== Investiga porque o modelo está overfittando e testa com: 1. Menos dados para treino 2. Modelos mais simples 3. Menos features 4. Regularização mais forte """
 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection 
import train_test_split, cross_val_score, validation_curve from sklearn.linear_model 
import LinearRegression, Ridge, Lasso 
from sklearn.preprocessing 
import StandardScaler, LabelEncoder from sklearn.metrics 
import mean_squared_error, r2_score, mean_absolute_error 
from sklearn.pipeline 
import Pipeline 
import warnings warnings.filterwarnings('ignore') 

class InvestigacaoOverfitting: 

def __init__(self): self.df = None self.resultados_investigacao = {} 

def carregar_dados(self): 
    """Carrega e examina os dados."""
 
        print(" INVESTIGAÇÃO DE OVERFITTING") 
        print("=" * 50) self.df = pd.read_csv("data/dados_limpos.csv") 
        print(f" Dados carregados: {len(self.df):,} registros") # Estatísticas básicas 
        print(f"\n ESTATÍSTICAS BÁSICAS:") 
        print(f" • Preço médio: R$ {self.df['preco'].mean():,.0f}") 
        print(f" • Preço mediano: R$ {self.df['preco'].median():,.0f}") 
        print(f" • Desvio padrão: R$ {self.df['preco'].std():,.0f}") 
        print(f" • Coef. Variação: {(self.df['preco'].std()/self.df['preco'].mean())*100:.1f}%") 
        return True 

def preparar_features_simples(self): 
    """Prepara features mais simples para evitar overfitting."""
 
        print(f"\n PREPARANDO FEATURES SIMPLES") 
        print("=" * 40) # Encoding simples para localização le = LabelEncoder() self.df['localizacao_encoded'] = le.fit_transform(self.df['localizacao']) # One-hot para tipo (simples) tipo_dummies = pd.get_dummies(self.df['tipo'], prefix='tipo') df_features = pd.concat([self.df, tipo_dummies], axis=1) # CONJUNTOS DE FEATURES DE COMPLEXIDADE CRESCENTE # 1. MUITO SIMPLES (3 features) features_muito_simples = ['area', 'quartos', 'localizacao_encoded'] # 2. SIMPLES (5 features) features_simples = ['area', 'quartos', 'banheiros', 'localizacao_encoded', 'preco_por_m2'] # 3. MODERADO (8 features) features_moderadas = features_simples + ['area_por_quarto', 'tipo_Casa', 'tipo_Apartamento'] # 4. COMPLEXO (todas as features do modelo anterior) features_complexas = features_moderadas + ['tipo_Cobertura'] conjuntos_features = { 'Muito_Simples': features_muito_simples, 'Simples': features_simples, 'Moderado': features_moderadas, 'Complexo': features_complexas } 
        print(" CONJUNTOS DE FEATURES PREPARADOS:") 
        for nome, features in conjuntos_features.items(): 
        print(f" • {nome}: {len(features)} features") 
        print(f" {features}") 
        return df_features, conjuntos_features 

def testar_tamanhos_amostra(self, df_features, features): 
    """Testa diferentes tamanhos de amostra para treino."""
 
        print(f"\n TESTANDO DIFERENTES TAMANHOS DE AMOSTRA") 
        print("=" * 50) tamanhos = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9] # Porcentagem dos dados para treino X = df_features[features] y = df_features['preco'] resultados_tamanho = [] 
        for tamanho in tamanhos: 
        print(f"\n Testando com {tamanho*100:.0f}% dos dados para treino:") # Divisão dos dados X_train, X_test, y_train, y_test = train_test_split( X, y, train_size=tamanho, random_state=42, stratify=df_features['localizacao'] ) # Modelo simples pipeline = Pipeline([ ('scaler', StandardScaler()), ('regressor', LinearRegression()) ]) # Treinar pipeline.fit(X_train, y_train) # Avaliar y_pred_train = pipeline.predict(X_train) y_pred_test = pipeline.predict(X_test) r2_train = r2_score(y_train, y_pred_train) r2_test = r2_score(y_test, y_pred_test) rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test)) mae_test = mean_absolute_error(y_test, y_pred_test) overfitting = r2_train - r2_test resultado = { 'tamanho_treino': tamanho, 'n_treino': len(X_train), 'n_teste': len(X_test), 'r2_train': r2_train, 'r2_test': r2_test, 'rmse_test': rmse_test, 'mae_test': mae_test, 'overfitting': overfitting } resultados_tamanho.append(resultado) 
        print(f" • Treino: {len(X_train):,} | Teste: {len(X_test):,}") 
        print(f" • R² treino: {r2_train:.4f}") 
        print(f" • R² teste: {r2_test:.4f}") 
        print(f" • Overfitting: {overfitting:.4f}") 
        print(f" • RMSE: R$ {rmse_test:,.0f}") 
        return pd.DataFrame(resultados_tamanho) 

def testar_complexidade_features(self, df_features, conjuntos_features): 
    """Testa diferentes níveis de complexidade de features."""
 
        print(f"\n TESTANDO COMPLEXIDADE DE FEATURES") 
        print("=" * 50) resultados_complexidade = [] 
        for nome, features in conjuntos_features.items(): 
        print(f"\n Testando conjunto: {nome} ({len(features)} features)") 
        try: X = df_features[features] y = df_features['preco'] # Divisão balanceada X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42, stratify=df_features['localizacao'] ) # Modelo linear simples pipeline = Pipeline([ ('scaler', StandardScaler()), ('regressor', LinearRegression()) ]) # Treinar pipeline.fit(X_train, y_train) # Avaliar y_pred_train = pipeline.predict(X_train) y_pred_test = pipeline.predict(X_test) r2_train = r2_score(y_train, y_pred_train) r2_test = r2_score(y_test, y_pred_test) rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test)) mae_test = mean_absolute_error(y_test, y_pred_test) # Validação cruzada cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2') resultado = { 'conjunto': nome, 'n_features': len(features), 'r2_train': r2_train, 'r2_test': r2_test, 'r2_cv_mean': cv_scores.mean(), 'r2_cv_std': cv_scores.std(), 'rmse_test': rmse_test, 'mae_test': mae_test, 'overfitting': r2_train - r2_test, 'features': ', '.join(features[:3]) + ('...' 
        if len(features) > 3 else '') } resultados_complexidade.append(resultado) 
        print(f" • R² treino: {r2_train:.4f}") 
        print(f" • R² teste: {r2_test:.4f}") 
        print(f" • R² CV: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})") 
        print(f" • Overfitting: {r2_train - r2_test:.4f}") 
        print(f" • RMSE: R$ {rmse_test:,.0f}") 
        print(f" • MAE: R$ {mae_test:,.0f}") 
        except Exception as e: 
        print(f" Erro: {str(e)}") continue 
        return pd.DataFrame(resultados_complexidade) 

def testar_regularizacao(self, df_features, features_otimas): 
    """Testa diferentes níveis de regularização."""
 
        print(f"\n TESTANDO REGULARIZAÇÃO (Ridge/Lasso)") 
        print("=" * 50) X = df_features[features_otimas] y = df_features['preco'] X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42, stratify=df_features['localizacao'] ) # Diferentes valores de alpha para regularização alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0] resultados_reg = [] 
        for alpha in alphas: 
        print(f"\n Alpha = {alpha}") # Ridge ridge = Pipeline([ ('scaler', StandardScaler()), ('regressor', Ridge(alpha=alpha)) ]) ridge.fit(X_train, y_train) y_pred_ridge = ridge.predict(X_test) r2_ridge = r2_score(y_test, y_pred_ridge) rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge)) # Lasso lasso = Pipeline([ ('scaler', StandardScaler()), ('regressor', Lasso(alpha=alpha, max_iter=2000)) ]) lasso.fit(X_train, y_train) y_pred_lasso = lasso.predict(X_test) r2_lasso = r2_score(y_test, y_pred_lasso) rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso)) 
        print(f" • Ridge - R²: {r2_ridge:.4f}, RMSE: R$ {rmse_ridge:,.0f}") 
        print(f" • Lasso - R²: {r2_lasso:.4f}, RMSE: R$ {rmse_lasso:,.0f}") resultados_reg.append({ 'alpha': alpha, 'ridge_r2': r2_ridge, 'ridge_rmse': rmse_ridge, 'lasso_r2': r2_lasso, 'lasso_rmse': rmse_lasso }) 
        return pd.DataFrame(resultados_reg) 

def executar_investigacao_completa(self): 
    """Executa investigação completa do overfitting."""
 
        if not self.carregar_dados(): 
        return False # Preparar features df_features, conjuntos_features = self.preparar_features_simples() # 1. Testar tamanhos de amostra com features simples 
        print(f"\n" + "="*80) 
        print("1⃣ INVESTIGAÇÃO: TAMANHO DA AMOSTRA") 
        print("="*80) features_para_teste = conjuntos_features['Simples'] df_tamanhos = self.testar_tamanhos_amostra(df_features, features_para_teste) # 2. Testar complexidade de features 
        print(f"\n" + "="*80) 
        print("2⃣ INVESTIGAÇÃO: COMPLEXIDADE DAS FEATURES") 
        print("="*80) df_complexidade = self.testar_complexidade_features(df_features, conjuntos_features) # 3. Encontrar melhor configuração 
        print(f"\n" + "="*80) 
        print("3⃣ ANÁLISE DOS RESULTADOS") 
        print("="*80) 
        print(" RANKING POR TAMANHO DE AMOSTRA (R² teste):") ranking_tamanho = df_tamanhos.sort_values('r2_test', ascending=False) 
        for idx, row in ranking_tamanho.head(3).iterrows(): 
        print(f" {idx+1}. {row['tamanho_treino']*100:.0f}% treino: R² = {row['r2_test']:.4f}, Overfitting = {row['overfitting']:.4f}") 
        print(f"\n RANKING POR COMPLEXIDADE (R² teste):") ranking_complexidade = df_complexidade.sort_values('r2_test', ascending=False) 
        for idx, row in ranking_complexidade.iterrows(): 
        print(f" {idx+1}. {row['conjunto']}: R² = {row['r2_test']:.4f}, Overfitting = {row['overfitting']:.4f}") # 4. Testar regularização na melhor configuração melhor_conjunto = ranking_complexidade.iloc[0]['conjunto'] features_otimas = conjuntos_features[melhor_conjunto] 
        print(f"\n" + "="*80) 
        print("4⃣ INVESTIGAÇÃO: REGULARIZAÇÃO") 
        print("="*80) 
        print(f" Testando regularização com: {melhor_conjunto}") df_regularizacao = self.testar_regularizacao(df_features, features_otimas) # 5. Recomendações finais self.gerar_recomendacoes(df_tamanhos, df_complexidade, df_regularizacao, melhor_conjunto, features_otimas) 
        return True 

def gerar_recomendacoes(self, df_tamanhos, df_complexidade, df_regularizacao, melhor_conjunto, features_otimas): 
    """Gera recomendações baseadas na investigação."""
 
        print(f"\n" + "="*80) 
        print(" RECOMENDAÇÕES FINAIS") 
        print("="*80) # Melhor tamanho de amostra melhor_tamanho = df_tamanhos.loc[df_tamanhos['overfitting'].abs().idxmin()] # Melhor regularização melhor_ridge = df_regularizacao.loc[df_regularizacao['ridge_r2'].idxmax()] melhor_lasso = df_regularizacao.loc[df_regularizacao['lasso_r2'].idxmax()] 
        print(f" CONFIGURAÇÃO RECOMENDADA:") 
        print(f" • Conjunto de features: {melhor_conjunto}") 
        print(f" • Features: {features_otimas}") 
        print(f" • Tamanho de treino: {melhor_tamanho['tamanho_treino']*100:.0f}%") 
        print(f" • Regularização Ridge: α = {melhor_ridge['alpha']}") 
        print(f" • Regularização Lasso: α = {melhor_lasso['alpha']}") 
        print(f"\n MÉTRICAS ESPERADAS:") melhor_config = df_complexidade[df_complexidade['conjunto'] == melhor_conjunto].iloc[0] 
        print(f" • R² teste: {melhor_config['r2_test']:.4f}") 
        print(f" • RMSE: R$ {melhor_config['rmse_test']:,.0f}") 
        print(f" • MAE: R$ {melhor_config['mae_test']:,.0f}") 
        print(f" • Overfitting: {melhor_config['overfitting']:.4f}") 
        print(f"\n DIAGNÓSTICO DO OVERFITTING:") 
        if melhor_config['overfitting'] > 0.05: 
        print(" OVERFITTING DETECTADO - usar regularização") elif melhor_config['overfitting'] < -0.05: 
        print(" UNDERFITTING DETECTADO - adicionar complexidade") 
        else: 
        print(" EQUILÍBRIO ADEQUADO entre viés e variância") 
        print(f"\n CONCLUSÃO:") 
        if melhor_config['r2_test'] > 0.85: 
        print(" MODELO EXCELENTE - R² > 85%") elif melhor_config['r2_test'] > 0.70: 
        print(" MODELO BOM - R² > 70%") 
        else: 
        print(" MODELO PRECISA MELHORIAS - R² < 70%") 

def main(): investigacao = InvestigacaoOverfitting() investigacao.executar_investigacao_completa() 
        if __name__ == "__main__": main()