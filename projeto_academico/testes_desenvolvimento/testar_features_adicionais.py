#!/usr/bin/env python3 # -*- coding: utf-8 -*- 
    """ TESTE DE FEATURES ADICIONAIS ============================ Testa adicionar mais features ao modelo base (76.1% R²) de forma controlada para verificar se podemos melhorar a performance sem overfitting. """
 
import pandas as pd 
import numpy as np 
from sklearn.model_selection 
import train_test_split, cross_val_score from sklearn.linear_model 
import LinearRegression, Ridge 
from sklearn.preprocessing 
import StandardScaler, LabelEncoder from sklearn.metrics 
import mean_squared_error, r2_score, mean_absolute_error 
from sklearn.pipeline 
import Pipeline 
import warnings warnings.filterwarnings('ignore') 

class TesteFeatures: 

def __init__(self): self.df = None self.modelo_base = None self.resultados = [] 

def carregar_dados(self): 
    """Carrega os dados."""
 
        print(" TESTE DE FEATURES ADICIONAIS") 
        print("=" * 50) self.df = pd.read_csv("data/dados_limpos.csv") 
        print(f" Dados carregados: {len(self.df):,} registros") 
        return True 

def preparar_features_expandidas(self): 
    """Prepara features expandidas de forma controlada."""
 
        print(f"\n PREPARANDO FEATURES EXPANDIDAS") 
        print("=" * 40) df_features = self.df.copy() # Encoding básico le = LabelEncoder() df_features['localizacao_encoded'] = le.fit_transform(df_features['localizacao']) # One-hot para tipo tipo_dummies = pd.get_dummies(df_features['tipo'], prefix='tipo') df_features = pd.concat([df_features, tipo_dummies], axis=1) # CRIAR FEATURES ENGENHEIRADAS ADICIONAIS # 1. Interações matemáticas df_features['area_x_quartos'] = df_features['area'] * df_features['quartos'] df_features['area_x_banheiros'] = df_features['area'] * df_features['banheiros'] df_features['quartos_x_banheiros'] = df_features['quartos'] * df_features['banheiros'] # 2. Razões e proporções df_features['banheiros_por_quarto'] = df_features['banheiros'] / df_features['quartos'] df_features['area_total_quartos_banheiros'] = df_features['area'] / (df_features['quartos'] + df_features['banheiros']) # 3. Features categóricas baseadas em faixas df_features['faixa_area'] = pd.cut(df_features['area'], bins=[0, 50, 100, 150, float('inf')], labels=['Pequeno', 'Medio', 'Grande', 'Extra_Grande']) faixa_dummies = pd.get_dummies(df_features['faixa_area'], prefix='area') df_features = pd.concat([df_features, faixa_dummies], axis=1) # 4. Features baseadas em preço por m² df_features['log_preco_m2'] = np.log1p(df_features['preco_por_m2']) df_features['preco_m2_standardized'] = (df_features['preco_por_m2'] - df_features['preco_por_m2'].mean()) / df_features['preco_por_m2'].std() # 5. Features de localização avançadas # Criar grupos de localização por preço médio loc_grupos = df_features.groupby('localizacao')['preco'].mean().sort_values(ascending=False) # Dividir em 3 grupos: Alto, Médio, Baixo n_locs = len(loc_grupos) alto = loc_grupos.index[:n_locs//3] medio = loc_grupos.index[n_locs//3:2*n_locs//3] baixo = loc_grupos.index[2*n_locs//3:] df_features['grupo_preco_alto'] = df_features['localizacao'].isin(alto).astype(int) df_features['grupo_preco_medio'] = df_features['localizacao'].isin(medio).astype(int) df_features['grupo_preco_baixo'] = df_features['localizacao'].isin(baixo).astype(int) 
        return df_features 

def definir_conjuntos_features(self, df_features): 
    """Define conjuntos progressivos de features."""
 # BASE (76.1% R²) - confirmed working features_base = ['area', 'quartos', 'banheiros', 'localizacao_encoded', 'preco_por_m2'] # ADIÇÃO 1: Características básicas do imóvel features_v1 = features_base + ['area_por_quarto'] # ADIÇÃO 2: Tipo de imóvel features_v2 = features_v1 + ['tipo_Casa', 'tipo_Apartamento', 'tipo_Cobertura'] # ADIÇÃO 3: Interações matemáticas features_v3 = features_v2 + ['area_x_quartos', 'banheiros_por_quarto'] # ADIÇÃO 4: Features de preço engenheiradas features_v4 = features_v3 + ['log_preco_m2'] # ADIÇÃO 5: Faixas de área features_v5 = features_v4 + ['area_Pequeno', 'area_Medio', 'area_Grande', 'area_Extra_Grande'] # ADIÇÃO 6: Grupos de localização features_v6 = features_v5 + ['grupo_preco_alto', 'grupo_preco_medio', 'grupo_preco_baixo'] # ADIÇÃO 7: Todas as interações features_v7 = features_v6 + ['area_x_banheiros', 'quartos_x_banheiros', 'area_total_quartos_banheiros'] conjuntos = { 'Base_76pct': features_base, 'V1_+area_por_quarto': features_v1, 'V2_+tipo_imovel': features_v2, 'V3_+interacoes_basicas': features_v3, 'V4_+preco_engenheirado': features_v4, 'V5_+faixas_area': features_v5, 'V6_+grupos_localizacao': features_v6, 'V7_+todas_interacoes': features_v7 } 
        print(" CONJUNTOS DE FEATURES PREPARADOS:") 
        for nome, features in conjuntos.items(): 
        print(f" • {nome}: {len(features)} features") 
        return conjuntos 

def testar_conjunto_features(self, df_features, nome, features): 
    """Testa um conjunto específico de features."""
 
        try: # Verificar se todas as features existem features_disponiveis = [f 
        for f in features 
        if f in df_features.columns] if len(features_disponiveis) != len(features): missing = set(features) - set(features_disponiveis) 
        print(f" Features ausentes: {missing}") features = features_disponiveis X = df_features[features] y = df_features['preco'] # Verificar valores NaN/inf 
        if X.isnull().any().any() or np.isinf(X).any().any(): 
        print(f" Removendo valores inválidos...") mask = ~(X.isnull().any(axis=1) | np.isinf(X).any(axis=1)) X = X[mask] y = y[mask] # Divisão dos dados (mesma estratégia da investigação) X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42, stratify=df_features.loc[X.index, 'localizacao'] ) # Modelo linear simples pipeline = Pipeline([ ('scaler', StandardScaler()), ('regressor', LinearRegression()) ]) # Treinar pipeline.fit(X_train, y_train) # Predições y_pred_train = pipeline.predict(X_train) y_pred_test = pipeline.predict(X_test) # Métricas r2_train = r2_score(y_train, y_pred_train) r2_test = r2_score(y_test, y_pred_test) rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test)) mae_test = mean_absolute_error(y_test, y_pred_test) # Validação cruzada cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2') # Erro relativo erro_relativo_mae = (mae_test / y_test.mean()) * 100 erro_relativo_rmse = (rmse_test / y_test.mean()) * 100 resultado = { 'nome': nome, 'n_features': len(features), 'r2_train': r2_train, 'r2_test': r2_test, 'r2_cv_mean': cv_scores.mean(), 'r2_cv_std': cv_scores.std(), 'rmse_test': rmse_test, 'mae_test': mae_test, 'erro_rel_mae': erro_relativo_mae, 'erro_rel_rmse': erro_relativo_rmse, 'overfitting': r2_train - r2_test, 'melhoria_vs_base': None # Será calculado depois } 
        return resultado 
        except Exception as e: 
        print(f" Erro: {str(e)}") 
        return None 

def executar_teste_progressivo(self): 
    """Executa teste progressivo de features."""
 
        if not self.carregar_dados(): 
        return False # Preparar features df_features = self.preparar_features_expandidas() conjuntos = self.definir_conjuntos_features(df_features) 
        print(f"\n" + "="*60) 
        print(" TESTANDO CONJUNTOS DE FEATURES PROGRESSIVAMENTE") 
        print("="*60) resultados = [] r2_base = None 
        for nome, features in conjuntos.items(): 
        print(f"\n Testando: {nome} ({len(features)} features)") resultado = self.testar_conjunto_features(df_features, nome, features) 
        if resultado: # Calcular melhoria vs base 
        if nome == 'Base_76pct': r2_base = resultado['r2_test'] resultado['melhoria_vs_base'] = 0.0 
        else: resultado['melhoria_vs_base'] = resultado['r2_test'] - r2_base resultados.append(resultado) 
        print(f" • R² treino: {resultado['r2_train']:.4f}") 
        print(f" • R² teste: {resultado['r2_test']:.4f}") 
        print(f" • R² CV: {resultado['r2_cv_mean']:.4f} (±{resultado['r2_cv_std']:.4f})") 
        print(f" • RMSE: R$ {resultado['rmse_test']:,.0f}") 
        print(f" • MAE: R$ {resultado['mae_test']:,.0f}") 
        print(f" • Erro rel. MAE: {resultado['erro_rel_mae']:.1f}%") 
        print(f" • Overfitting: {resultado['overfitting']:.4f}") 
        if resultado['melhoria_vs_base'] is not None: 
        print(f" • Melhoria vs Base: {resultado['melhoria_vs_base']:+.4f}") 
        return resultados 

def analisar_resultados(self, resultados): 
    """Analisa os resultados dos testes."""
 
        print(f"\n" + "="*80) 
        print(" ANÁLISE COMPARATIVA DOS RESULTADOS") 
        print("="*80) df_resultados = pd.DataFrame(resultados) # Ranking por R² teste 
        print(" RANKING POR R² TESTE:") ranking = df_resultados.sort_values('r2_test', ascending=False) 
        for i, (idx, row) in enumerate(ranking.iterrows(), 1): melhoria_str = f"({row['melhoria_vs_base']:+.4f})" 
        if row['melhoria_vs_base'] != 0 else "" 
        print(f" {i}. {row['nome']}: R² = {row['r2_test']:.4f} {melhoria_str}") 
        print(f" • Features: {row['n_features']} | MAE: R$ {row['mae_test']:,.0f} | Overfitting: {row['overfitting']:.4f}") # Melhor modelo melhor = ranking.iloc[0] 
        print(f"\n MELHOR CONFIGURAÇÃO:") 
        print(f" • Modelo: {melhor['nome']}") 
        print(f" • R² teste: {melhor['r2_test']:.4f} ({melhor['r2_test']*100:.1f}%)") 
        print(f" • Melhoria vs Base: {melhor['melhoria_vs_base']:+.4f}") 
        print(f" • RMSE: R$ {melhor['rmse_test']:,.0f}") 
        print(f" • MAE: R$ {melhor['mae_test']:,.0f} ({melhor['erro_rel_mae']:.1f}%)") 
        print(f" • Overfitting: {melhor['overfitting']:.4f}") # Diagnóstico 
        print(f"\n DIAGNÓSTICO:") 
        if melhor['melhoria_vs_base'] > 0.02: 
        print(" MELHORIA SIGNIFICATIVA - vale adicionar features!") elif melhor['melhoria_vs_base'] > 0.005: 
        print(" MELHORIA MODESTA - features adicionais ajudam") 
        else: 
        print(" MELHORIA MÍNIMA - modelo base já era ótimo") 
        if abs(melhor['overfitting']) < 0.02: 
        print(" SEM OVERFITTING - modelo equilibrado") 
        else: 
        print(" POSSÍVEL OVERFITTING - considerar regularização") 
        return melhor 

def main(): teste = TesteFeatures() resultados = teste.executar_teste_progressivo() 
        if resultados: melhor = teste.analisar_resultados(resultados) 
        print(f"\n TESTE CONCLUÍDO!") 
        print(f"Melhor R²: {melhor['r2_test']:.4f}") 
        print(f"Melhoria obtida: {melhor['melhoria_vs_base']:+.4f}") 
        if __name__ == "__main__": main()