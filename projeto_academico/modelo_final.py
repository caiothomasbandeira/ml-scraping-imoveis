#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MODELO FINAL DE REGRESSÃO LINEAR
=================================

Implementa o modelo final otimizado com R² = 91.6%
Baseado nos testes de features adicionais.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import pickle
import warnings

warnings.filterwarnings('ignore')

# Configurar matplotlib
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")
sns.set_palette("husl")


class ModeloFinalImoveis:
    def __init__(self):
        self.df = None
        self.modelo = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.features_utilizadas = None

    def carregar_e_preparar_dados(self):
        """Carrega e prepara os dados com as features otimizadas."""
        print("📊 MODELO FINAL DE PREDIÇÃO DE PREÇOS DE IMÓVEIS")
        print("=" * 60)
        
        # Carregar dados
        self.df = pd.read_csv("data/dados_limpos.csv")
        print(f"📊 Dados carregados: {len(self.df):,} registros")
        
        # Preparar features (mesmo processo do teste que deu 91.6%)
        df_features = self.df.copy()
        
        # 1. Encoding básico
        le = LabelEncoder()
        df_features['localizacao_encoded'] = le.fit_transform(df_features['localizacao'])
        
        # 2. One-hot para tipo
        tipo_dummies = pd.get_dummies(df_features['tipo'], prefix='tipo')
        df_features = pd.concat([df_features, tipo_dummies], axis=1)
        
        # 3. Interações matemáticas
        df_features['area_x_quartos'] = df_features['area'] * df_features['quartos']
        df_features['area_x_banheiros'] = df_features['area'] * df_features['banheiros']
        df_features['quartos_x_banheiros'] = df_features['quartos'] * df_features['banheiros']
        
        # 4. Razões e proporções
        df_features['banheiros_por_quarto'] = df_features['banheiros'] / df_features['quartos']
        df_features['area_total_quartos_banheiros'] = df_features['area'] / (df_features['quartos'] + df_features['banheiros'])
        
        # 5. Features categóricas baseadas em faixas
        df_features['faixa_area'] = pd.cut(
            df_features['area'], 
            bins=[0, 50, 100, 150, float('inf')], 
            labels=['Pequeno', 'Medio', 'Grande', 'Extra_Grande']
        )
        faixa_dummies = pd.get_dummies(df_features['faixa_area'], prefix='area')
        df_features = pd.concat([df_features, faixa_dummies], axis=1)
        
        # 6. Features de preço engenheiradas
        df_features['log_preco_m2'] = np.log1p(df_features['preco_por_m2'])
        
        # 7. Grupos de localização por preço
        loc_grupos = df_features.groupby('localizacao')['preco'].mean().sort_values(ascending=False)
        n_locs = len(loc_grupos)
        alto = loc_grupos.index[:n_locs//3]
        medio = loc_grupos.index[n_locs//3:2*n_locs//3]
        baixo = loc_grupos.index[2*n_locs//3:]
        
        df_features['grupo_preco_alto'] = df_features['localizacao'].isin(alto).astype(int)
        df_features['grupo_preco_medio'] = df_features['localizacao'].isin(medio).astype(int)
        df_features['grupo_preco_baixo'] = df_features['localizacao'].isin(baixo).astype(int)
        
        # 8. Features finais (V7 - melhor resultado)
        self.features_utilizadas = [
            'area', 'quartos', 'banheiros', 'localizacao_encoded', 'preco_por_m2',
            'area_por_quarto', 'tipo_Casa', 'tipo_Apartamento', 'tipo_Cobertura',
            'area_x_quartos', 'banheiros_por_quarto', 'log_preco_m2',
            'area_Pequeno', 'area_Medio', 'area_Grande', 'area_Extra_Grande',
            'grupo_preco_alto', 'grupo_preco_medio', 'grupo_preco_baixo',
            'area_x_banheiros', 'quartos_x_banheiros', 'area_total_quartos_banheiros'
        ]
        
        print(f"🔧 Features preparadas: {len(self.features_utilizadas)} variáveis")
        return df_features

    def treinar_modelo(self, df_features):
        """Treina o modelo final."""
        print(f"\n🤖 TREINAMENTO DO MODELO FINAL")
        print("=" * 40)
        
        # Preparar dados
        X = df_features[self.features_utilizadas]
        y = df_features['preco']
        
        # Divisão treino/teste
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=df_features['localizacao']
        )
        
        # Modelo com pipeline
        self.modelo = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
        
        # Treinar
        print("🔄 Treinando modelo...")
        self.modelo.fit(X_train, y_train)
        
        # Predições
        y_pred_train = self.modelo.predict(X_train)
        self.y_pred = self.modelo.predict(self.X_test)
        
        # Métricas
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(self.y_test, self.y_pred)
        rmse_test = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        mae_test = mean_absolute_error(self.y_test, self.y_pred)
        
        # Validação cruzada
        cv_scores = cross_val_score(self.modelo, X_train, y_train, cv=5, scoring='r2')
        
        print(f"✅ Modelo treinado com sucesso!")
        print(f"   • R² treino: {r2_train:.4f}")
        print(f"   • R² teste: {r2_test:.4f}")
        print(f"   • R² CV: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"   • RMSE: R$ {rmse_test:,.0f}")
        print(f"   • MAE: R$ {mae_test:,.0f}")
        print(f"   • Erro relativo: {(mae_test/self.y_test.mean())*100:.1f}%")
        print(f"   • Overfitting: {r2_train - r2_test:.4f}")
        
        return {
            'r2_train': r2_train,
            'r2_test': r2_test,
            'r2_cv_mean': cv_scores.mean(),
            'r2_cv_std': cv_scores.std(),
            'rmse_test': rmse_test,
            'mae_test': mae_test,
            'erro_relativo': (mae_test/self.y_test.mean())*100,
            'overfitting': r2_train - r2_test
        }

    def executar_pipeline_completo(self):
        """Executa pipeline completo do modelo final."""
        # Preparar dados
        df_features = self.carregar_e_preparar_dados()
        
        # Treinar modelo
        metricas = self.treinar_modelo(df_features)
        
        # Salvar modelo
        with open('data/modelo_final_imoveis.pkl', 'wb') as f:
            pickle.dump(self.modelo, f)
        
        print(f"\n" + "="*60)
        print("🎉 MODELO FINAL CONCLUÍDO COM SUCESSO!")
        print("="*60)
        print(f"📊 R² Final: {metricas['r2_test']:.4f} ({metricas['r2_test']*100:.1f}%)")
        print(f"📊 Erro Médio: {metricas['erro_relativo']:.1f}%")
        print(f"💾 Modelo salvo: data/modelo_final_imoveis.pkl")
        print("="*60)
        
        return metricas


def main():
    modelo = ModeloFinalImoveis()
    metricas = modelo.executar_pipeline_completo()
    return metricas


if __name__ == "__main__":
    main()