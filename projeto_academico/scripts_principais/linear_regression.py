#!/usr/bin/env python3 # -*- coding: utf-8 -*- 
    """ MODELO DE REGRESSÃO LINEAR PARA ATIVIDADE PRÁTICA =============================================== Módulo responsável pelo treinamento do modelo de regressão linear. Conforme ESPECIFICAMENTE solicitado na atividade prática. Funcionalidades: - Divisão dos dados em treino e teste - Treinamento do modelo de regressão linear usando Scikit-learn - Avaliação com R² e MSE conforme pedido pelo professor """
 
import pandas as pd 
import numpy as np 
from sklearn.model_selection 
import train_test_split from sklearn.linear_model 
import LinearRegression 
from sklearn.preprocessing 
import StandardScaler, LabelEncoder from sklearn.metrics 
import mean_squared_error, r2_score, mean_absolute_error 
import joblib 
import matplotlib.pyplot as plt 

class LinearRegressionModel: 
    """ Modelo de Regressão Linear para previsão de preços de imóveis. Implementação EXATA do que foi solicitado na atividade prática: - Regressão Linear usando Scikit-learn - Divisão treino/teste - Métricas R² e MSE """
 

def __init__(self): 
    """Inicializa o modelo de regressão linear."""
 self.model = LinearRegression() self.scaler = StandardScaler() self.label_encoders = {} self.feature_names = [] self.X_train = None self.X_test = None self.y_train = None self.y_test = None self.y_pred = None 

def treinar_modelo(self, df, test_size=0.2, random_state=42): 
    """ Treina o modelo de regressão linear conforme solicitado. Args: df (pd.DataFrame): Dataset limpo test_size (float): Proporção para teste (padrão 20%) random_state (int): Seed para reprodutibilidade Returns: dict: Resultados da avaliação do modelo """
 
        print("Iniciando treinamento do modelo de Regressão Linear...") # 1. Preparar features e target 
        print("Preparando features e target...") X, y = self._preparar_dados(df) # 2. Dividir dados em treino e teste (conforme solicitado) 
        print(f"Dividindo dados: {int((1-test_size)*100)}% treino, {int(test_size*100)}% teste") self.X_train, self.X_test, self.y_train, self.y_test = train_test_split( X, y, test_size=test_size, random_state=random_state ) 
        print(f"Conjunto de treino: {len(self.X_train)} registros") 
        print(f"Conjunto de teste: {len(self.X_test)} registros") # 3. Padronizar features (boa prática) 
        print("Padronizando features...") self.X_train_scaled = self.scaler.fit_transform(self.X_train) self.X_test_scaled = self.scaler.transform(self.X_test) # 4. Treinar modelo de Regressão Linear 
        print("Treinando modelo de Regressão Linear...") self.model.fit(self.X_train_scaled, self.y_train) # 5. Fazer predições 
        print("Realizando predições...") y_train_pred = self.model.predict(self.X_train_scaled) self.y_pred = self.model.predict(self.X_test_scaled) # 6. Avaliar modelo (R² e MSE conforme solicitado) 
        print("Avaliando performance do modelo...") results = self._avaliar_modelo(y_train_pred) # 7. Salvar modelo self._salvar_modelo() # 8. Gerar relatório self._gerar_relatorio(results) 
        print("Treinamento concluído!") 
        return results 

def _preparar_dados(self, df): 
    """Prepara features e target para o modelo."""
 # Resetar índice para evitar problemas df = df.reset_index(drop=True) # Target (variável a ser prevista) y = df['preco'].copy() # Features numéricas numeric_features = ['area', 'quartos', 'banheiros', 'preco_por_m2'] X_numeric = df[numeric_features].copy() # Features categóricas - encoding categorical_features = ['localizacao', 'tipo'] X_categorical = pd.DataFrame(index=df.index) 
        for col in categorical_features: 
        if col in df.columns: # Usar Label Encoding para simplicidade le = LabelEncoder() X_categorical[col] = le.fit_transform(df[col]) self.label_encoders[col] = le # Combinar features X = pd.concat([X_numeric, X_categorical], axis=1) self.feature_names = X.columns.tolist() 
        print(f"Features utilizadas: {self.feature_names}") 
        print(f"Shape dos dados: X={X.shape}, y={y.shape}") 
        return X, y 

def _avaliar_modelo(self, y_train_pred): 
    """ Avalia o modelo usando as métricas solicitadas na atividade. Returns: dict: Métricas de avaliação """
 # Métricas no conjunto de treino r2_train = r2_score(self.y_train, y_train_pred) mse_train = mean_squared_error(self.y_train, y_train_pred) rmse_train = np.sqrt(mse_train) # Métricas no conjunto de teste (principais) r2_test = r2_score(self.y_test, self.y_pred) mse_test = mean_squared_error(self.y_test, self.y_pred) rmse_test = np.sqrt(mse_test) mae_test = mean_absolute_error(self.y_test, self.y_pred) results = { 'r2_score': r2_test, 'mse': mse_test, 'rmse': rmse_test, 'mae': mae_test, 'r2_train': r2_train, 'mse_train': mse_train, 'rmse_train': rmse_train, 'n_train': len(self.y_train), 'n_test': len(self.y_test), 'feature_importance': dict(zip(self.feature_names, self.model.coef_)) } # Exibir resultados 
        print("\n" + "="*50) 
        print("RESULTADOS DA REGRESSÃO LINEAR") 
        print("="*50) 
        print(f"R² Score (Teste): {r2_test:.4f}") 
        print(f"MSE (Teste): {mse_test:,.0f}") 
        print(f"RMSE (Teste): R$ {rmse_test:,.2f}") 
        print(f"MAE (Teste): R$ {mae_test:,.2f}") print() 
        print(f"R² Score (Treino): {r2_train:.4f}") 
        print(f"RMSE (Treino): R$ {rmse_train:,.2f}") 
        print("="*50) # Interpretação do R² 
        if r2_test >= 0.8: interpretacao = "Excelente ajuste" elif r2_test >= 0.6: interpretacao = "Bom ajuste" elif r2_test >= 0.4: interpretacao = "Ajuste moderado" 
        else: interpretacao = "Ajuste fraco" 
        print(f"Interpretação: {interpretacao}") 
        print(f"O modelo explica {r2_test*100:.1f}% da variância dos preços") 
        return results 

def _salvar_modelo(self): 
    """Salva o modelo treinado."""
 model_data = { 'model': self.model, 'scaler': self.scaler, 'label_encoders': self.label_encoders, 'feature_names': self.feature_names } joblib.dump(model_data, 'data/modelo_regressao_linear.pkl') 
        print("Modelo salvo em 'data/modelo_regressao_linear.pkl'") 

def _gerar_relatorio(self, results): 
    """Gera relatório detalhado do modelo."""
 with open('data/relatorio_modelo.txt', 'w', encoding='utf-8') as f: f.write("RELATÓRIO DO MODELO DE REGRESSÃO LINEAR\n") f.write("=" * 50 + "\n\n") f.write(f"Data do treinamento: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M:%S')}\n") f.write(f"Algoritmo: Regressão Linear (Scikit-learn)\n\n") f.write("CONFIGURAÇÃO DO MODELO\n") f.write("-" * 30 + "\n") f.write(f"Features utilizadas: {', '.join(self.feature_names)}\n") f.write(f"Registros de treino: {results['n_train']}\n") f.write(f"Registros de teste: {results['n_test']}\n\n") f.write("MÉTRICAS DE PERFORMANCE\n") f.write("-" * 30 + "\n") f.write(f"R² Score (Teste): {results['r2_score']:.4f}\n") f.write(f"MSE (Teste): {results['mse']:,.0f}\n") f.write(f"RMSE (Teste): R$ {results['rmse']:,.2f}\n") f.write(f"MAE (Teste): R$ {results['mae']:,.2f}\n\n") f.write("COEFICIENTES DO MODELO\n") f.write("-" * 30 + "\n") 
        for feature, coef in results['feature_importance'].items(): f.write(f"{feature}: {coef:,.2f}\n") 

def prever(self, dados_novos): 
    """ Faz predições para novos dados. Args: dados_novos (pd.DataFrame): Dados para predição Returns: np.array: Previsões de preços """
 # Preparar dados da mesma forma que no treinamento X_new = dados_novos[self.feature_names].copy() # Aplicar encoders nas variáveis categóricas 
        for col, encoder in self.label_encoders.items(): 
        if col in X_new.columns: X_new[col] = encoder.transform(X_new[col]) # Padronizar X_new_scaled = self.scaler.transform(X_new) # Prever previsoes = self.model.predict(X_new_scaled) 
        return previsoes # Função para teste 
        if __name__ == "__main__": # Teste com dados simulados data = { 'preco': [400000, 350000, 600000, 250000, 500000], 'area': [80, 70, 120, 50, 90], 'quartos': [3, 2, 4, 2, 3], 'banheiros': [2, 1, 3, 1, 2], 'localizacao': ['Asa Norte', 'Ceilândia', 'Asa Sul', 'Taguatinga', 'Guará'], 'tipo': ['Apartamento', 'Apartamento', 'Casa', 'Apartamento', 'Apartamento'], 'preco_por_m2': [5000, 5000, 5000, 5000, 5556] } df_test = pd.DataFrame(data) model = LinearRegressionModel() results = model.treinar_modelo(df_test) 
        print(f"\nTeste concluído! R² = {results['r2_score']:.4f}")