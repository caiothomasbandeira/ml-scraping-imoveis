#!/usr/bin/env python3 # -*- coding: utf-8 -*- 
    """ ANÁLISE EXPLORATÓRIA AVANÇADA DOS DADOS ======================================= Análise profunda e detalhada dos dados imobiliários antes da modelagem. Inclui análises estatísticas, visualizações e validações completas. """
 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy 
import stats from scipy.stats 
import pearsonr, spearmanr, normaltest 
import warnings warnings.filterwarnings('ignore') # Configurar matplotlib para português plt.rcParams['font.size'] = 10 plt.rcParams['figure.figsize'] = (12, 8) sns.set_style("whitegrid") sns.set_palette("husl") 

class AnaliseAvancada: 

def __init__(self): 
    """Inicializa a análise avançada."""
 self.df = None self.insights = [] 

def carregar_dados(self): 
    """Carrega e prepara os dados."""
 
        print("=" * 80) 
        print(" ANÁLISE EXPLORATÓRIA AVANÇADA DOS DADOS IMOBILIÁRIOS") 
        print("=" * 80) # Carregar dados limpos 
        try: self.df = pd.read_csv("data/dados_limpos.csv") 
        print(f" Dados carregados: {len(self.df):,} registros") 
        except FileNotFoundError: 
        print(" Arquivo dados_limpos.csv não encontrado!") 
        print("Execute primeiro a análise básica.") 
        return False return True 

def analise_descritiva_avancada(self): 
    """Análise descritiva detalhada."""
 
        print(f"\n" + "="*60) 
        print(" 1. ANÁLISE DESCRITIVA AVANÇADA") 
        print("="*60) # Informações gerais 
        print(f" INFORMAÇÕES GERAIS:") 
        print(f" • Shape: {self.df.shape}") 
        print(f" • Memória utilizada: {self.df.memory_usage(deep=True).sum() / 1024**2:.1f} MB") 
        print(f" • Tipos de dados:") 
        for col, dtype in self.df.dtypes.items(): 
        print(f" - {col}: {dtype}") # Análise de variáveis numéricas numericas = ['preco', 'area', 'quartos', 'banheiros', 'preco_por_m2', 'area_por_quarto'] numericas = [col 
        for col in numericas 
        if col in self.df.columns] 
        print(f"\n ESTATÍSTICAS DESCRITIVAS DETALHADAS:") desc = self.df[numericas].describe(percentiles=[.01, .05, .10, .25, .50, .75, .90, .95, .99]) 
        print(desc.round(2)) # Análise de assimetria e curtose 
        print(f"\n ANÁLISE DE DISTRIBUIÇÃO:") 
        for col in numericas: skewness = stats.skew(self.df[col]) kurtosis = stats.kurtosis(self.df[col]) 
        print(f" • {col}:") 
        print(f" - Assimetria: {skewness:.3f} {'(Assimétrica à direita)' 
        if skewness > 1 else '(Assimétrica à esquerda)' if skewness < -1 else '(Aproximadamente simétrica)'}") 
        print(f" - Curtose: {kurtosis:.3f} {'(Leptocúrtica)' 
        if kurtosis > 0 else '(Platicúrtica)'}") # Análise de valores únicos 
        print(f"\n ANÁLISE DE CARDINALIDADE:") 
        for col in self.df.columns: unique_count = self.df[col].nunique() unique_pct = (unique_count / len(self.df)) * 100 
        print(f" • {col}: {unique_count:,} únicos ({unique_pct:.1f}%)") 

def analise_correlacoes_avancada(self): 
    """Análise de correlações detalhada."""
 
        print(f"\n" + "="*60) 
        print(" 2. ANÁLISE DE CORRELAÇÕES AVANÇADA") 
        print("="*60) numericas = ['preco', 'area', 'quartos', 'banheiros', 'preco_por_m2'] numericas = [col 
        for col in numericas 
        if col in self.df.columns] # Correlação de Pearson 
        print(f" CORRELAÇÃO DE PEARSON:") corr_pearson = self.df[numericas].corr(method='pearson') 
        print(corr_pearson.round(3)) # Correlação de Spearman 
        print(f"\n CORRELAÇÃO DE SPEARMAN (não-linear):") corr_spearman = self.df[numericas].corr(method='spearman') 
        print(corr_spearman.round(3)) # Testes de significância 
        print(f"\n TESTES DE SIGNIFICÂNCIA DAS CORRELAÇÕES:") 
        for i, col1 in enumerate(numericas): 
        for j, col2 in enumerate(numericas): 
        if i < j: # Pearson r_pearson, p_pearson = pearsonr(self.df[col1], self.df[col2]) # Spearman r_spearman, p_spearman = spearmanr(self.df[col1], self.df[col2]) 
        print(f" • {col1} vs {col2}:") 
        print(f" - Pearson: r={r_pearson:.3f}, p={p_pearson:.2e} {'' 
        if p_pearson < 0.05 else ''}") 
        print(f" - Spearman: r={r_spearman:.3f}, p={p_spearman:.2e} {'' 
        if p_spearman < 0.05 else ''}") # Identificar correlações fortes 
        print(f"\n CORRELAÇÕES SIGNIFICATIVAS (|r| > 0.3):") mask = np.abs(corr_pearson) > 0.3 
        for i in range(len(corr_pearson)): 
        for j in range(i+1, len(corr_pearson)): 
        if mask.iloc[i, j]: col1, col2 = corr_pearson.index[i], corr_pearson.columns[j] r = corr_pearson.iloc[i, j] interpretacao = "Forte" 
        if abs(r) > 0.7 else "Moderada" direcao = "positiva" if r > 0 else "negativa" 
        print(f" • {col1} ↔ {col2}: {r:.3f} ({interpretacao} {direcao})") 

def analise_por_localizacao(self): 
    """Análise detalhada por localização."""
 
        print(f"\n" + "="*60) 
        print(" 3. ANÁLISE POR LOCALIZAÇÃO") 
        print("="*60) # Estatísticas por localização loc_stats = self.df.groupby('localizacao').agg({ 'preco': ['count', 'mean', 'median', 'std', 'min', 'max'], 'area': ['mean', 'median'], 'quartos': ['mean'], 'banheiros': ['mean'], 'preco_por_m2': ['mean', 'median', 'std'] }).round(0) # Achatar nomes das colunas loc_stats.columns = ['_'.join(col).strip() 
        for col in loc_stats.columns] 
        print(f" ESTATÍSTICAS DETALHADAS POR LOCALIZAÇÃO:") 
        print(loc_stats.sort_values('preco_mean', ascending=False)) # Análise de variabilidade 
        print(f"\n ANÁLISE DE VARIABILIDADE POR LOCALIZAÇÃO:") 
        for local in self.df['localizacao'].unique(): dados_local = self.df[self.df['localizacao'] == local]['preco'] cv = (dados_local.std() / dados_local.mean()) * 100 
        print(f" • {local}: CV = {cv:.1f}% ({'Alta variabilidade' 
        if cv > 50 else 'Média variabilidade' if cv > 30 else 'Baixa variabilidade'})") # Testes estatísticos entre regiões 
        print(f"\n TESTE ANOVA - DIFERENÇAS ENTRE REGIÕES:") grupos_preco = [self.df[self.df['localizacao'] == local]['preco'].values 
        for local in self.df['localizacao'].unique()] f_stat, p_value = stats.f_oneway(*grupos_preco) 
        print(f" • F-statistic: {f_stat:.3f}") 
        print(f" • p-value: {p_value:.2e}") 
        print(f" • Resultado: {' Diferenças significativas entre regiões' 
        if p_value < 0.05 else ' Sem diferenças significativas'}") 

def analise_por_tipo_imovel(self): 
    """Análise por tipo de imóvel."""
 
        print(f"\n" + "="*60) 
        print(" 4. ANÁLISE POR TIPO DE IMÓVEL") 
        print("="*60) tipo_stats = self.df.groupby('tipo').agg({ 'preco': ['count', 'mean', 'median', 'std'], 'area': ['mean', 'median'], 'quartos': ['mean'], 'banheiros': ['mean'], 'preco_por_m2': ['mean', 'median'] }).round(0) tipo_stats.columns = ['_'.join(col).strip() 
        for col in tipo_stats.columns] 
        print(tipo_stats.sort_values('preco_mean', ascending=False)) # Distribuição percentual 
        print(f"\n DISTRIBUIÇÃO POR TIPO:") dist_tipo = self.df['tipo'].value_counts(normalize=True) * 100 
        for tipo, pct in dist_tipo.items(): 
        print(f" • {tipo}: {pct:.1f}%") 

def analise_outliers_detalhada(self): 
    """Análise detalhada de outliers."""
 
        print(f"\n" + "="*60) 
        print(" 5. ANÁLISE DETALHADA DE OUTLIERS") 
        print("="*60) numericas = ['preco', 'area', 'preco_por_m2'] numericas = [col 
        for col in numericas 
        if col in self.df.columns] for col in numericas: 
        print(f"\n {col.upper()}:") # Método IQR Q1 = self.df[col].quantile(0.25) Q3 = self.df[col].quantile(0.75) IQR = Q3 - Q1 lower_bound = Q1 - 1.5 * IQR upper_bound = Q3 + 1.5 * IQR outliers_iqr = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)] # Método Z-score z_scores = np.abs(stats.zscore(self.df[col])) outliers_z = self.df[z_scores > 3] 
        print(f" • IQR Method: {len(outliers_iqr)} outliers ({len(outliers_iqr)/len(self.df)*100:.1f}%)") 
        print(f" • Z-score Method: {len(outliers_z)} outliers ({len(outliers_z)/len(self.df)*100:.1f}%)") 
        print(f" • Bounds IQR: [{lower_bound:,.0f}, {upper_bound:,.0f}]") 
        if len(outliers_iqr) > 0: 
        print(f" • Outliers extremos (IQR):") extremos = outliers_iqr.nlargest(3, col)[['preco', 'area', 'localizacao', col]] 
        for idx, row in extremos.iterrows(): 
        print(f" - {row['localizacao']}: {row[col]:,.0f}") 

def analise_normalidade(self): 
    """Análise de normalidade das distribuições."""
 
        print(f"\n" + "="*60) 
        print(" 6. ANÁLISE DE NORMALIDADE") 
        print("="*60) numericas = ['preco', 'area', 'preco_por_m2'] numericas = [col 
        for col in numericas 
        if col in self.df.columns] for col in numericas: # Teste de Shapiro-Wilk (para amostras menores) # Teste de D'Agostino (para amostras maiores) 
        if len(self.df) > 5000: stat, p_value = normaltest(self.df[col]) teste = "D'Agostino" 
        else: 
from scipy.stats 
import shapiro stat, p_value = shapiro(self.df[col].sample(min(5000, len(self.df)))) teste = "Shapiro-Wilk" 
        print(f" {col.upper()} - Teste {teste}:") 
        print(f" • Estatística: {stat:.3f}") 
        print(f" • p-value: {p_value:.2e}") 
        print(f" • Resultado: {' Não normal' 
        if p_value < 0.05 else ' Normal'}") # Sugerir transformações se necessário if p_value < 0.05: skewness = stats.skew(self.df[col]) 
        if skewness > 1: 
        print(f" • Sugestão: Transformação log (assimetria = {skewness:.2f})") elif skewness < -1: 
        print(f" • Sugestão: Transformação quadrática (assimetria = {skewness:.2f})") 

def analise_qualidade_dados(self): 
    """Análise da qualidade dos dados."""
 
        print(f"\n" + "="*60) 
        print(" 7. ANÁLISE DE QUALIDADE DOS DADOS") 
        print("="*60) # Valores ausentes missing = self.df.isnull().sum() 
        print(f" VALORES AUSENTES:") 
        if missing.sum() == 0: 
        print(" Nenhum valor ausente encontrado") 
        else: 
        for col, count in missing[missing > 0].items(): pct = (count / len(self.df)) * 100 
        print(f" • {col}: {count} ({pct:.1f}%)") # Valores duplicados duplicados = self.df.duplicated().sum() 
        print(f"\n VALORES DUPLICADOS:") 
        print(f" • Registros duplicados: {duplicados} ({duplicados/len(self.df)*100:.1f}%)") # Consistência dos dados 
        print(f"\n CONSISTÊNCIA DOS DADOS:") inconsistencias = [] # Verificar se área por quarto faz sentido 
        if 'area_por_quarto' in self.df.columns: area_por_quarto_baixa = (self.df['area_por_quarto'] < 10).sum() area_por_quarto_alta = (self.df['area_por_quarto'] > 100).sum() 
        if area_por_quarto_baixa > 0: inconsistencias.append(f" {area_por_quarto_baixa} registros com área por quarto < 10m²") 
        if area_por_quarto_alta > 0: inconsistencias.append(f" {area_por_quarto_alta} registros com área por quarto > 100m²") # Verificar preços por m² extremos 
        if 'preco_por_m2' in self.df.columns: preco_m2_baixo = (self.df['preco_por_m2'] < 1000).sum() preco_m2_alto = (self.df['preco_por_m2'] > 100000).sum() 
        if preco_m2_baixo > 0: inconsistencias.append(f" {preco_m2_baixo} registros com preço/m² < R$ 1.000") 
        if preco_m2_alto > 0: inconsistencias.append(f" {preco_m2_alto} registros com preço/m² > R$ 100.000") 
        if not inconsistencias: 
        print(" Nenhuma inconsistência detectada") 
        else: 
        for inc in inconsistencias: 
        print(inc) 

def insights_finais(self): 
    """Gera insights finais para a modelagem."""
 
        print(f"\n" + "="*60) 
        print(" 8. INSIGHTS PARA MODELAGEM") 
        print("="*60) insights = [] # Análise da variável target target_var = 'preco' skewness = stats.skew(self.df[target_var]) 
        if skewness > 1: insights.append(" VARIÁVEL TARGET (preço) tem alta assimetria - considerar transformação log") # Correlações importantes numericas = ['area', 'quartos', 'banheiros', 'preco_por_m2'] numericas = [col 
        for col in numericas 
        if col in self.df.columns] for col in numericas: 
        if col in self.df.columns: corr = self.df[target_var].corr(self.df[col]) 
        if abs(corr) > 0.3: insights.append(f" {col} tem correlação {'forte' 
        if abs(corr) > 0.7 else 'moderada'} com preço (r={corr:.3f})") # Variáveis categóricas importantes 
        if 'localizacao' in self.df.columns: var_precos = self.df.groupby('localizacao')['preco'].var() 
        if var_precos.std() > var_precos.mean(): insights.append(" Grande variabilidade de preços entre localizações - encoding importante") # Outliers Q1 = self.df[target_var].quantile(0.25) Q3 = self.df[target_var].quantile(0.75) IQR = Q3 - Q1 outliers_count = ((self.df[target_var] < Q1 - 1.5*IQR) | (self.df[target_var] > Q3 + 1.5*IQR)).sum() outliers_pct = (outliers_count / len(self.df)) * 100 
        if outliers_pct > 5: insights.append(f" {outliers_pct:.1f}% de outliers - considerar tratamento robusto") # Tamanho da amostra 
        if len(self.df) > 1000: insights.append(" Amostra grande suficiente para modelos complexos") 
        else: insights.append(" Amostra pequena - focar em modelos simples") 
        print(" RECOMENDAÇÕES PARA MODELAGEM:") 
        for i, insight in enumerate(insights, 1): 
        print(f" {i}. {insight}") 
        return insights 

def salvar_relatorio_completo(self): 
    """Salva relatório completo da análise."""
 
        print(f"\n Salvando relatório completo...") with open('data/relatorio_analise_avancada.txt', 'w', encoding='utf-8') as f: f.write("RELATÓRIO DE ANÁLISE EXPLORATÓRIA AVANÇADA\n") f.write("=" * 50 + "\n\n") f.write(f"Data: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M:%S')}\n") f.write(f"Dataset: {len(self.df)} registros\n\n") # Estatísticas principais f.write("ESTATÍSTICAS PRINCIPAIS:\n") f.write(f"- Preço médio: R$ {self.df['preco'].mean():,.0f}\n") f.write(f"- Preço mediano: R$ {self.df['preco'].median():,.0f}\n") f.write(f"- Área média: {self.df['area'].mean():.0f}m²\n") # Correlações importantes f.write("\nCORRELAÇÕES IMPORTANTES:\n") numericas = ['preco', 'area', 'quartos', 'banheiros'] numericas = [col 
        for col in numericas 
        if col in self.df.columns] corr_matrix = self.df[numericas].corr() f.write(corr_matrix.to_string()) 
        print(" Relatório salvo em: data/relatorio_analise_avancada.txt") 

def executar_analise_completa(self): 
    """Executa análise completa."""
 
        if not self.carregar_dados(): 
        return False # Executar todas as análises self.analise_descritiva_avancada() self.analise_correlacoes_avancada() self.analise_por_localizacao() self.analise_por_tipo_imovel() self.analise_outliers_detalhada() self.analise_normalidade() self.analise_qualidade_dados() # Insights finais insights = self.insights_finais() # Salvar relatório self.salvar_relatorio_completo() 
        print(f"\n" + "="*80) 
        print(" ANÁLISE EXPLORATÓRIA AVANÇADA CONCLUÍDA!") 
        print("="*80) 
        print(f" Dataset analisado: {len(self.df):,} registros") 
        print(f" Relatório detalhado salvo") 
        print(" Pronto para modelagem com insights aprofundados") 
        print("="*80) 
        return True 

def main(): 
    """Função principal."""
 analise = AnaliseAvancada() sucesso = analise.executar_analise_completa() 
        if sucesso: 
        print("\n CONTINUAR PARA REGRESSÃO LINEAR?") resposta = input("Digite 'sim' ou 's' para prosseguir: ").lower().strip() 
        if resposta in ['sim', 's', 'yes', 'y']: 
        print(" Prosseguindo para modelagem...") 
        return True 
        else: 
        print(" Análise finalizada. Execute a modelagem quando estiver pronto.") 
        return False return False 
        if __name__ == "__main__": main()