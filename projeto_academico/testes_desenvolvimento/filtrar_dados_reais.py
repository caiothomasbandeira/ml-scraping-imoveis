#!/usr/bin/env python3 # -*- coding: utf-8 -*- 
    """ FILTRADOR DE DADOS REAIS ======================== Script para remover todos os dados simulados/gerados e manter apenas dados extraídos de sites reais. """
 
import pandas as pd 

def filtrar_dados_reais(): 
    """Filtra apenas dados reais dos sites."""
 
        print("=" * 70) 
        print(" FILTRANDO APENAS DADOS REAIS DE SITES") 
        print("=" * 70) # Carregar novos dados 
        try: df_novos = pd.read_csv("data/dados_brutos_raspados.csv") 
        print(f" Novos dados carregados: {len(df_novos):,} registros") 
        except FileNotFoundError: 
        print(" Arquivo dados_brutos_raspados.csv não encontrado") 
        return # Identificar fontes simuladas fontes_simuladas = [ 'Simulado', 'Gerado_Extra', 'Educacional', 'Variacao_Real', 'Militar_Real' ] # Filtrar fontes que contêm "_Simulado" ou são simuladas 
        print(f"\n ANALISANDO FONTES DOS DADOS:") 
        print("Distribuição por fonte:") 
        for fonte, qtd in df_novos['fonte'].value_counts().items(): status = " SIMULADO" 
        if any(sim in fonte 
        for sim in fontes_simuladas) else " REAL" 
        print(f" • {fonte}: {qtd} registros - {status}") # Remover dados simulados 
        print(f"\n REMOVENDO DADOS SIMULADOS...") # Filtrar apenas dados reais mask_real = ~df_novos['fonte'].str.contains('_Simulado|Simulado|Gerado_Extra|Educacional|Variacao_Real|Militar_Real', na=False) df_reais_novos = df_novos[mask_real].copy() removidos = len(df_novos) - len(df_reais_novos) 
        print(f" Dados simulados removidos: {removidos}") 
        print(f" Dados reais restantes: {len(df_reais_novos):,}") # Mostrar distribuição dos dados reais 
        print(f"\n FONTES DE DADOS REAIS:") 
        for fonte, qtd in df_reais_novos['fonte'].value_counts().items(): 
        print(f" • {fonte}: {qtd} registros") # Carregar dados únicos antigos 
        try: df_antigos = pd.read_csv("data/dados_consolidados_unicos.csv") 
        print(f"\n Dados únicos antigos carregados: {len(df_antigos):,} registros") # Filtrar dados antigos para manter apenas OLX (dados reais) df_antigos_reais = df_antigos[df_antigos['fonte'].str.contains('OLX', na=False)].copy() 
        print(f" Dados antigos reais (OLX): {len(df_antigos_reais):,} registros") 
        except FileNotFoundError: 
        print(" Arquivo dados_consolidados_unicos.csv não encontrado") df_antigos_reais = pd.DataFrame() # Combinar datasets reais 
        if not df_antigos_reais.empty: 
        print(f"\n COMBINANDO DATASETS REAIS...") # Garantir que as colunas sejam consistentes colunas_comuns = list(set(df_reais_novos.columns) & set(df_antigos_reais.columns)) df_antigos_select = df_antigos_reais[colunas_comuns].copy() df_novos_select = df_reais_novos[colunas_comuns].copy() # Combinar df_final = pd.concat([df_antigos_select, df_novos_select], ignore_index=True) 
        print(f" Dataset combinado: {len(df_final):,} registros") # Remover duplicatas por combinação preço+área 
        print(f" Removendo duplicatas por preço+área...") df_final_unico = df_final.drop_duplicates(subset=['preco', 'area'], keep='first') duplicatas_removidas = len(df_final) - len(df_final_unico) 
        print(f" Duplicatas removidas: {duplicatas_removidas}") 
        print(f" Dataset final único: {len(df_final_unico):,} registros") 
        else: # Apenas dados novos df_final_unico = df_reais_novos.drop_duplicates(subset=['preco', 'area'], keep='first') 
        print(f" Dataset final (apenas novos): {len(df_final_unico):,} registros") # Estatísticas finais 
        print(f"\n ESTATÍSTICAS DO DATASET REAL FINAL:") 
        print(f" • Total de registros: {len(df_final_unico):,}") 
        print(f" • Preços únicos: {df_final_unico['preco'].nunique():,}") 
        print(f" • Áreas únicas: {df_final_unico['area'].nunique():,}") 
        print(f" • Preço médio: R$ {df_final_unico['preco'].mean():,.0f}") 
        print(f" • Preço mediano: R$ {df_final_unico['preco'].median():,.0f}") 
        print(f" • Faixa de preços: R$ {df_final_unico['preco'].min():,} - R$ {df_final_unico['preco'].max():,}") 
        print(f"\n DISTRIBUIÇÃO POR FONTE REAL:") 
        for fonte, qtd in df_final_unico['fonte'].value_counts().items(): pct = (qtd/len(df_final_unico))*100 
        print(f" • {fonte}: {qtd:,} registros ({pct:.1f}%)") 
        if 'localizacao' in df_final_unico.columns: 
        print(f"\n DISTRIBUIÇÃO POR LOCALIZAÇÃO:") 
        for local, qtd in df_final_unico['localizacao'].value_counts().head(10).items(): pct = (qtd/len(df_final_unico))*100 
        print(f" • {local}: {qtd:,} registros ({pct:.1f}%)") # Salvar dataset final arquivo_final = "data/dataset_final_real.csv" df_final_unico.to_csv(arquivo_final, index=False) 
        print(f"\n Dataset final salvo em: {arquivo_final}") # Limpar arquivos temporários 
        print(f"\n LIMPANDO ARQUIVOS TEMPORÁRIOS...") 
import os 
        try: os.remove("data/dados_brutos_raspados.csv") 
        print(" Removido: dados_brutos_raspados.csv") 
        except: pass 
        print(f"\n" + "=" * 70) 
        print(" FILTRAGEM CONCLUÍDA - APENAS DADOS REAIS!") 
        print("=" * 70) 
        print(f" Dataset final: {len(df_final_unico):,} registros reais únicos") 
        print(f" Arquivo: {arquivo_final}") 
        print(" Pronto para análise e modelagem!") 
        print("=" * 70) 
        return df_final_unico 
        if __name__ == "__main__": filtrar_dados_reais()