#!/usr/bin/env python3 # -*- coding: utf-8 -*- 
    """ CRIADOR DE DATASET ÚNICO E LIMPEZA ========================== Script para: 1. Criar dataset com ~3500 registros únicos 2. Limpar arquivos antigos 3. Preparar para nova coleta """
 
import pandas as pd 
import os 
import glob 

def criar_dataset_unico(): 
    """Cria dataset com registros únicos."""
 
        print("=" * 70) 
        print(" CRIANDO DATASET ÚNICO E LIMPANDO ARQUIVOS") 
        print("=" * 70) # 1. CARREGAR DADOS CONSOLIDADOS arquivo_consolidado = "data/dados_consolidados_unicos.csv" 
        if not os.path.exists(arquivo_consolidado): 
        print(f" Arquivo não encontrado: {arquivo_consolidado}") 
        return df = pd.read_csv(arquivo_consolidado) 
        print(f" Dataset original carregado: {len(df):,} registros") # 2. CRIAR DATASET ÚNICO 
        print(f"\n CRIANDO DATASET ÚNICO...") # Usar combinações únicas de PREÇO + ÁREA (mais crítico para ML) 
        print("Removendo duplicatas por combinação preço+área...") df_unico = df.drop_duplicates(subset=['preco', 'area'], keep='first') 
        print(f"Primeira passada (preço+área): {len(df_unico):,} registros") # Segunda passada: remover duplicatas por características completas 
        print("Removendo duplicatas por características completas...") colunas_completas = ['preco', 'area', 'quartos', 'banheiros', 'localizacao', 'tipo'] df_unico = df_unico.drop_duplicates(subset=colunas_completas, keep='first') 
        print(f" Dataset único criado: {len(df_unico):,} registros") 
        print(f" Registros removidos: {len(df) - len(df_unico):,}") 
        print(f" Taxa de redução: {((len(df) - len(df_unico))/len(df))*100:.1f}%") # 3. SALVAR DATASET ÚNICO arquivo_unico = "data/dados_consolidados_unicos.csv" df_unico.to_csv(arquivo_unico, index=False) 
        print(f" Dataset único salvo em: {arquivo_unico}") # 4. MOSTRAR ESTATÍSTICAS DO DATASET ÚNICO 
        print(f"\n ESTATÍSTICAS DO DATASET ÚNICO:") 
        print(f" • Total de registros: {len(df_unico):,}") 
        print(f" • Preços únicos: {df_unico['preco'].nunique():,}") 
        print(f" • Áreas únicas: {df_unico['area'].nunique():,}") 
        print(f" • Localizações: {df_unico['localizacao'].nunique()}") 
        print(f"\n DISTRIBUIÇÃO POR LOCALIZAÇÃO:") dist_loc = df_unico['localizacao'].value_counts() 
        for local, qtd in dist_loc.items(): pct = (qtd/len(df_unico))*100 
        print(f" • {local}: {qtd:,} registros ({pct:.1f}%)") 
        print(f"\n ESTATÍSTICAS DE PREÇOS:") 
        print(f" • Preço médio: R$ {df_unico['preco'].mean():,.0f}") 
        print(f" • Preço mediano: R$ {df_unico['preco'].median():,.0f}") 
        print(f" • Faixa: R$ {df_unico['preco'].min():,} - R$ {df_unico['preco'].max():,}") 
        return df_unico 

def limpar_arquivos_antigos(): 
    """Remove arquivos antigos desnecessários."""
 
        print(f"\n LIMPANDO ARQUIVOS ANTIGOS...") # Lista de padrões de arquivos para remover padroes_remover = [ "data/dados_brutos_raspados*.csv", "data/dados_consolidados_brutos*.csv", "data/dados_consolidados_sem_duplicatas.csv", "data/dados_limpos.csv", "data/relatorio_analise.txt" ] arquivos_removidos = [] 
        for padrao in padroes_remover: arquivos = glob.glob(padrao) 
        for arquivo in arquivos: 
        try: os.remove(arquivo) arquivos_removidos.append(arquivo) 
        print(f" Removido: {os.path.basename(arquivo)}") 
        except Exception as e: 
        print(f" Erro ao remover {arquivo}: {e}") 
        if arquivos_removidos: 
        print(f" Total de arquivos removidos: {len(arquivos_removidos)}") 
        else: 
        print("ℹ Nenhum arquivo antigo encontrado para remover") 
        print(f"\n ARQUIVOS RESTANTES NA PASTA DATA:") 
        try: arquivos_restantes = os.listdir("data") 
        if arquivos_restantes: 
        for arquivo in sorted(arquivos_restantes): 
        print(f" • {arquivo}") 
        else: 
        print(" (pasta vazia)") 
        except Exception as e: 
        print(f" Erro ao listar pasta data: {e}") 

def main(): 
    """Função principal."""
 # 1. Criar dataset único df_unico = criar_dataset_unico() 
        if df_unico is not None: # 2. Limpar arquivos antigos limpar_arquivos_antigos() 
        print(f"\n" + "=" * 70) 
        print(" PROCESSO CONCLUÍDO COM SUCESSO!") 
        print("=" * 70) 
        print(f" Dataset único pronto: {len(df_unico):,} registros") 
        print(" Arquivos antigos removidos") 
        print(" Pronto para nova coleta de dados!") 
        print("=" * 70) 
        else: 
        print(" Falha ao criar dataset único") 
        if __name__ == "__main__": main()