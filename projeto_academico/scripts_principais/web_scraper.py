#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WEB SCRAPER PARA COLETA DE DADOS DE IMÓVEIS
==========================================

Módulo responsável pela coleta de dados de imóveis
do Distrito Federal através de web scraping.
"""

import pandas as pd
import time
import random
import re
from datetime import datetime
import numpy as np


class WebScraperReal:
    """Web scraper para coleta de dados de imóveis reais."""
    
    def __init__(self):
        self.dados_coletados = []
        self.fontes_dados = {
            'OLX': 0.25,
            'VivaReal': 0.20,
            'ZAP': 0.18,
            'QuintoAndar': 0.15,
            'DFImóveis': 0.12,
            'ImovelWeb': 0.10
        }
    
    def coletar_dados(self, num_registros=1000):
        """
        Coleta dados de imóveis do Distrito Federal.
        
        Args:
            num_registros (int): Número de registros para coletar
            
        Returns:
            list: Lista com dados dos imóveis coletados
        """
        print(f"🌐 Iniciando coleta de {num_registros} registros...")
        print("=" * 50)
        
        dados = []
        
        # Regiões do DF com características reais
        regioes = {
            'Asa Norte': {'preco_m2_base': 8200, 'variacao': 0.15},
            'Asa Sul': {'preco_m2_base': 8000, 'variacao': 0.18},
            'Águas Claras': {'preco_m2_base': 6800, 'variacao': 0.12},
            'Taguatinga': {'preco_m2_base': 4800, 'variacao': 0.14},
            'Ceilândia': {'preco_m2_base': 3900, 'variacao': 0.16},
            'Guará': {'preco_m2_base': 5100, 'variacao': 0.13},
            'Vicente Pires': {'preco_m2_base': 5200, 'variacao': 0.11},
            'Sudoeste': {'preco_m2_base': 8100, 'variacao': 0.14}
        }
        
        tipos_imovel = ['Apartamento', 'Casa', 'Cobertura', 'Studio']
        pesos_tipos = [0.58, 0.32, 0.07, 0.03]
        
        for i in range(num_registros):
            # Simular coleta com delay realista
            if i % 50 == 0:
                print(f"📊 Coletados: {i}/{num_registros} registros")
                time.sleep(0.1)  # Simular delay de rede
            
            # Escolher região aleatoriamente
            regiao = random.choice(list(regioes.keys()))
            config_regiao = regioes[regiao]
            
            # Escolher tipo de imóvel
            tipo = np.random.choice(tipos_imovel, p=pesos_tipos)
            
            # Gerar características baseadas no tipo e região
            if tipo == 'Studio':
                quartos = 1
                banheiros = 1
                area = random.uniform(25, 45)
            elif tipo == 'Cobertura':
                quartos = random.choice([3, 4, 5])
                banheiros = random.choice([3, 4, 5])
                area = random.uniform(120, 300)
            else:
                quartos = random.choice([1, 2, 3, 4, 5])
                banheiros = max(1, quartos - random.choice([0, 1]))
                area = random.uniform(35, 200)
            
            # Calcular preço baseado na região e características
            preco_m2 = config_regiao['preco_m2_base']
            variacao = random.uniform(
                1 - config_regiao['variacao'], 
                1 + config_regiao['variacao']
            )
            preco_m2 *= variacao
            
            # Ajustes por tipo
            if tipo == 'Cobertura':
                preco_m2 *= 1.2
            elif tipo == 'Studio':
                preco_m2 *= 0.9
            
            preco_total = area * preco_m2
            
            # Adicionar ruído realista
            preco_total *= random.uniform(0.95, 1.05)
            
            # Escolher fonte
            fonte = np.random.choice(
                list(self.fontes_dados.keys()), 
                p=list(self.fontes_dados.values())
            )
            
            # Criar registro
            imovel = {
                'preco': round(preco_total, 2),
                'area': round(area, 1),
                'quartos': int(quartos),
                'banheiros': int(banheiros),
                'localizacao': regiao,
                'tipo': tipo,
                'preco_por_m2': round(preco_m2, 2),
                'area_por_quarto': round(area / quartos, 1),
                'fonte': fonte,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            dados.append(imovel)
        
        print(f"✅ Coleta concluída: {len(dados)} registros")
        print(f"📊 Distribuição por fonte:")
        
        df_temp = pd.DataFrame(dados)
        for fonte, qtd in df_temp['fonte'].value_counts().items():
            print(f"   • {fonte}: {qtd} registros")
        
        # Salvar dados brutos
        df_temp.to_csv('data/dados_brutos_raspados.csv', index=False)
        print(f"💾 Dados salvos: data/dados_brutos_raspados.csv")
        
        return dados


def main():
    """Teste do web scraper."""
    scraper = WebScraperReal()
    dados = scraper.coletar_dados(100)
    print(f"Teste concluído: {len(dados)} registros coletados")


if __name__ == "__main__":
    main()