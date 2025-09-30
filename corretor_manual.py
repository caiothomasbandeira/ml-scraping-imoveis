#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CORRETOR MANUAL ESPECÍFICO - FORMATAÇÃO PYTHON
==============================================
Script especializado para corrigir os arquivos específicos corrompidos.
"""

import os
import re
from pathlib import Path


def corrigir_arquivo_python_manual(caminho_arquivo):
    """Correção manual específica para o padrão identificado."""
    
    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        conteudo = f.read()
    
    # Backup
    backup_path = f"{caminho_arquivo}.backup_manual"
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(conteudo)
    
    # Padrões específicos para correção
    correcoes = [
        # Shebang e encoding
        (r'#!/usr/bin/env python3 # -\*- coding: utf-8 -\*-', '#!/usr/bin/env python3\n# -*- coding: utf-8 -*-'),
        
        # Imports - separar por linha
        (r' import ([a-zA-Z_][a-zA-Z0-9_]*)', r'\nimport \1'),
        (r' from ([a-zA-Z_][a-zA-Z0-9_.]*)', r'\nfrom \1'),
        
        # Definições de classe
        (r' class ([a-zA-Z_][a-zA-Z0-9_]*)', r'\n\nclass \1'),
        
        # Definições de função
        (r' def ([a-zA-Z_][a-zA-Z0-9_]*)', r'\n\n    def \1'),
        
        # Estruturas de controle
        (r' if ([^:]+:)', r'\n        if \1'),
        (r' else:', r'\n        else:'),
        (r' elif ([^:]+:)', r'\n        elif \1'),
        (r' for ([^:]+:)', r'\n        for \1'),
        (r' while ([^:]+:)', r'\n        while \1'),
        (r' try:', r'\n        try:'),
        (r' except([^:]*:)', r'\n        except\1'),
        (r' finally:', r'\n        finally:'),
        (r' with ([^:]+:)', r'\n        with \1'),
        
        # Print statements
        (r' print\(', r'\n        print('),
        
        # Return statements  
        (r' return ', r'\n        return '),
        
        # Comments e docstrings
        (r' """', r'\n    """'),
        (r'""" ', r'"""\n'),
        (r" '''", r"\n    '''"),
        (r"''' ", r"'''\n"),
        
        # Assignments importantes
        (r' ([a-zA-Z_][a-zA-Z0-9_]*) = ', r'\n        \1 = '),
        
        # Limpar espaços excessivos
        (r'\n\s*\n\s*\n', r'\n\n'),
        (r'^\n+', r''),
    ]
    
    # Aplicar correções
    resultado = conteudo
    for padrao, substituicao in correcoes:
        resultado = re.sub(padrao, substituicao, resultado)
    
    # Salvar arquivo corrigido
    with open(caminho_arquivo, 'w', encoding='utf-8') as f:
        f.write(resultado)
    
    print(f"✅ {os.path.basename(caminho_arquivo)} corrigido")


def main():
    """Corrige os arquivos principais manualmente."""
    print("🔧 CORREÇÃO MANUAL ESPECÍFICA")
    print("=" * 40)
    
    # Arquivos principais para correção
    base_path = Path("c:/Users/maril/OneDrive/Documentos/Trabalho_ML_Scrapping_Imóveis/projeto_academico")
    
    arquivos_prioritarios = [
        base_path / "main.py",
        base_path / "modelo_final.py", 
        base_path / "scripts_principais" / "web_scraper.py",
        base_path / "scripts_principais" / "data_analyzer.py",
        base_path / "scripts_principais" / "linear_regression.py",
        base_path / "scripts_principais" / "visualizations.py",
    ]
    
    for arquivo in arquivos_prioritarios:
        if arquivo.exists():
            corrigir_arquivo_python_manual(arquivo)
        else:
            print(f"⚠️  {arquivo} não encontrado")
    
    print("=" * 40)
    print("✅ Correção manual concluída!")


if __name__ == "__main__":
    main()