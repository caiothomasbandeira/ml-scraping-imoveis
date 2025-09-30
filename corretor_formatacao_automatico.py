#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCRIPT DE ENGENHARIA REVERSA - CORREÇÃO AUTOMÁTICA DE FORMATAÇÃO
================================================================

Script inteligente que corrige automaticamente a formatação de arquivos Python
corrompidos durante a limpeza de emojis.

Padrão identificado:
- Quebras de linha removidas
- Código comprimido em uma linha
- Comentários e strings preservados

Estratégia de correção:
1. Identificar padrões de código Python
2. Inserir quebras de linha nos pontos corretos
3. Preservar indentação e estrutura
"""

import os
import re
from pathlib import Path


class CorretorFormatacaoPython:
    def __init__(self):
        self.patterns = [
            # Imports
            (r'(\bimport\s+\w+)', r'\n\1'),
            (r'(\bfrom\s+[\w.]+\s+import\s+[^#\n]+)', r'\n\1'),
            
            # Definições de função e classe
            (r'(\bdef\s+\w+\([^)]*\):)', r'\n\n\1'),
            (r'(\bclass\s+\w+[^:]*:)', r'\n\n\1'),
            
            # Blocos de controle
            (r'(\bif\s+[^:]+:)', r'\n    \1'),
            (r'(\belse:)', r'\n    \1'),
            (r'(\belif\s+[^:]+:)', r'\n    \1'),
            (r'(\bfor\s+[^:]+:)', r'\n    \1'),
            (r'(\bwhile\s+[^:]+:)', r'\n    \1'),
            (r'(\btry:)', r'\n    \1'),
            (r'(\bexcept[^:]*:)', r'\n    \1'),
            (r'(\bfinally:)', r'\n    \1'),
            (r'(\bwith\s+[^:]+:)', r'\n    \1'),
            
            # Comentários de seção
            (r'(\s*#\s*[A-Z][^#\n]*)', r'\n        \1'),
            
            # Print statements
            (r'(\bprint\s*\([^)]*\))', r'\n        \1'),
            
            # Assignments principais
            (r'(\b\w+\s*=\s*[^#\n]+)', r'\n        \1'),
            
            # Return statements
            (r'(\breturn\s+[^#\n]*)', r'\n        \1'),
            
            # Docstrings
            (r'("""[^"]*""")', r'\n    \1\n'),
            (r"('''[^']*''')", r'\n    \1\n'),
        ]

    def corrigir_arquivo(self, caminho_arquivo):
        """Corrige um arquivo Python específico."""
        try:
            with open(caminho_arquivo, 'r', encoding='utf-8') as f:
                conteudo = f.read()
            
            # Backup
            backup_path = f"{caminho_arquivo}.backup_formatacao"
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(conteudo)
            
            # Aplicar correções sequenciais
            conteudo_corrigido = self.aplicar_correcoes(conteudo)
            
            # Salvar versão corrigida
            with open(caminho_arquivo, 'w', encoding='utf-8') as f:
                f.write(conteudo_corrigido)
            
            print(f"✅ {caminho_arquivo} corrigido")
            return True
            
        except Exception as e:
            print(f"❌ Erro em {caminho_arquivo}: {e}")
            return False
    
    def aplicar_correcoes(self, conteudo):
        """Aplica as correções de formatação ao conteudo."""
        # Primeira passada: quebras de linha básicas
        resultado = conteudo
        
        # Separar imports
        resultado = re.sub(r'(\bimport\s+\w+)', r'\n\1', resultado)
        resultado = re.sub(r'(\bfrom\s+[\w.]+\s+import\s+[^#\n]+)', r'\n\1', resultado)
        
        # Separar definições de classe e função
        resultado = re.sub(r'(\bclass\s+\w+[^:]*:)', r'\n\n\1', resultado)
        resultado = re.sub(r'(\bdef\s+\w+\([^)]*\):)', r'\n\n\1', resultado)
        
        # Separar docstrings
        resultado = re.sub(r'(""".+?""")', r'\n    \1\n', resultado, flags=re.DOTALL)
        
        # Separar prints e principais statements
        resultado = re.sub(r'(\bprint\s*\([^)]+\))', r'\n        \1', resultado)
        resultado = re.sub(r'(\breturn\s+[^#\n]*)', r'\n        \1', resultado)
        
        # Separar estruturas de controle
        resultado = re.sub(r'(\bif\s+[^:]+:)', r'\n        \1', resultado)
        resultado = re.sub(r'(\belse:)', r'\n        \1', resultado)
        resultado = re.sub(r'(\bfor\s+[^:]+:)', r'\n        \1', resultado)
        resultado = re.sub(r'(\btry:)', r'\n        \1', resultado)
        resultado = re.sub(r'(\bexcept[^:]*:)', r'\n        \1', resultado)
        
        # Limpar múltiplas quebras de linha
        resultado = re.sub(r'\n{3,}', '\n\n', resultado)
        resultado = re.sub(r'^\n+', '', resultado)
        
        return resultado
    
    def corrigir_pasta(self, pasta_path):
        """Corrige todos os arquivos Python de uma pasta."""
        pasta = Path(pasta_path)
        arquivos_python = list(pasta.rglob("*.py"))
        
        print(f"🔧 Corrigindo {len(arquivos_python)} arquivos Python em {pasta_path}")
        print("=" * 60)
        
        sucessos = 0
        for arquivo in arquivos_python:
            if self.corrigir_arquivo(arquivo):
                sucessos += 1
        
        print("=" * 60)
        print(f"✅ {sucessos}/{len(arquivos_python)} arquivos corrigidos com sucesso!")
        
        return sucessos


def main():
    """Executa a correção automática."""
    print("🚀 CORRETOR AUTOMÁTICO DE FORMATAÇÃO PYTHON")
    print("=" * 60)
    
    # Pasta do projeto acadêmico  
    pasta_projeto = Path("c:/Users/maril/OneDrive/Documentos/Trabalho_ML_Scrapping_Imóveis/projeto_academico")
    
    if not pasta_projeto.exists():
        print(f"❌ Pasta não encontrada: {pasta_projeto}")
        return
    
    corretor = CorretorFormatacaoPython()
    sucessos = corretor.corrigir_pasta(pasta_projeto)
    
    print(f"\n🎉 CORREÇÃO CONCLUÍDA!")
    print(f"📊 Arquivos corrigidos: {sucessos}")
    print(f"📂 Backups salvos com extensão: .backup_formatacao")
    
    return sucessos


if __name__ == "__main__":
    main()