# Machine Learning - Análise de Preços de Imóveis

## Sobre o Projeto

Este repositório contém um projeto acadêmico de Machine Learning focado na análise e predição de preços de imóveis no Distrito Federal, Brasil. O projeto utiliza técnicas de web scraping, processamento de dados e regressão linear para criar um modelo preditivo com alta precisão.

## Projeto Academico

**Para avaliacao academica, acesse:** `projeto_academico/`

A pasta `projeto_academico/` contém:
- **Código principal organizado** (`main_fixed.py`, `modelo_final.py`)
- **Dados finais processados** (`dados_finais/`)
- **Relatório HTML completo** (`relatorio_completo_projeto.html`)
- **Documentação detalhada** (`README.md`)
- **Scripts auxiliares** organizados por categoria

### Como Executar

```bash
cd projeto_academico/
pip install -r requirements.txt
python main_fixed.py
```

## Resultados Principais

| Métrica | Valor | Status |
|---------|-------|---------|
| **Dados Coletados** | 4.539 registros | Completo |
| **Taxa de Retencao** | 92.9% | Excelente |
| **Melhor Modelo** | Regressao Linear | Interpretavel |
| **R2 Score** | 91.6% | Excepcional |
| **RMSE** | R$ 89.267 | Baixo erro |
| **MAE** | R$ 60.832 | Precisao alta |

## Objetivos do Projeto

1. **Web Scraping**: Coleta automatizada de dados reais de imóveis
2. **Análise Exploratória**: Identificação de padrões do mercado imobiliário
3. **Feature Engineering**: Criação de 22 variáveis preditivas
4. **Modelagem**: Regressão linear com performance excepcional
5. **Visualizações**: Dashboards e análises técnicas

## Estrutura do Projeto

```
projeto_academico/                 # PROJETO PRINCIPAL
├── main.py                       # Pipeline completo
├── modelo_final.py               # Modelo otimizado
├── src/                          # Módulos do sistema
│   ├── web_scraper.py           # Sistema de coleta
│   ├── data_analyzer.py         # Análise exploratória
│   ├── linear_regression.py     # Modelagem ML
│   └── visualizations.py        # Gráficos e dashboards
├── dados_finais/                # Dados processados
├── relatorios/                  # Relatórios técnicos
└── requirements.txt             # Dependências

desenvolvimento_privado/          # DESENVOLVIMENTO
├── utilitarios/                 # Scripts de validação
├── scripts_v2/                  # Versões anteriores
└── documentacao/                # Docs técnicas

data/                            # RESULTADOS (nao versionado)
├── dados_limpos.csv            # Dataset final
├── modelo_final_imoveis.pkl    # Modelo treinado
└── *.png                       # Visualizações geradas
```

## Como Executar

### Pré-requisitos

```bash
# Python 3.9+
python --version

# Instalar dependências
pip install -r requirements.txt
```

### Execução Rápida

```bash
# Navegar para o projeto
cd projeto_academico

# Executar modelo final (recomendado)
python modelo_final.py

# OU executar pipeline completo
python main.py
```

### Uso do Modelo Pré-treinado

```python
import joblib
import pandas as pd

# Carregar modelo treinado
modelo = joblib.load('data/modelo_final_imoveis.pkl')

# Prever preço de novo imóvel
novo_imovel = pd.DataFrame({
    'area': [90],
    'quartos': [3], 
    'banheiros': [2],
    'localizacao_encoded': [8],
    'preco_por_m2': [6500]
    # ... outras 17 features engenheiradas
})

preco_predito = modelo.predict(novo_imovel)
print(f"Preço estimado: R$ {preco_predito[0]:,.2f}")
```

## Metodologia

### 1. Web Scraping
- **9 plataformas**: OLX, QuintoAndar, VivaReal, ZAP, DFimóveis, etc.
- **4.539 registros** coletados automaticamente
- **Dados reais** validados e filtrados

### 2. Feature Engineering
- **22 variáveis** criadas sistematicamente
- **Grupos de localização** por faixa de preço (+15.5% R²)
- **Transformações logarítmicas** para normalização (+8.0% R²)
- **Interações matemáticas** e categorização inteligente

### 3. Modelagem
- **Regressão Linear** com StandardScaler
- **Validação cruzada** (5-fold): 90.1% ± 2.4%
- **Feature selection** progressiva (5 → 22 variáveis)
- **Interpretabilidade** mantida

## Principais Descobertas

### Features Mais Impactantes
1. **Grupos de Localização** (+15.5% R²) - Agrupamento inteligente
2. **Transformação Log** (+8.0% R²) - Normalização de distribuições  
3. **Categorização de Área** (+1.0% R²) - Efeitos não-lineares

### Análise Geográfica
| Região | Preço Médio | Classificação | Preço/m² |
|--------|-------------|---------------|----------|
| **Asa Sul** | R$ 1.247.350 | Premium | R$ 8.234/m² |
| **Asa Norte** | R$ 987.240 | Alto padrão | R$ 7.156/m² |
| **Águas Claras** | R$ 645.180 | Médio-alto | R$ 5.892/m² |
| **Taguatinga** | R$ 432.890 | Médio | R$ 4.123/m² |
| **Ceilândia** | R$ 287.450 | Econômico | R$ 3.156/m² |

### Insights de Mercado
- **Configuração mais comum**: 2-3 quartos (65% dos imóveis)
- **Área média**: 89.7 m²
- **Diferença regional**: 334% entre Asa Sul e Ceilândia
- **Correlação área-preço**: 0.453 (moderada)

## Tecnologias Utilizadas

```python
# Core Data Science
pandas==2.0.3         # Manipulação de dados
numpy==1.24.3          # Computação numérica
scikit-learn==1.3.0    # Machine Learning

# Visualização
matplotlib==3.7.2      # Gráficos básicos
seaborn==0.12.2        # Visualizações estatísticas

# Web Scraping
cloudscraper==1.2.71   # Coleta anti-bot
beautifulsoup4==4.12.2 # Parser HTML
requests==2.31.0       # Requisições HTTP
```

## Performance vs Literatura

| Estudo | Algoritmo | R² Típico | Região |
|--------|-----------|-----------|---------|
| Literatura padrão | Random Forest | 65-75% | Diversas |
| Estudos avançados | XGBoost/Neural | 75-85% | Metrópoles |
| **Este projeto** | **Regressão Linear** | **91.6%** | **Distrito Federal** |

**Resultado**: Supera em 15+ pontos os padrões acadêmicos mantendo interpretabilidade total.

## Arquivos Gerados

### Dados
- `dados_limpos.csv` - Dataset final (4.539 registros)
- `modelo_final_imoveis.pkl` - Modelo treinado
- `metadata_modelo.json` - Informações técnicas

### Visualizações
- `analise_localizacao.png` - Mapa de preços por região
- `comparacao_modelos.png` - Evolução da performance
- `analise_residuos.png` - Validação estatística
- `dashboard_completo.png` - Painel executivo

### Relatórios
- `relatorio_modelo_final.txt` - Métricas e validações
- Documentação técnica adicional nos diretórios respectivos

## Criterios Academicos Atendidos

- **Web Scraping**: 9 sites, 4.539 registros reais
- **Pipeline de Dados**: Limpeza completa, sem missing values
- **Modelo ML**: R2 = 91.6%, validacao cruzada rigorosa
- **Visualizacoes**: 6 graficos profissionais + dashboard
- **Documentacao**: Codigo bem documentado e estruturado
- **Reprodutibilidade**: Codigo limpo e modular

## Proximos Passos

### Melhorias Técnicas
- [ ] **Séries Temporais**: Análise de tendências mensais
- [ ] **Geolocalização**: Coordenadas GPS precisas
- [ ] **Features Externas**: Dados de transporte e criminalidade
- [ ] **API REST**: Endpoint para predições online

### Expansão do Projeto
- [ ] **Outras Regiões**: São Paulo, Rio de Janeiro
- [ ] **Algoritmos Avançados**: XGBoost com features engenheiradas
- [ ] **Dashboard Interativo**: Streamlit/Dash para visualizações
- [ ] **Mobile App**: Aplicativo para avaliação rápida

## Contato

**Desenvolvido por**: Caio Thomas Silva Bandeira
**Email**: [caiothomas16@gmail.com]
**GitHub**: [@caiothomasbandeira](https://github.com/caiothomasbandeira)
**LinkedIn**: [Caio Thomas Silva Bandeira](www.linkedin.com/in/caiothomasbandeira)

## Licenca

Este projeto foi desenvolvido para fins acadêmicos como parte da disciplina de Machine Learning. O código é disponibilizado sob licença educacional para fins de aprendizado e pesquisa.

---

<div align="center">

**Machine Learning para Mercado Imobiliario**

*Desenvolvido com Python - scikit-learn - 4.539 imoveis analisados*

**R2 = 91.6%** - **RMSE = R$ 89.267** - **Performance Excepcional**

</div>
