# Análise de Preços de Imóveis - Distrito Federal# Análise Preditiva do Mercado Imobiliário - Distrito Federal



## Visão Geral![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)

![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)

Este projeto implementa um pipeline completo de Machine Learning para previsão de preços de imóveis no Distrito Federal, desenvolvido conforme os requisitos da atividade prática de análise de preços de imóveis. O projeto inclui web scraping para coleta de dados, análise exploratória, aplicação de regressão linear e visualização dos resultados.![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)

![License](https://img.shields.io/badge/License-Academic-yellow.svg)

## Resultados Principais

## Visão Geral

| Métrica | Valor |

|---------|-------|Este projeto implementa um **pipeline completo de Machine Learning** para previsão de preços de imóveis no Distrito Federal, desde a coleta de dados via web scraping até o desenvolvimento de modelos preditivos avançados.

| **Dados Coletados** | 637 registros |

| **Taxa de Retenção** | 97.6% (622 registros limpos) |###  Resultados Principais

| **Melhor Modelo** | Gradient Boosting |

| **R² Score** | 0.9904 || Métrica | Valor |

| **RMSE** | R$ 16,320 ||---------|-------|

| **MAE** | R$ 12,918 || **Dados Coletados** | 637 registros |

| **Taxa de Retenção** | 97.6% |

## Objetivos do Projeto| **Melhor Modelo** | Gradient Boosting |

| **R² Score** | 0.9904 |

1. **Web Scraping**: Coleta automatizada de dados de imóveis do Distrito Federal| **RMSE** | R$ 16,320 |

2. **Análise de Dados**: Limpeza, tratamento de valores ausentes e remoção de outliers| **Modelos Testados** | 8 algoritmos |

3. **Regressão Linear**: Treinamento de modelo para previsão de preços

4. **Visualização de Dados**: Gráficos e visualizações dos resultados## Objetivos do Projeto



## Estrutura do Projeto1. **Coleta Automatizada**: Desenvolver sistema robusto de web scraping

2. **Qualidade dos Dados**: Implementar pipeline de limpeza e validação

```3. **Análise Exploratória**: Identificar padrões e insights de mercado

projeto_academico/4. **Modelagem Preditiva**: Criar modelo preciso para previsão de preços

├── main.py                      # Pipeline principal5. **Documentação**: Gerar relatórios profissionais e reproduzíveis

├── requirements.txt             # Dependências do projeto

├── README.md                   # Documentação## Arquitetura do Projeto

├── src/                        # Módulos do sistema

│   ├── data_collector.py       # Web scraping```

│   ├── data_processor.py       # Limpeza de dados📁 Trabalho_ML_Scrapping_Imóveis/

│   ├── exploratory_analyzer.py # Análise exploratória├── � projeto_academico/         #  PROJETO PÚBLICO

│   └── predictive_modeler.py   # Modelagem│   ├── main.py                  # Pipeline principal

├── data/                       # Dados e resultados│   ├── requirements.txt         # Dependências

└── docs/                       # Documentação adicional│   ├── README.md               # Documentação acadêmica

```│   ├── 📁 src/                 # Módulos principais

│   │   ├── data_collector.py   # Sistema de coleta

## Como Executar│   │   ├── data_processor.py   # Pipeline de limpeza

│   │   ├── exploratory_analyzer.py # Análise exploratória

### Pré-requisitos│   │   └── predictive_modeler.py   # Modelagem ML

│   ├── 📁 data/               # Dados processados

```bash│   └── 📁 docs/               # Documentação adicional

# Python 3.9+│

python --version├── � desenvolvimento_privado/   #  DESENVOLVIMENTO PRIVADO

│   ├── 📁 scripts_experimentais/ # Scripts originais

# Instalar dependências│   ├── 📁 scripts_v2/          # Scripts da versão 2

pip install -r requirements.txt│   ├── � backups/             # Backups históricos

```│   ├── 📁 documentacao/        # Docs de desenvolvimento

│   └── 📁 utilitarios/         # Ferramentas auxiliares

### Execução do Pipeline Completo│

├── 📁 data/                     #  RESULTADOS COMPARTILHADOS

```bash│   ├── raw_data_v2.csv         # Dados coletados

# Navegar para o projeto acadêmico│   ├── clean_data_v2.csv       # Dados processados

cd projeto_academico│   ├── best_model_v2.pkl       # Modelo treinado

│   └── *.png                   # Visualizações

# Executar pipeline completo│

python main.py└── README.md                    # Documentação principal

``````



## Metodologia##  Como Executar



### 1. Web Scraping### 1️ Pré-requisitos

- Utilização do BeautifulSoup para extração de dados

- Coleta de informações: preço, localização, número de quartos, área, banheiros```bash

- Tratamento de dados inconsistentes e validação# Python 3.9+

python --version

### 2. Análise de Dados

- Limpeza dos dados coletados# Instalar dependências

- Tratamento de valores ausentespip install -r requirements.txt

- Remoção de outliers estatísticos```

- Análise exploratória para entender distribuições e relações

### 2️ Execução do Projeto Acadêmico

### 3. Regressão Linear

- Divisão dos dados em conjuntos de treino e teste (80%/20%)```bash

- Implementação de múltiplos algoritmos de regressão# Navegar para o projeto acadêmico

- Validação cruzada e seleção do melhor modelocd projeto_academico

- Avaliação usando R² e MSE

# Executar pipeline completo

### 4. Visualização de Dadospython main.py

- Gráficos de dispersão para relações entre variáveis```

- Histogramas para distribuições

- Gráficos de resíduos para validação do modelo### 3️ Uso Rápido (Modelo Pré-treinado)

- Visualizações geográficas por região

```python

## Algoritmos Implementadosimport joblib

import pandas as pd

1. **Linear Regression** - Modelo base de regressão linear

2. **Ridge Regression** - Regressão com regularização L2  # Carregar modelo

3. **Lasso Regression** - Regressão com regularização L1modelo = joblib.load('data/best_model_v2.pkl')

4. **Decision Tree** - Árvore de decisão para regressão

5. **Random Forest** - Ensemble de árvores de decisão# Prever preço de novo imóvel

6. **Gradient Boosting** - Algoritmo de boosting (melhor performance)novo_imovel = pd.DataFrame({

    'area': [90],

## Performance do Modelo    'quartos': [3],

    'banheiros': [2],

### Gradient Boosting (Melhor Modelo)    'preco_por_m2': [5000],

    'area_por_quarto': [30],

| Métrica | Treino | Teste | Interpretação |    # ... outras features

|---------|--------|-------|---------------|})

| **R²** | 0.9915 | 0.9904 | Explica 99.04% da variância dos preços |

| **RMSE** | R$ 15,568 | R$ 16,320 | Erro médio quadrático baixo |preco_predito = modelo.predict(novo_imovel)

| **MAE** | R$ 12,071 | R$ 12,918 | Erro absoluto médio aceitável |print(f"Preço estimado: R$ {preco_predito[0]:,.2f}")

```

## Principais Descobertas

##  Pipeline de Machine Learning

### Features Mais Importantes

###  Fluxo Completo

1. **Área (55.07%)** - Principal determinante do preço

2. **Preço por m² (42.10%)** - Influência da localização```mermaid

3. **Quartos (0.71%)** - Configuração do imóvelgraph LR

4. **Localização específica** - Premium por região    A[ Web Scraping] --> B[ Data Cleaning]

    B --> C[ EDA]

### Análise por Região    C --> D[ Modeling]

    D --> E[ Evaluation]

| Região | Preço Médio | N° de Imóveis | Preço/m² Médio |    E --> F[ Report]

|--------|-------------|---------------|----------------|```

| **Asa Sul** | R$ 639,018 | 89 | R$ 7,234/m² |

| **Sudoeste** | R$ 546,516 | 45 | R$ 6,421/m² |###  Algoritmos Implementados

| **Asa Norte** | R$ 538,645 | 112 | R$ 6,102/m² |

| **Águas Claras** | R$ 494,689 | 134 | R$ 5,892/m² |1. **Linear Regression** - Baseline linear

| **Ceilândia** | R$ 273,148 | 78 | R$ 3,156/m² |2. **Ridge Regression** - Regularização L2  

3. **Lasso Regression** - Regularização L1

### Insights de Mercado4. **Elastic Net** - Combinação L1/L2

5. **Decision Tree** - Árvore de decisão

- **Configuração mais comum**: 2 quartos (33.1% dos imóveis)6. **Random Forest** - Ensemble de árvores

- **Área média dos imóveis**: 86.6 m²7. **Gradient Boosting** -  **Melhor modelo**

- **Preço médio por m²**: R$ 5,192/m²8. **Support Vector Regression** - SVR

- **Amplitude de preços**: R$ 150.000 a R$ 1.200.000

###  Performance do Melhor Modelo

## Estrutura dos Dados

| Métrica | Treino | Teste | Interpretação |

### Variáveis Coletadas|---------|--------|-------|---------------|

| **R²** | 0.9915 | 0.9904 | Explica 99.04% da variância |

| Variável | Tipo | Descrição | Exemplo || **RMSE** | R$ 15,568 | R$ 16,320 | Erro médio baixo |

|----------|------|-----------|---------|| **MAE** | R$ 12,071 | R$ 12,918 | Erro absoluto médio |

| `preco` | Numérica | Preço do imóvel em reais | R$ 450.000 |

| `area` | Numérica | Área total em m² | 85 m² |##  Principais Descobertas

| `quartos` | Numérica | Número de quartos | 3 |

| `banheiros` | Numérica | Número de banheiros | 2 |###  Features Mais Importantes

| `localizacao` | Categórica | Região do Distrito Federal | Águas Claras |

| `tipo` | Categórica | Tipo do imóvel | Apartamento |1. **Área (55.07%)** - Principal determinante do preço

2. **Preço por m² (42.10%)** - Padrão da localização

### Features Engenheiradas3. **Quartos (0.71%)** - Configuração do imóvel

4. **Asa Sul (0.51%)** - Premium de localização

- **`preco_por_m2`**: Valor por metro quadrado5. **Área por quarto (0.34%)** - Indicador de amplitude

- **`area_por_quarto`**: Relação área/quartos

- **`categoria_preco`**: Faixas de preço (baixo, médio, alto)###  Análise Geográfica

- **`categoria_area`**: Classificação por tamanho

| Região | Preço Médio | Classificação |

## Tecnologias Utilizadas|--------|-------------|---------------|

| **Asa Sul** | R$ 639,018 |  Premium |

### Bibliotecas Principais| **Sudoeste** | R$ 546,516 |  Alto padrão |

| **Asa Norte** | R$ 538,645 |  Alto padrão |

```python| **Águas Claras** | R$ 494,689 |  Médio-alto |

# Manipulação de dados| **Ceilândia** | R$ 273,148 |  Econômico |

import pandas as pd

import numpy as np###  Insights de Mercado



# Machine Learning- **Configuração mais comum**: 2 quartos (33.1%)

from sklearn.linear_model import LinearRegression, Ridge, Lasso- **Área média**: 86.6 m²

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor- **Preço por m² médio**: R$ 5,192/m²

from sklearn.model_selection import train_test_split, cross_val_score- **Diferença regional**: 134% entre Asa Sul e Ceilândia

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

##  Estrutura dos Dados

# Visualização

import matplotlib.pyplot as plt###  Variáveis Coletadas

import seaborn as sns

| Variável | Tipo | Descrição | Exemplo |

# Web Scraping|----------|------|-----------|---------|

import requests| `preco` | Numérica | Preço do imóvel | R$ 450.000 |

from bs4 import BeautifulSoup| `area` | Numérica | Área em m² | 85 m² |

import time| `quartos` | Numérica | Número de quartos | 3 |

```| `banheiros` | Numérica | Número de banheiros | 2 |

| `localizacao` | Categórica | Região do DF | Águas Claras |

## Arquivos Gerados| `tipo` | Categórica | Tipo do imóvel | Apartamento |



### Dados###  Features Engenheiradas

- `raw_data_v2.csv` - Dados brutos coletados

- `clean_data_v2.csv` - Dados limpos e processados- **`preco_por_m2`**: Valor por metro quadrado

- `full_cleaned_data_v2.csv` - Dataset completo para análise- **`area_por_quarto`**: Amplitude média dos cômodos

- **`categoria_preco`**: Segmentação por faixas

### Modelos- **`categoria_area`**: Classificação por tamanho

- `best_model_v2.pkl` - Modelo treinado para produção

- `model_info_v2.pkl` - Metadados e informações do modelo##  Tecnologias Utilizadas



### Visualizações###  Principais Bibliotecas

- `distribuicoes_v2.png` - Histogramas das variáveis principais

- `correlacoes_v2.png` - Matriz de correlação```python

- `analise_localizacao_v2.png` - Análise geográfica# Core

- `comparacao_modelos_v2.png` - Performance dos modelosimport pandas as pd              # Manipulação de dados

- `analise_residuos_v2.png` - Análise de resíduosimport numpy as np               # Computação numérica

- `dashboard_eda_v2.png` - Dashboard da análise exploratória

# Machine Learning

### Relatóriosfrom sklearn.ensemble import RandomForestRegressor

- `insights_eda_v2.txt` - Principais descobertas da análisefrom sklearn.model_selection import train_test_split

- `relatorio_modelagem_v2.txt` - Relatório completo da modelagemfrom sklearn.metrics import mean_squared_error



## Critérios de Avaliação Atendidos# Visualização

import matplotlib.pyplot as plt

### Qualidade do Web Scrapingimport seaborn as sns

- Código eficiente e robusto para coleta de dados

- Tratamento de erros e validação de dados coletados# Web Scraping

- Coleta de 637 registros com alta taxa de sucessoimport requests

import BeautifulSoup

### Completude dos Dados```

- Taxa de retenção de 97.6% após limpeza

- Tratamento adequado de valores ausentes###  Ferramentas de Desenvolvimento

- Remoção sistemática de outliers

- **Python 3.9+** - Linguagem principal

### Análise Exploratória- **Jupyter Notebook** - Prototipagem

- Análise estatística descritiva completa- **Git** - Controle de versão

- Identificação de padrões e correlações- **VS Code** - IDE

- Insights relevantes sobre o mercado imobiliário

##  Documentação Adicional

### Modelo de Regressão

- Implementação de múltiplos algoritmos-  **[Relatório HTML Completo](relatorio_completo_ml_imoveis.html)** - Análise detalhada

- R² de 0.9904 demonstra alta precisão-  **[Análise Exploratória](data/)** - Visualizações e insights

- Validação cruzada e métricas robustas-  **[Modelos Treinados](data/)** - Arquivos .pkl dos modelos

-  **[Dados Limpos](data/clean_data_v2.csv)** - Dataset final

### Visualizações

- Gráficos claros e informativos##  Próximos Passos

- Visualizações técnicas e profissionais

- Dashboard completo dos resultados###  Melhorias Futuras



## Contato1. **Análise Temporal** - Incluir séries históricas

2. **Geocodificação** - Coordenadas geográficas precisas

**Desenvolvido por**: Caio Thomas Bandeira  3. **API REST** - Endpoint para predições

**Email**: caiothomas16@gmail.com  4. **Dashboard** - Interface interativa

**GitHub**: caiothomasbandeira  5. **A/B Testing** - Validação com dados reais

**LinkedIn**: [Caio Thomas Bandeira](https://www.linkedin.com/in/caiothomasbandeira)

### Manutenção

## Licença

- **Atualização mensal** dos dados

Este projeto foi desenvolvido para fins acadêmicos como parte da atividade prática de análise de preços de imóveis. O código é disponibilizado para fins educacionais.- **Re-treinamento trimestral** dos modelos

- **Monitoramento** de performance

---- **Validação** com dados externos



**Análise de Preços de Imóveis - Distrito Federal**  ## Como Contribuir

*Projeto acadêmico desenvolvido com Python, scikit-learn e técnicas de Machine Learning*

```bash

**637 imóveis analisados • R² = 0.9904 • RMSE = R$ 16,320**# 1. Fork do projeto
git clone https://github.com/seu-usuario/projeto-ml-imoveis

# 2. Criar branch para feature
git checkout -b feature/nova-funcionalidade

# 3. Commit das mudanças
git commit -m "Adiciona nova funcionalidade"

# 4. Push da branch
git push origin feature/nova-funcionalidade

# 5. Abrir Pull Request
```

## Contato

-  **Email**: [caiothomas16@gmail.com]
-  **GitHub**: [caiothomasbandeira]
-  **LinkedIn**: [[link](https://www.linkedin.com/in/caiothomasbandeira)]

##  Licença

Este projeto está sob licença acadêmica. Desenvolvido como projeto educacional para demonstração de técnicas de Machine Learning aplicadas ao mercado imobiliário.

---

<div align="center">

**Machine Learning para Mercado Imobiliário**

Desenvolvido com dedicação usando Python e Scikit-learn

 **637 imóveis analisados** •  **R² = 0.9904** •  **RMSE = R$ 16,320**

</div>