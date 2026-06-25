# TCC-2-
APLICAÇÃO DE INTELIGÊNCIA ARTIFICIAL NA ENGENHARIA DE SOFTWARE PARA ANÁLISE DE CONFIABILIDADE E MANUTENÇÃO PREDITIVA

Este repositório contém um experimento com o dataset NASA CMAPSS FD001 para manutenção preditiva.

Melhorias aplicadas neste branch:
- Seeds para numpy, random e tensorflow para melhorar reprodutibilidade.
- Download dos dados com retries e timeout usando requests.
- Cálculo do RUL do conjunto de teste mapeado por unidade (mais robusto).
- Split por unidade para validação no treinamento da LSTM (evita vazamento entre treino/val).
- Salvamento de artefatos: resultados.json, rf_model.joblib, lstm_cls.h5, lstm_reg.h5.
- Arquivo requirements.txt incluso.

Como executar
1. Criar e ativar um ambiente virtual (recomendado):

   python -m venv .venv
   source .venv/bin/activate   # Linux/Mac
   .venv\Scripts\activate    # Windows (PowerShell)

2. Instalar dependências:

   pip install -r requirements.txt

3. Executar o experimento:

   python experimento_nasa_cmapss.py

Os dados serão baixados automaticamente para a pasta `cmapss_data/` e os artefatos serão salvos no diretório atual.
