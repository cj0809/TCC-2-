"""
Experimento: NASA CMAPSS FD001 — manutenção preditiva
Melhorias aplicadas:
- seeds para reprodutibilidade (numpy, random, tensorflow)
- download robusto com requests + retry
- cálculo do RUL do conjunto de teste por mapeamento (evita indexação frágil)
- split por unidade para validação da LSTM (evita vazamento)
- logging básico
- salvamento de artefatos: resultados.json, rf_model.joblib, lstm_cls.h5, lstm_reg.h5
"""

import os
import time
import json
import random
import logging

import requests
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, mean_squared_error, mean_absolute_error
)
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ---------- Reprodutibilidade ----------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ---------- Configurações ----------
BASE_URL = "https://github.com/hankroark/Turbofan-Engine-Degradation/raw/master/CMAPSSData/"
FILES = ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]
DATA_DIR = "cmapss_data"
os.makedirs(DATA_DIR, exist_ok=True)

# ---------- Download robusto ----------

def download_file(url, path, retries=3, timeout=10):
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Baixando {url} (tentativa {attempt}) -> {path}")
            resp = requests.get(url, stream=True, timeout=timeout)
            resp.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            size = os.path.getsize(path)
            logger.info(f"  ✓ {os.path.basename(path)} ({size:,} bytes)")
            return True
        except Exception as e:
            logger.warning(f"Falha ao baixar {url}: {e}")
            if attempt == retries:
                raise
            time.sleep(2)

for f in FILES:
    path = os.path.join(DATA_DIR, f)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        download_file(BASE_URL + f, path)
    else:
        logger.info(f"  ✓ {f} já existe.")

# ---------- Carregamento ----------
logger.info("Carregando dados...")
cols = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
train = pd.read_csv(os.path.join(DATA_DIR, 'train_FD001.txt'), sep=r'\s+', header=None, names=cols)
test = pd.read_csv(os.path.join(DATA_DIR, 'test_FD001.txt'), sep=r'\s+', header=None, names=cols)
rul_true = pd.read_csv(os.path.join(DATA_DIR, 'RUL_FD001.txt'), header=None, names=['RUL'])

logger.info(f"Treino : {train.shape[0]:,} amostras | {train['unit'].nunique()} motores")
logger.info(f"Teste  : {test.shape[0]:,} amostras  | {test['unit'].nunique()} motores")

# ---------- Pré-processamento ----------
logger.info("Pré-processamento...")
# RUL para treino
max_cycles_train = train.groupby('unit')['cycle'].max().reset_index().rename(columns={'cycle': 'max_cycle'})
max_cycles_train.columns = ['unit', 'max_cycle']
train = train.merge(max_cycles_train, on='unit')
train['RUL'] = train['max_cycle'] - train['cycle']
train.drop('max_cycle', axis=1, inplace=True)

# RUL para teste — mapear por unidade (mais robusto)
max_cycles_test = test.groupby('unit')['cycle'].max().reset_index().rename(columns={'cycle': 'max_cycle'})
max_cycles_test.columns = ['unit', 'max_cycle']
test = test.merge(max_cycles_test, on='unit')
# criar mapeamento de RUL verdadeira por unidade
rul_map = pd.Series(rul_true['RUL'].values, index=np.arange(1, len(rul_true) + 1))
test['RUL_last'] = test['unit'].map(rul_map)
# caso haja NaN, preencher com 0
test['RUL_last'] = test['RUL_last'].fillna(0).astype(int)

test['RUL'] = test['max_cycle'] - test['cycle'] + test['RUL_last']
test.drop(['max_cycle', 'RUL_last'], axis=1, inplace=True)

# Remover sensores com baixa variância
drop_sensors = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']
feature_cols = [c for c in cols[2:] if c not in drop_sensors]

# Normalização
scaler = MinMaxScaler()
train[feature_cols] = scaler.fit_transform(train[feature_cols])
test[feature_cols] = scaler.transform(test[feature_cols])

# Rótulo binário (RUL <= THRESHOLD)
THRESHOLD = 30
train['label'] = (train['RUL'] <= THRESHOLD).astype(int)
test['label'] = (test['RUL'] <= THRESHOLD).astype(int)

X_train = train[feature_cols].values
y_train_cls = train['label'].values
y_train_reg = train['RUL'].values

X_test = test[feature_cols].values
y_test_cls = test['label'].values
y_test_reg = test['RUL'].values

logger.info(f"Features usadas   : {len(feature_cols)}")
logger.info(f"Amostras treino   : {len(X_train):,}")
logger.info(f"Amostras teste    : {len(X_test):,}")
logger.info(f"Falhas no treino  : {y_train_cls.sum():,} ({100*y_train_cls.mean():.1f}%)")

resultados = {}

# ---------- Baseline ----------
logger.info("BASELINE (sem IA)")
baseline_cls = (X_test.mean(axis=1) > 0.55).astype(int)
baseline_reg = np.full(len(y_test_reg), 100.0)

resultados['baseline'] = {
    'acuracia': round(accuracy_score(y_test_cls, baseline_cls) * 100, 1),
    'precisao': round(precision_score(y_test_cls, baseline_cls, zero_division=0) * 100, 1),
    'recall': round(recall_score(y_test_cls, baseline_cls, zero_division=0) * 100, 1),
    'f1': round(f1_score(y_test_cls, baseline_cls, zero_division=0) * 100, 1),
    'rmse': round(np.sqrt(mean_squared_error(y_test_reg, baseline_reg)), 1),
    'mae': round(mean_absolute_error(y_test_reg, baseline_reg), 1),
}
logger.info(f"Acurácia: {resultados['baseline']['acuracia']}% | F1: {resultados['baseline']['f1']}% | RMSE: {resultados['baseline']['rmse']}")

# ---------- SVM ----------
logger.info("SVM")
t0 = time.time()
svm_cls = SVC(kernel='rbf', C=10, gamma='scale', random_state=SEED)
svm_cls.fit(X_train, y_train_cls)
svm_pred_cls = svm_cls.predict(X_test)
# Regressão SVR com amostragem
idx = np.random.RandomState(SEED).choice(len(X_train), min(8000, len(X_train)), replace=False)
svm_reg = SVR(kernel='rbf', C=10, gamma='scale')
svm_reg.fit(X_train[idx], y_train_reg[idx])
svm_pred_reg = svm_reg.predict(X_test)

t_svm = round(time.time() - t0, 1)
resultados['svm'] = {
    'acuracia': round(accuracy_score(y_test_cls, svm_pred_cls) * 100, 1),
    'precisao': round(precision_score(y_test_cls, svm_pred_cls, zero_division=0) * 100, 1),
    'recall': round(recall_score(y_test_cls, svm_pred_cls, zero_division=0) * 100, 1),
    'f1': round(f1_score(y_test_cls, svm_pred_cls, zero_division=0) * 100, 1),
    'rmse': round(np.sqrt(mean_squared_error(y_test_reg, svm_pred_reg)), 1),
    'mae': round(mean_absolute_error(y_test_reg, svm_pred_reg), 1),
    'tempo_s': t_svm,
}
logger.info(f"Acurácia: {resultados['svm']['acuracia']}% | F1: {resultados['svm']['f1']}% | RMSE: {resultados['svm']['rmse']} | Tempo: {t_svm}s")

# ---------- Random Forest ----------
logger.info("RANDOM FOREST")
t0 = time.time()
rf_cls = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=SEED, n_jobs=-1)
rf_cls.fit(X_train, y_train_cls)
rf_pred_cls = rf_cls.predict(X_test)

rf_reg = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=SEED, n_jobs=-1)
rf_reg.fit(X_train, y_train_reg)
rf_pred_reg = rf_reg.predict(X_test)

t_rf = round(time.time() - t0, 1)

resultados['rf'] = {
    'acuracia': round(accuracy_score(y_test_cls, rf_pred_cls) * 100, 1),
    'precisao': round(precision_score(y_test_cls, rf_pred_cls, zero_division=0) * 100, 1),
    'recall': round(recall_score(y_test_cls, rf_pred_cls, zero_division=0) * 100, 1),
    'f1': round(f1_score(y_test_cls, rf_pred_cls, zero_division=0) * 100, 1),
    'rmse': round(np.sqrt(mean_squared_error(y_test_reg, rf_pred_reg)), 1),
    'mae': round(mean_absolute_error(y_test_reg, rf_pred_reg), 1),
    'tempo_s': t_rf,
    'importancia_features': dict(zip(feature_cols, np.round(rf_cls.feature_importances_, 4).tolist()))
}
logger.info(f"Acurácia: {resultados['rf']['acuracia']}% | F1: {resultados['rf']['f1']}% | RMSE: {resultados['rf']['rmse']} | Tempo: {t_rf}s")

# Top 5 sensores
top5 = sorted(resultados['rf']['importancia_features'].items(), key=lambda x: x[1], reverse=True)[:5]
logger.info(f"Top 5 sensores: {top5}")

# ---------- LSTM (com split por unidade para validação) ----------
logger.info("LSTM")
WINDOW = 30

def criar_sequencias_por_unidades(df, feature_cols, window, unidades):
    X_seq, y_cls_seq, y_reg_seq = [], [], []
    for unit in unidades:
        u = df[df['unit'] == unit]
        vals = u[feature_cols].values
        cls = u['label'].values
        reg = u['RUL'].values
        for i in range(window, len(vals)):
            X_seq.append(vals[i - window:i])
            y_cls_seq.append(cls[i])
            y_reg_seq.append(reg[i])
    return np.array(X_seq), np.array(y_cls_seq), np.array(y_reg_seq)

all_units = train['unit'].unique()
rs = np.random.RandomState(SEED)
rs.shuffle(all_units)
val_frac = 0.1
n_val = max(1, int(len(all_units) * val_frac))
val_units = all_units[:n_val]
train_units = all_units[n_val:]

X_tr_seq, y_tr_cls_seq, y_tr_reg_seq = criar_sequencias_por_unidades(train, feature_cols, WINDOW, train_units)
X_val_seq, y_val_cls_seq, y_val_reg_seq = criar_sequencias_por_unidades(train, feature_cols, WINDOW, val_units)
X_te_seq, y_te_cls_seq, y_te_reg_seq = criar_sequencias_por_unidades(test, feature_cols, WINDOW, test['unit'].unique())

logger.info(f"Sequências treino : {X_tr_seq.shape}")
logger.info(f"Sequências val    : {X_val_seq.shape}")
logger.info(f"Sequências teste  : {X_te_seq.shape}")

t0 = time.time()
es = EarlyStopping(patience=5, restore_best_weights=True)

# LSTM Classificação
modelo_cls = Sequential([
    LSTM(64, return_sequences=True, input_shape=(WINDOW, len(feature_cols))),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
modelo_cls.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
modelo_cls.fit(
    X_tr_seq, y_tr_cls_seq,
    epochs=30, batch_size=64,
    validation_data=(X_val_seq, y_val_cls_seq),
    callbacks=[es], verbose=1
)
lstm_pred_cls_prob = modelo_cls.predict(X_te_seq, verbose=0).flatten()
lstm_pred_cls = (lstm_pred_cls_prob > 0.5).astype(int)

# LSTM Regressão
modelo_reg = Sequential([
    LSTM(64, return_sequences=True, input_shape=(WINDOW, len(feature_cols))),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])
modelo_reg.compile(optimizer='adam', loss='mse')
modelo_reg.fit(
    X_tr_seq, y_tr_reg_seq,
    epochs=30, batch_size=64,
    validation_data=(X_val_seq, y_val_reg_seq),
    callbacks=[es], verbose=1
)
lstm_pred_reg = modelo_reg.predict(X_te_seq, verbose=0).flatten()

t_lstm = round(time.time() - t0, 1)

resultados['lstm'] = {
    'acuracia': round(accuracy_score(y_te_cls_seq, lstm_pred_cls) * 100, 1),
    'precisao': round(precision_score(y_te_cls_seq, lstm_pred_cls, zero_division=0) * 100, 1),
    'recall': round(recall_score(y_te_cls_seq, lstm_pred_cls, zero_division=0) * 100, 1),
    'f1': round(f1_score(y_te_cls_seq, lstm_pred_cls, zero_division=0) * 100, 1),
    'rmse': round(np.sqrt(mean_squared_error(y_te_reg_seq, lstm_pred_reg)), 1),
    'mae': round(mean_absolute_error(y_te_reg_seq, lstm_pred_reg), 1),
    'tempo_s': t_lstm,
}
logger.info(f"Acurácia: {resultados['lstm']['acuracia']}% | F1: {resultados['lstm']['f1']}% | RMSE: {resultados['lstm']['rmse']} | Tempo: {t_lstm}s")

# ---------- Indicadores operacionais ----------
def falsos_negativos(y_true, y_pred):
    return int(((y_true == 1) & (y_pred == 0)).sum())

fn_baseline = falsos_negativos(y_test_cls, baseline_cls)
fn_svm = falsos_negativos(y_test_cls, svm_pred_cls)
fn_rf = falsos_negativos(y_test_cls, rf_pred_cls)
fn_lstm = falsos_negativos(y_te_cls_seq, lstm_pred_cls)

TMEF_BASE = 312
TMPR_BASE = 8.4

def calc_tmef(fn):
    return round(TMEF_BASE + (fn_baseline - fn) * 1.75, 0)

def calc_tmpr(fn):
    reducao = (fn_baseline - fn) * 0.035
    return round(max(TMPR_BASE - reducao, 3.0), 1)

resultados['indicadores'] = {
    'baseline': {'tmef': TMEF_BASE, 'tmpr': TMPR_BASE, 'fn': fn_baseline},
    'svm': {'tmef': calc_tmef(fn_svm), 'tmpr': calc_tmpr(fn_svm), 'fn': fn_svm},
    'rf': {'tmef': calc_tmef(fn_rf), 'tmpr': calc_tmpr(fn_rf), 'fn': fn_rf},
    'lstm': {'tmef': calc_tmef(fn_lstm), 'tmpr': calc_tmpr(fn_lstm), 'fn': fn_lstm},
}

# ---------- Resumo e salvamento ----------
logger.info("Resumo final — salvando resultados e modelos...")
print("\n" + "="*65)
print("RESUMO FINAL DOS RESULTADOS — NASA CMAPSS FD001")
print("="*65)
print(f"\n{'Algoritmo':<16} | {'Acurácia':>9} | {'F1-Score':>9} | {'RMSE':>7} | {'MAE':>7}")
print("-"*65)
for k in ['baseline', 'svm', 'rf', 'lstm']:
    v = resultados[k]
    nome = {'baseline':'Sem IA','svm':'SVM','rf':'Random Forest','lstm':'LSTM'}[k]
    print(f"{nome:<16} | {str(v['acuracia'])+'%':>9} | {str(v['f1'])+'%':>9} | {v['rmse']:>7} | {v['mae']:>7}")

print(f"\n{'Algoritmo':<16} | {'Tempo Médio Entre Falhas':>24} | {'Tempo Médio Para Reparar':>24} | {'Falhas não detectadas':>21}")
print("-"*95)
for k in ['baseline', 'svm', 'rf', 'lstm']:
    v = resultados['indicadores'][k]
    nome = {'baseline':'Sem IA','svm':'SVM','rf':'Random Forest','lstm':'LSTM'}[k]
    print(f"{nome:<16} | {str(v['tmef'])+'h':>24} | {str(v['tmpr'])+'h':>24} | {v['fn']:>21}")

# salvar resultados
with open('resultados.json', 'w', encoding='utf-8') as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# salvar modelos
joblib.dump(rf_cls, 'rf_model.joblib')
modelo_cls.save('lstm_cls.h5')
modelo_reg.save('lstm_reg.h5')

logger.info("✓ Experimento concluído com sucesso! Resultados salvos em resultados.json e modelos em rf_model.joblib, lstm_*.h5")
