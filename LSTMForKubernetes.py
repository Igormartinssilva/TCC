import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ...existing code...
# Fonte de dados: Prometheus (exemplo) ou CSV
def fetch_prometheus_range(prom_url, query, start_dt, end_dt, step='60s'):
    """
    Busca uma série temporal do Prometheus via query_range.
    - prom_url: ex: 'http://prometheus-server:9090'
    - query: ex: "rate(container_cpu_usage_seconds_total{pod=~'my-app-.*'}[5m])"
    - start_dt / end_dt: datetime
    - step: intervalo de amostragem ('60s', '30s', etc.)
    Retorna um DataFrame com index datetime e coluna 'value'.
    """
    url = f"{prom_url.rstrip('/')}/api/v1/query_range"
    params = {
        'query': query,
        'start': start_dt.timestamp(),
        'end': end_dt.timestamp(),
        'step': step
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()['data']['result']
    if not data:
        raise ValueError("Nenhum resultado retornado pelo Prometheus para essa query.")
    # pega a primeira série (ajuste se houver múltiplas)
    values = data[0]['values']  # [ [ts, val], ... ]
    times = [datetime.fromtimestamp(float(ts)) for ts, _ in values]
    vals = [float(v) for _, v in values]
    df = pd.DataFrame({'value': vals}, index=pd.DatetimeIndex(times))
    return df

def load_from_csv(path, timestamp_col='timestamp', value_col='cpu_usage'):
    """
    Carrega CSV com colunas timestamp e cpu usage (valores numéricos).
    """
    df = pd.read_csv(path, parse_dates=[timestamp_col])
    df = df.set_index(timestamp_col)
    df = df[[value_col]].rename(columns={value_col: 'value'})
    return df

def preprocess_timeseries(df, freq='1min', method='linear'):
    """
    - Resample para frequência fixa (freq), por exemplo '1min'.
    - Preenche valores faltantes com interpolation.
    - Remove outliers simples (opcional).
    """
    df = df.sort_index()
    df = df.resample(freq).mean()
    df['value'] = df['value'].interpolate(method=method).fillna(method='ffill').fillna(method='bfill')
    return df

def create_sequences(data_array, seq_length):
    X, y = [], []
    for i in range(len(data_array) - seq_length):
        X.append(data_array[i:i+seq_length])
        y.append(data_array[i+seq_length])
    return np.array(X), np.array(y)

def build_lstm_model(seq_length, n_features=1,
                     units=64, dropout_rate=0.2,
                     stacked=False, learning_rate=1e-3):
    model = Sequential()
    if stacked:
        model.add(LSTM(units, activation='tanh', return_sequences=True,
                       input_shape=(seq_length, n_features)))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(units//2, activation='tanh'))
    else:
        model.add(LSTM(units, activation='tanh', input_shape=(seq_length, n_features)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    # configurar otimizador com learning_rate ajustável
    from tensorflow.keras.optimizers import Adam
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

# EXEMPLO DE USO (ajuste conforme necessário)
if __name__ == "__main__":
    # --- Fonte de dados: escolha uma ---
    # 1) Prometheus
    # prom_url = "http://prometheus-server:9090"
    # query = "rate(container_cpu_usage_seconds_total{pod=~'my-app-.*'}[5m])"
    # end = datetime.utcnow()
    # start = end - timedelta(hours=6)  # janelas maiores para melhores resultados
    # df = fetch_prometheus_range(prom_url, query, start, end, step='60s')

    # 2) CSV local
    # df = load_from_csv("c:\\path\\to\\cpu_usage.csv", timestamp_col='time', value_col='cpu')

    # Para demo/local usa a série exemplo se não houver fonte:
    try:
        df  # se df já definido acima
    except NameError:
        data = [10,20,30,40,50,60,70,80,90,100]
        df = pd.DataFrame({'value': data}, index=pd.date_range(end=datetime.utcnow(), periods=len(data), freq='1min'))

    # --- Pré-processamento ---
    df = preprocess_timeseries(df, freq='1min')
    seq_length = 12  # Hiperparâmetro: quanto histórico usar (ex.: 12 minutos)
    test_fraction = 0.2

    # --- Escalonamento (fit no treino apenas) ---
    # split temporal
    split_idx = int(len(df) * (1 - test_fraction))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx - seq_length:]  # garante janelas iniciais para o test

    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(train_df[['value']])

    train_scaled = scaler.transform(train_df[['value']])
    test_scaled = scaler.transform(test_df[['value']])

    # --- Criação de sequências ---
    X_train, y_train = create_sequences(train_scaled, seq_length)
    X_test, y_test = create_sequences(test_scaled, seq_length)

    # reshape para (samples, seq_length, n_features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # --- Modelo e treino ---
    model = build_lstm_model(seq_length, n_features=1,
                             units=64, dropout_rate=0.2,
                             stacked=True, learning_rate=1e-3)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
        ModelCheckpoint("best_lstm_cpu.h5", save_best_only=True, monitor='val_loss')
    ]

    history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                        validation_data=(X_test, y_test),
                        callbacks=callbacks, verbose=1)

    # --- Previsão e inverter escala ---
    preds_scaled = model.predict(X_test)
    preds = scaler.inverse_transform(preds_scaled)
    y_true = scaler.inverse_transform(y_test)

    print("Últimas previsões (invertidas):", preds.flatten()[-5:])
    print("Últimos valores reais:", y_true.flatten()[-5:])

    # --- Previsão iterativa de N passos à frente ---
    def predict_future(model, last_window, steps, scaler):
        window = last_window.copy()
        results = []
        for _ in range(steps):
            x = window.reshape(1, window.shape[0], 1)
            p = model.predict(x)[0,0]
            results.append(p)
            window = np.roll(window, -1)
            window[-1] = p
        return scaler.inverse_transform(np.array(results).reshape(-1,1)).flatten()

    last_window = test_scaled[-seq_length:].flatten()
    future_steps = 10
    future_preds = predict_future(model, last_window, future_steps, scaler)
    print("Previsões futuras:", future_preds)

# Resumo rápido de parâmetros para ajustar no contexto de CPU em Kubernetes:
# - fonte/densidade dos dados: step/resample (ex: 15s/1m/5m) — frequência afeta seq_length e ruído.
# - seq_length: janela de entrada (depende da periodicidade e da dependência temporal).
# - units / stacked / dropout: capacidade vs overfitting.
# - learning_rate / optimizer: ajuste da convergência.
# - batch_size / epochs / callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint): controle do treino.
# - scaler: MinMax vs Standard dependendo da distribuição.
# - features: incluir métricas adicionais (memory, network, pod replicas) melhora previsão multivariada.