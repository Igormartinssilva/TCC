import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ...existing code...
# 1. Preparação de dados de exemplo
data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# Transforma os dados em um array numpy e redimensiona para o escalonador
data = np.array(data).reshape(-1, 1)

# Escalonamento dos dados
# Parâmetros ajustáveis:
# - feature_range: intervalo em que os dados serão escalonados. Ex: (0,1) ou (-1,1).
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Criação de sequências
def create_sequences(data, seq_length):
    X, y = [], []
    # Parâmetros influentes:
    # - seq_length: tamanho da janela temporal (quantos passos anteriores usar para prever o próximo).
    #   Aumentar seq_length geralmente demanda mais dados e pode capturar dependências mais longas.
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 3
X, y = create_sequences(scaled_data, seq_length)

# 2. Construção do modelo
model = Sequential()
# Parâmetros do LSTM que você pode alterar:
# - units: número de neurônios na camada LSTM (mais unidades -> mais capacidade, mas maior risco de overfitting).
# - activation: função de ativação interna (ex: 'tanh', 'relu'); para LSTM a ativação padrão interna costuma ser 'tanh'
#   (mudar pode afetar desempenho).
# - input_shape: (seq_length, n_features) — ajustar se usar múltiplas features por timestep.
# - return_sequences: True se quiser empilhar outra camada LSTM depois; False se esta for a última camada LSTM.
# - dropout: taxa de dropout para as entradas (regularização).
# - recurrent_dropout: taxa de dropout para as conexões recorrentes.
# - kernel_regularizer / recurrent_regularizer: regularizadores L1/L2 para evitar overfitting.
# - recurrent_activation: função de ativação para portas internas (ex: 'sigmoid').
# Exemplo de uso:
# model.add(LSTM(units=50, activation='relu', input_shape=(seq_length, 1),
#                return_sequences=False, dropout=0.2, recurrent_dropout=0.0))
model.add(LSTM(units=50, activation='relu', input_shape=(seq_length, 1)))
model.add(Dense(units=1))

# 3. Compilação e treinamento
# Parâmetros ajustáveis na compilação/treino:
# - optimizer: 'adam', 'rmsprop', 'sgd', etc. Você pode configurar a taxa de aprendizado: Adam(learning_rate=0.001).
# - loss: função de perda (ex: 'mean_squared_error', 'mae', 'huber', etc).
# - metrics: métricas que quer acompanhar.
# - epochs: número de épocas de treino (aumentar = mais treino, risco de overfitting).
# - batch_size: tamanho do lote por iteração (impacta convergência e uso de memória).
# - verbose: 0/1/2 para controlar saída do treinamento.
# - validation_split: fração dos dados usada para validação.
# - shuffle: embaralhar os dados a cada época (útil em muitos casos).
# - callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau são comuns para controlar overfitting e taxa de aprendizado.
model.compile(optimizer='adam', loss='mean_squared_error')
# Exemplo: model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mae')
model.fit(X, y, epochs=1000, verbose=0)

# 4. Previsão com novos dados
# Observação importante:
# - Qualquer novo dado deve ser escalado exatamente com o mesmo scaler usado no treino.
# - Se o modelo foi treinado com input_shape=(seq_length, 1), new_data deve ter essa forma e ser escalado.
# Exemplo corrigido: escalar a nova sequência antes de prever.
new_data_raw = np.array([80, 90, 100]).reshape(-1, 1)
# escalona usando o mesmo scaler
new_data_scaled = scaler.transform(new_data_raw)
# reorganiza para (1, seq_length, 1)
new_data = new_data_scaled.reshape(1, seq_length, 1)

scaled_prediction = model.predict(new_data)
# Inverter o escalonamento da previsão
prediction = scaler.inverse_transform(scaled_prediction)
print("Previsão do próximo valor:", prediction[0][0])

# Resumo rápido dos hiperparâmetros que mais impactam:
# - seq_length: janela temporal (define contexto).
# - units: capacidade do modelo.
# - dropout / recurrent_dropout / regularizers: controlam overfitting.
# - optimizer / learning_rate: afetam velocidade e qualidade da convergência.
# - loss: define o objetivo de otimização.
# - epochs / batch_size / validation_split / callbacks: controlam o processo de treino.
# Ajuste esses parâmetros com validação (ex: validação cruzada ou validation_split) e monitore