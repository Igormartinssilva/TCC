# Time-LLM - Guia Completo de Parâmetros

## 📊 Tempo de Treinamento

Com base no seu setup atual (batch_size=16, GPU/CPU):
- **Tempo por epoch**: ~1-2 horas (CPU) ou ~10-15 minutos (GPU)
- **Total com 50 epochs**: ~50-100 horas (CPU)

**Recomendações para acelerar:**
1. Reduzir `--train_epochs` de 50 para 10-20
2. Aumentar `--batch_size` de 16 para 32-64 (se tiver VRAM)
3. Reduzir `--seq_len` de 512 para 256
4. Usar GPU (CUDA/NVIDIA) em vez de CPU

---

## 🎛️ Parâmetros Disponíveis

### **TASK & MODEL BASICS** (Obrigatórios)

| Parâmetro | Tipo | Default | Opções | Descrição | Exemplo |
|-----------|------|---------|--------|-----------|---------|
| `--task_name` | str | - | `long_term_forecast`, `short_term_forecast`, `imputation`, `classification`, `anomaly_detection` | Tipo de tarefa a executar | `long_term_forecast` |
| `--is_training` | int | - | `0`, `1` | 0=teste/inferência, 1=treinar | `1` |
| `--model_id` | str | - | Qualquer string | ID único do experimento (nome) | `ETTh1_512_96_gpt2` |
| `--model_comment` | str | - | Qualquer string | Comentário/descrição do experimento | `TimeLLM-GPT2-v1` |
| `--model` | str | - | `TimeLLM`, `Autoformer`, `DLinear` | Modelo a usar | `TimeLLM` |
| `--data` | str | - | `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `electricity`, `traffic`, `weather`, `m4` | Dataset | `ETTh1` |

---

### **DATA LOADING** (Paths e Dataset)

| Parâmetro | Tipo | Default | Descrição | Exemplo |
|-----------|------|---------|-----------|---------|
| `--root_path` | str | `./dataset` | Caminho raiz dos datasets | `./dataset` ou `/path/to/datasets` |
| `--data_path` | str | `ETTh1.csv` | Nome do arquivo CSV dentro de root_path | `ETTh1.csv` ou `electricity.csv` |
| `--features` | str | `M` | `M`=multivariate→multivariate, `S`=univariate, `MS`=multivariate→univariate | `M` |
| `--target` | str | `OT` | Coluna alvo (para modo S ou MS) | `OT` |
| `--loader` | str | `modal` | Tipo de loader (modal ou sem especificar) | `modal` |
| `--freq` | str | `h` | Frequência temporal: `s`(segundo), `t`(minuto), `h`(hora), `d`(dia), `b`(negócio), `w`(semana), `m`(mês) | `h` para ETT, `d` para climate |
| `--num_workers` | int | `10` | Threads para carregar dados (Windows: use `0`) | `0` (Windows) ou `4` (Linux) |

---

### **FORECASTING TASK** (Tamanhos de Sequência)

| Parâmetro | Tipo | Default | Faixa | Descrição | Exemplo |
|-----------|------|---------|-------|-----------|---------|
| `--seq_len` | int | `96` | Tipicamente 96-512 | Tamanho da sequência de entrada | `512` |
| `--label_len` | int | `48` | ~metade de seq_len | Comprimento do label (começo do input) | `48` |
| `--pred_len` | int | `96` | 1-720 | Quantos passos prever no futuro | `96` |
| `--seasonal_patterns` | str | `Monthly` | `Daily`, `Weekly`, `Monthly`, `Quarterly`, `Yearly` | Padrão sazonal (M4 dataset) | `Monthly` |

---

### **MODEL ARCHITECTURE** (Dimensões)

| Parâmetro | Tipo | Default | Faixa Recomendada | Descrição | Exemplo |
|-----------|------|---------|------------------|-----------|---------|
| `--d_model` | int | `16` | 16-256 | Dimensão do modelo transformer | `32` |
| `--d_ff` | int | `32` | 32-1024 | Dimensão da feed-forward layer | `128` |
| `--n_heads` | int | `8` | 4-16 | Num de cabeças de atenção | `8` |
| `--e_layers` | int | `2` | 1-6 | Num de encoder layers | `2` |
| `--d_layers` | int | `1` | 1-4 | Num de decoder layers | `1` |
| `--moving_avg` | int | `25` | 5-49 | Window size moving average | `25` |
| `--factor` | int | `1` | 1-10 | Fator sparse attention | `3` |
| `--dropout` | float | `0.1` | 0.0-0.5 | Dropout rate | `0.1` |
| `--embed` | str | `timeF` | `timeF`, `fixed`, `learned` | Tipo embedding temporal | `timeF` |
| `--activation` | str | `gelu` | `gelu`, `relu`, `sigmoid` | Função ativação | `gelu` |
| `--enc_in` | int | `7` | Depende do dataset | Num features entrada encoder | `7` (ETT) ou `21` (electricity) |
| `--dec_in` | int | `7` | Mesmo que enc_in | Num features entrada decoder | `7` |
| `--c_out` | int | `7` | Mesmo que enc_in/dec_in | Num features saída | `7` |

---

### **PATCH EMBEDDING** (Time-LLM específico)

| Parâmetro | Tipo | Default | Faixa | Descrição | Exemplo |
|-----------|------|---------|-------|-----------|---------|
| `--patch_len` | int | `16` | 8-32 | Tamanho de cada patch | `16` |
| `--stride` | int | `8` | 1-16 | Passo entre patches | `8` |
| `--prompt_domain` | int | `0` | `0`, `1` | Use prompt de domínio (0=não, 1=sim) | `0` |

---

### **LLM CONFIGURATION** (Backbone Language Model)

| Parâmetro | Tipo | Default | Opções | Descrição | Exemplo |
|-----------|------|---------|--------|-----------|---------|
| `--llm_model` | str | `LLAMA` | `LLAMA`, `GPT2`, `BERT` | Qual LLM usar como backbone | `GPT2` |
| `--llm_dim` | int | `4096` | 768 (GPT2/BERT), 4096 (LLAMA) | Dimensão embedding LLM | `768` (GPT2) |
| `--llm_layers` | int | `6` | 6-32 | Quantas camadas do LLM usar | `12` (GPT2 tem 12) |

**Nota sobre LLMs:**
- **GPT2**: llm_dim=768, llm_layers=12, rápido, recomendado para teste
- **LLAMA-7B**: llm_dim=4096, llm_layers=32, mais poderoso, requer mais VRAM
- **BERT-base**: llm_dim=768, llm_layers=12

---

### **OPTIMIZATION** (Treinamento)

| Parâmetro | Tipo | Default | Faixa | Descrição | Exemplo |
|-----------|------|---------|-------|-----------|---------|
| `--train_epochs` | int | `10` | 1-500 | Número de epochs de treinamento | `50` |
| `--align_epochs` | int | `10` | 1-50 | Epochs para alinhar embeddings | `10` |
| `--batch_size` | int | `32` | 8-256 | Tamanho batch treinamento | `16` |
| `--eval_batch_size` | int | `8` | 4-64 | Tamanho batch validação/teste | `8` |
| `--learning_rate` | float | `0.0001` | 1e-5 a 1e-3 | Learning rate (otimizador Adam) | `0.0001` |
| `--lradj` | str | `type1` | `type1`, `type2`, `type3`, `COS`, `TST`, `constant`, `PEMS` | Estratégia ajuste LR | `type1` |
| `--patience` | int | `10` | 3-20 | Early stopping patience | `10` |
| `--pct_start` | float | `0.2` | 0.0-1.0 | % do treinamento em warm-up (OneCycleLR) | `0.2` |
| `--loss` | str | `MSE` | `MSE`, `MAE`, `MAPE` | Função perda | `MSE` |
| `--itr` | int | `1` | 1-5 | Quantas vezes rodar experimento | `1` |
| `--percent` | int | `100` | 1-100 | % dos dados para treinar | `100` |

---

### **MISCELLANEOUS** (Diversos)

| Parâmetro | Tipo | Default | Descrição | Exemplo |
|-----------|------|---------|-----------|---------|
| `--checkpoints` | str | `./checkpoints/` | Diretório salvar modelos | `./checkpoints/` |
| `--des` | str | `test` | Descrição experimento (salvo em logs) | `Exp` |
| `--seed` | int | `2021` | Random seed | `2021` |
| `--use_amp` | flag | False | Usar Automatic Mixed Precision | `--use_amp` |
| `--output_attention` | flag | False | Salvar mapas atenção | `--output_attention` |

---

## 📋 EXEMPLOS DE COMANDOS

### **Windows (com `uv`)**

#### 1️⃣ Teste Rápido (5 epochs, batch pequeno)
```powershell
uv run python run_main_windows_fixed.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_512_96_gpt2 --model TimeLLM --data ETTh1 --features M --seq_len 512 --label_len 48 --pred_len 96 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des Exp --itr 1 --d_model 32 --d_ff 128 --batch_size 16 --learning_rate 0.0001 --llm_model GPT2 --llm_dim 768 --llm_layers 12 --train_epochs 50 --model_comment TimeLLM-GPT2
```

#### 2️⃣ Produção (50 epochs, melhor tuning)
```powershell
uv run python run_main_windows_fixed.py `
  --task_name long_term_forecast `
  --is_training 1 `
  --model_id ETTh1_production `
  --model TimeLLM `
  --data ETTh1 `
  --features M `
  --root_path ./dataset/ETT-small/ `
  --data_path ETTh1.csv `
  --seq_len 512 `
  --label_len 48 `
  --pred_len 96 `
  --batch_size 16 `
  --train_epochs 50 `
  --d_model 32 `
  --d_ff 128 `
  --llm_model GPT2 `
  --llm_dim 768 `
  --llm_layers 12 `
  --learning_rate 0.0001 `
  --enc_in 7 `
  --dec_in 7 `
  --c_out 7 `
  --patience 10 `
  --lradj type1 `
  --model_comment TimeLLM-GPT2-prod
```

#### 3️⃣ Dataset Electricity
```powershell
uv run python run_main_windows_fixed.py `
  --task_name long_term_forecast `
  --is_training 1 `
  --model_id electricity_exp `
  --model TimeLLM `
  --data electricity `
  --features M `
  --root_path ./dataset/ `
  --data_path electricity.csv `
  --seq_len 336 `
  --pred_len 96 `
  --batch_size 32 `
  --train_epochs 10 `
  --enc_in 321 `
  --dec_in 321 `
  --c_out 321 `
  --llm_model GPT2 `
  --llm_dim 768 `
  --model_comment electricity-baseline
```

#### 4️⃣ Teste Inferência (is_training=0)
```powershell
uv run python run_main_windows_fixed.py `
  --task_name long_term_forecast `
  --is_training 0 `
  --model_id ETTh1_inference `
  --model TimeLLM `
  --data ETTh1 `
  --root_path ./dataset/ETT-small/ `
  --data_path ETTh1.csv `
  --seq_len 512 `
  --pred_len 96 `
  --batch_size 1 `
  --enc_in 7 `
  --dec_in 7 `
  --c_out 7 `
  --checkpoints ./checkpoints/ `
  --llm_model GPT2 `
  --model_comment inference-test
```

---

### **Linux** (idêntico, sem backticks)

#### 1️⃣ Teste Rápido
```bash
uv run python run_main_windows_fixed.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id test_quick \
  --model TimeLLM \
  --data ETTh1 \
  --features M \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --seq_len 96 \
  --pred_len 96 \
  --batch_size 8 \
  --train_epochs 5 \
  --d_model 16 \
  --d_ff 32 \
  --llm_model GPT2 \
  --llm_dim 768 \
  --llm_layers 6 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --model_comment quick-test
```

#### 2️⃣ Com Aceleração GPU (Linux)
```bash
# Usando accelerate para multi-GPU
accelerate launch --multi_gpu run_main_windows_fixed.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ETTh1_gpu \
  --model TimeLLM \
  --data ETTh1 \
  --features M \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --batch_size 64 \
  --train_epochs 100 \
  --d_model 64 \
  --d_ff 256 \
  --llm_model LLAMA \
  --llm_dim 4096 \
  --llm_layers 32 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --model_comment llama-7b-gpu
```

#### 3️⃣ Produção (50 epochs)
```bash
uv run python run_main_windows_fixed.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ETTh1_production \
  --model TimeLLM \
  --data ETTh1 \
  --features M \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --batch_size 32 \
  --train_epochs 50 \
  --d_model 32 \
  --d_ff 128 \
  --llm_model GPT2 \
  --llm_dim 768 \
  --llm_layers 12 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --learning_rate 0.0001 \
  --patience 15 \
  --model_comment gpt2-production
```

---

## 🚀 RECOMENDAÇÕES POR HARDWARE

### CPU (Intel/AMD)
```bash
--batch_size 8
--train_epochs 10
--seq_len 256
--d_model 16
--llm_model GPT2
```
**Tempo estimado**: ~5-10 horas para 10 epochs

### GPU (1x RTX 3060 / 12GB VRAM)
```bash
--batch_size 32
--train_epochs 50
--seq_len 512
--d_model 32
--llm_model GPT2
```
**Tempo estimado**: ~2-3 horas para 50 epochs

### GPU (1x A100 / 80GB VRAM)
```bash
--batch_size 128
--train_epochs 100
--seq_len 512
--d_model 64
--llm_model LLAMA
--llm_layers 32
```
**Tempo estimado**: ~30 minutos para 100 epochs

---

## 📊 DATASETS DISPONIVEIS

| Dataset | Caminho | Freq | Features | Recomendado |
|---------|---------|------|----------|------------|
| **ETTh1** | `dataset/ETT-small/ETTh1.csv` | Horária | 7 | ✅ Sim (teste rápido) |
| **ETTh2** | `dataset/ETT-small/ETTh2.csv` | Horária | 7 | ✅ Sim |
| **ETTm1** | `dataset/ETT-small/ETTm1.csv` | 15min | 7 | ✅ Sim |
| **ETTm2** | `dataset/ETT-small/ETTm2.csv` | 15min | 7 | ✅ Sim |
| **Electricity** | `dataset/electricity/electricity.csv` | Horária | 321 | Para experimentos |
| **Traffic** | `dataset/traffic/traffic.csv` | Horária | 862 | Para experimentos |
| **Weather** | `dataset/weather/weather.csv` | 10min | 21 | Para experimentos |
| **M4** | `dataset/m4/` | Variado | 1 | Benchmark |

---

## 🔧 TROUBLESHOOTING

**Problema**: `RuntimeError: expected scalar type BFloat16 but found Float`
- **Solução**: Já foi corrigida em `models/TimeLLM.py`

**Problema**: `AttributeError: np.Inf was removed`
- **Solução**: Já foi corrigida em `utils/tools.py` (np.inf)

**Problema**: Multiprocessing erro no Windows
- **Solução**: Use `num_workers=0` e `if __name__ == '__main__':`

**Problema**: Encoding error (CP1252)
- **Solução**: Arquivo já foi corrigido com `encoding='utf-8'`

**Problema**: Muito lento
- **Solução**: Reduzir `batch_size`, `seq_len`, ou usar GPU

---

## 📝 ESTRUTURA OUTPUT

Após rodar, os resultados serão salvos em:
```
checkpoints/
├── long_term_forecast_ETTh1_512_96_gpt2_ft_M_sl512_ll48_pl96_dm32_nh8_el2_dl1_df128_fc3_eb_Exp_0-TimeLLM-GPT2/
│   ├── checkpoint.pth (melhor modelo)
│   └── logs.txt
```

---

## 🤖 COMPARAÇÃO DOS MODELOS

### **Time-LLM** (Recomendado para usar)
**Arquitetura**: Reprogramming framework com LLM como backbone
- **Características principais**:
  - Usa modelos pre-treinados (GPT2, LLAMA, BERT) como backbone
  - Patch Reprogramming: converte séries temporais em "text like" representations
  - Prompt Augmentation: usa prompts declarativos para guiar o LLM
  - Funciona com **qualquer LLM** via HuggingFace

- **Vantagens**:
  ✅ Aproveita conhecimento pre-treinado de LLMs
  ✅ Pode usar modelos maiores (LLAMA-7B) para melhor performance
  ✅ Flexível: trocar LLM é simples
  ✅ State-of-the-art em muitos benchmarks (ICLR'24)

- **Desvantagens**:
  ❌ Mais lento (requer processamento LLM)
  ❌ Precisa de mais VRAM (especialmente com LLMs grandes)
  ❌ Complexo: mais parâmetros para tunar

- **Melhor para**:
  - Alta acurácia (produção crítica)
  - Quando tiver GPU disponível
  - Datasets menores (~700 pontos)

- **Comando típico**:
```powershell
--model TimeLLM --llm_model GPT2 --llm_dim 768 --d_model 32 --d_ff 128
```

- **Baseline TimeLLM (ETTh1, seq_len=512, pred_len=96)**:
  - MSE: ~0.38
  - MAE: ~0.42

---

### **Autoformer** (Para comparação)
**Arquitetura**: Transformer com Auto-Correlation mecanismo
- **Características principais**:
  - Decomposição série em trend + seasonality
  - Auto-correlation prioriza padrões relevantes
  - Arquitetura pure transformer (sem LLM)

- **Vantagens**:
  ✅ Mais simples que Time-LLM
  ✅ Rápido: ~2x mais rápido que Time-LLM
  ✅ Bom para séries com padrão seasonal forte
  ✅ Menor consumo VRAM

- **Desvantagens**:
  ❌ Menor acurácia que Time-LLM
  ❌ Não aproveita conhecimento pre-treinado
  ❌ Menos flexível

- **Melhor para**:
  - Testes rápidos
  - CPU
  - Séries com sazonalidade clara

- **Comando típico**:
```powershell
--model Autoformer --d_model 32 --d_ff 128 --e_layers 2 --d_layers 1
```

- **Baseline Autoformer (ETTh1, seq_len=512, pred_len=96)**:
  - MSE: ~0.51
  - MAE: ~0.50

---

### **DLinear** (Mais Simples)
**Arquitetura**: Decomposition Linear - modelo linear puro com decomposição
- **Características principais**:
  - MLP simples: 2 camadas fully-connected
  - Decomposição: trend + seasonality
  - Sem atenção, sem LLM: extremamente leve

- **Vantagens**:
  ✅ Mais rápido: ~5-10x que Time-LLM
  ✅ Mínimo consumo VRAM (~200MB)
  ✅ Simples: poucos parâmetros
  ✅ Excelente baseline

- **Desvantagens**:
  ❌ Menor acurácia
  ❌ Não captura relações complexas
  ❌ Ruim para padrões não-lineares

- **Melhor para**:
  - Baseline rápido
  - CPU antigo
  - IoT / edge devices
  - Prototipagem rápida

- **Comando típico**:
```powershell
--model DLinear --d_model 16 --d_ff 32
```

- **Baseline DLinear (ETTh1, seq_len=512, pred_len=96)**:
  - MSE: ~0.65
  - MAE: ~0.60

---

### **📊 Quadro Comparativo**

| Aspecto | Time-LLM | Autoformer | DLinear |
|---------|----------|-----------|---------|
| **Acurácia** | ⭐⭐⭐⭐⭐ (SOTA) | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Velocidade** | 2.3s/iter (CPU) | 1.0s/iter | 0.3s/iter |
| **VRAM** | 4-16GB | 2-4GB | 200-500MB |
| **Com GPU** | Muito rápido | Rápido | Ultra-rápido |
| **Complexidade** | Alta | Média | Baixa |
| **Tuning** | Muito trabalho | Médio | Mínimo |
| **ICLR'24?** | ✅ Sim (novo) | ❌ Não (2023) | ❌ Não (2023) |

---

## 📊 DATASETS DETALHADOS

### **ETT (Electricity Transformer Temperature)**
Conjunto de dados real de uma subestação elétrica chinesa.

#### **ETTh1** (Horário)
- **Frequência**: Cada hora
- **Período**: 2016-2018 (2 anos)
- **Features**: 7
  - `OT`: Oil Temperature (alvo principal)
  - `V SL`: Voltage (low)
  - `V HL`: Voltage (high)
  - `T`: Temperature
  - `AP`: Apparent Power
  - `Var`: Reactive Power
  - `QF`: Power Factor
- **Total de pontos**: ~17,400
- **Tamanho arquivo**: ~2MB
- **Use para**: Testes rápidos, desenvolvimento

**Análise**: Série com sazonalidade horária e mudanças diárias. Bom para validar modelos.

---

#### **ETTh2** (Horário, Período 2)
- **Frequência**: Cada hora
- **Período**: 2016-2018 (2 anos, segundo período)
- **Features**: 7 (idênticas ao ETTh1)
- **Total de pontos**: ~17,400
- **Tamanho arquivo**: ~2MB
- **Use para**: Validação cruzada, comparação

**Diferença com ETTh1**: Outro período do mesmo equipamento. Mais desafiador (mais volatilidade).

---

#### **ETTm1 & ETTm2** (15 minutos)
- **Frequência**: A cada 15 minutos
- **Período**: 2016-2018
- **Features**: 7 (idênticas)
- **Total de pontos**: ~69,600 (4x mais que ETTh)
- **Tamanho arquivo**: ~5-6MB
- **Use para**: Séries de frequência fina

---

### **Electricity** (Disponível se baixado)
- **Frequência**: Horária
- **Features**: 321 (consumo de 321 clientes)
- **Total de pontos**: ~26,000
- **Use para**: Multivariate learning
- **Desafio**: Muito maior (321 features vs 7 do ETT)

---

### **Traffic** (Disponível se baixado)
- **Frequência**: Horária
- **Features**: 862 (sensores de tráfego viário)
- **Total de pontos**: ~17,500
- **Use para**: Séries com padrão forte (rush hour)
- **Desafio**: Altamente não-estacionário

---

### **Weather** (Disponível se baixado)
- **Frequência**: 10 minutos
- **Features**: 21 (temperatura, umidade, vento, etc)
- **Total de pontos**: ~52,600
- **Use para**: Dados climáticos reais
- **Desafio**: Ruído ambiental, padrões sazonais complexos

---

### **M4** (Monthly, Quarterly, Yearly)
- **Origem**: M4 Forecasting Competition (financeiro)
- **Features**: 1 (univariate)
- **Total de séries**: 1000s
- **Frequências**: 
  - Daily: ~4k séries
  - Weekly: ~1.5k
  - Monthly: ~20k
  - Quarterly: ~5.8k
  - Yearly: ~4.3k
- **Use para**: Benchmark em escala
- **Desafio**: Séries muito heterogêneas

---

### 🎯 Qual Dataset Escolher?

| Caso | Dataset | Razão |
|------|---------|-------|
| Teste rápido (2 min) | **ETTh1 com seq_len=96** | Rápido, problema simples |
| Validação (1 hora) | **ETTh1 com seq_len=512** | Tamanho moderado |
| Experimento séri (1 GPU-dia) | **ETTh2 ou ETTm1** | Mais desafiador |
| Comparison benchmark | **ETTh1 + ETTh2 + ETTm1** | Padrão indústria |
| Novos testes | **Electricity ou Traffic** | Maior, mais realista |
| Competição | **M4** | SOTA submissions |

---

### 📈 Estatísticas dos Datasets

```
ETTh1:
├── Min: 25.3°C
├── Max: 46.5°C
├── Mean: 35.8°C
├── Seasonal period: 24 (horário)
└── Trend: crescimento leve

Electricity:
├── Min: 0 kWh
├── Max: 26,000 kWh
├── Mean: 10,000 kWh
├── Seasonal period: 24 (horário) + 7 (semanal)
└── Trend: aumento anual
```

---

## � TRANSFER LEARNING & GENERALIZAÇÃO

### O que é Transfer Learning no Time-LLM?

Time-LLM é **especialista no dataset em que foi treinado**, especialmente quando usa LLMs pre-treinados (GPT-2, LLAMA). Quando você treina em ETTh1 (temperatura de óleo), o modelo aprende padrões **muito específicos** daquele domínio.

**Pergunta comum**: "Se treinar em ETTh1, o modelo entenderá qualquer série temporal?"
**Resposta**: Parcialmente. O modelo aprenderá some padrões básicos de séries temporais, mas terá performance degradada em novos domínios.

---

### 3 Estratégias de Transfer Learning

#### **1️⃣ Zero-Shot** (Sem fine-tuning)
Usar o modelo treinado em ETTh1 para prever Electricity diretamente.

- **Performance esperada**: ~60% em domínios similares
- **Tempo**: 0 minutos (apenas inferência)
- **VRAM**: Mínimo
- **Código**:
```powershell
uv run python run_main_windows_fixed.py `
  --is_training 0 `
  --model TimeLLM `
  --data electricity `
  --pred_len 96 `
  --checkpoints ./checkpoints/seu_checkpoint/
```

**Quando usar**:
- ✅ Quick test/prototipagem
- ✅ Dados similares ( temporal patterns)
- ❌ Não para produção

---

#### **2️⃣ Fine-Tuning** (Recomendado)
Treinar apenas as camadas finais por poucas epochs, "adaptando" o modelo.

- **Performance esperada**: ~80-90% em domínios similares
- **Tempo**: 1-6 horas (5-20 epochs, depende do dataset)
- **VRAM**: Moderado
- **Código para Electricity**:
```powershell
uv run python run_main_windows_fixed.py `
  --task_name long_term_forecast `
  --is_training 1 `
  --model_id electricity_finetuned `
  --model TimeLLM `
  --data electricity `
  --features M `
  --root_path ./dataset/ `
  --data_path electricity.csv `
  --seq_len 336 `
  --pred_len 96 `
  --batch_size 32 `
  --train_epochs 10 `
  --learning_rate 0.00001 `
  --enc_in 321 `
  --dec_in 321 `
  --c_out 321 `
  --llm_model GPT2 `
  --llm_layers 6 `
  --model_comment electricity-finetuned
```

**Quando usar**:
- ✅ Dataset novo mas similar (Eletricidade, Tráfego)
- ✅ Dados suficientes (~1000+ pontos)
- ✅ Balanceado entre tempo e acurácia
- ❌ Dataset muito diferente

---

#### **3️⃣ Full Retrain** (Mais robusto)
Treinar do zero com todos os parâmetros.

- **Performance esperada**: ~95-100%
- **Tempo**: 50+ epochs (110+ horas em CPU)
- **VRAM**: Alto
- **Código**:
```powershell
uv run python run_main_windows_fixed.py `
  --task_name long_term_forecast `
  --is_training 1 `
  --model_id electricity_scratch `
  --model TimeLLM `
  --data electricity `
  --features M `
  --root_path ./dataset/ `
  --data_path electricity.csv `
  --seq_len 336 `
  --pred_len 96 `
  --batch_size 32 `
  --train_epochs 100 `
  --learning_rate 0.0001 `
  --train_epochs 100 `
  --enc_in 321 `
  --dec_in 321 `
  --c_out 321 `
  --llm_model GPT2 `
  --model_comment electricity-fullretrain
```

**Quando usar**:
- ✅ Dataset muito diferente
- ✅ Alta acurácia crítica
- ✅ Recursos computacionais disponíveis
- ❌ Tempo limitado

---

### 📊 Performance Esperada por Dataset

Assumindo: **Modelo treinado em ETTh1, testado em novo dataset**

| Novo Dataset | Similaridade | Zero-Shot | Fine-Tune (10 ep) | Full Retrain | Recomendação |
|--------------|--------------|-----------|-------------------|--------------|--------------|
| **ETTh2** | Muito alta | 85-95% | 90-98% | 95-99% | **Zero-shot** funciona! |
| **ETTm1** | Muito alta | 80-92% | 88-96% | 93-99% | **Zero-shot** se urgente |
| **Electricity** | Alta | 75-85% | 85-92% | 90-98% | **Fine-tune 10-15 epochs** |
| **Traffic** | Moderada | 60-75% | 75-85% | 85-95% | **Fine-tune 20-30 epochs** |
| **Weather** | Moderada | 65-80% | 78-88% | 88-96% | **Fine-tune 15-20 epochs** |
| **Crypto/Finance** | Baixa | 40-60% | 60-75% | 75-90% | **Full Retrain necessário** |

---

### 🎯 Como Escolher a Estratégia?

```
Novo dataset disponível?
│
├─ NÃO (apenas inferência) → Use Zero-Shot
│
├─ SIM, mas poucos dados (<500 pontos) → Fine-tune 5-10 epochs
│
├─ SIM, dados moderados (500-5k pontos)
│  └─ Similar a ETTh1? → Fine-tune 10-15 epochs
│  └─ Diferente? → Fine-tune 20-30 epochs
│
└─ SIM, muitos dados (>5k pontos) e crítico?
   └─ Temporal patterns similares? → Fine-tune 30-50 epochs
   └─ Muito diferente? → Full Retrain com learning_rate=0.0001
```

---

### 🔧 Boas Práticas de Fine-Tuning

1. **Learning rate menor**: Use `--learning_rate 0.00001` (10x menor)
   - Preserva conhecimento pré-aprendido
   - Evita "catastrophic forgetting"

2. **Poucas epochs**: Comece com `--train_epochs 10`
   - Fine-tune converge rápido
   - 1-2 horas no total

3. **Valide frequently**: Monitore loss no novo dataset
   - Se converge rápido: bom sinal
   - Se não melhora: aumentar epochs ou learning rate

4. **Checkpoint intermediários**: Salve pesos após cada epoch
   - Time-LLM já faz isso automaticamente
   - Permite rollback se overfitting

---

### 📈 Exemplo Prático: ETTh1 → Traffic

**Cenário**: Treinou em ETTh1, quer prever tráfego viário

**Passo 1: Zero-shot test** (5 minutos)
```powershell
uv run python run_main_windows_fixed.py `
  --is_training 0 `
  --model TimeLLM `
  --data traffic `
  --pred_len 96 `
  --enc_in 862 `
  --dec_in 862 `
  --c_out 862 `
  --checkpoints ./checkpoints/seu_modelo_ETTh1/
```
Resultado esperado: MSE ~0.8 (ruim, degradação ~50%)

**Passo 2: Fine-tune** (4-8 horas)
```powershell
uv run python run_main_windows_fixed.py `
  --task_name long_term_forecast `
  --is_training 1 `
  --model_id traffic_finetuned `
  --model TimeLLM `
  --data traffic `
  --seq_len 336 `
  --pred_len 96 `
  --batch_size 16 `
  --train_epochs 20 `
  --learning_rate 0.00001 `
  --enc_in 862 `
  --dec_in 862 `
  --c_out 862 `
  --d_model 32 `
  --d_ff 128 `
  --llm_model GPT2 `
  --model_comment traffic-finetuned-20ep
```
Resultado esperado: MSE ~0.35 (melhor, 60% ganho)

---

### 💡 Insights Importantes

1. **Séries similares compartilham padrões básicos**
   - Temperatura vs Eletricidade: ambas têm sazonalidade diária
   - LLM aprende esses padrões em ETTh1

2. **Fine-tuning é MAIS eficiente que retrain do zero**
   - Fine-tune 10 epochs ≈ Retrain 2 epochs em novo dado
   - Razão: pesos já estão próximos do ótimo

3. **Learning rate é crítico em transfer learning**
   - `0.0001` (treinamento novo): grande passos
   - `0.00001` (fine-tune): pequenos ajustes
   - `0.000001` (very-low-data): micro-ajustes

4. **Domains **muito** diferentes precisam retrain**
   - Crypto prices ≠ Temperatura (padrões totalmente distintos)
   - Fine-tuning ajuda, mas não é suficiente

---

## �💡 DICAS IMPORTANTES

1. **Sempre use `--model_comment`** para nomear experimentos
2. **Comece pequeno**: teste com `--train_epochs 5`, depois aumente
3. **Monitorar loss**: loss deve decrescer ao longo das epochs
4. **Early stopping**: parâmetro `--patience` interrompe se não melhorar
5. **Learning rate**: comece com 0.0001, ajuste se convergir lentamente
6. **Batch size**: quanto maior, melhor (até a VRAM permitir)
7. **Model selection**: comece com DLinear (baseline), depois Autoformer, depois Time-LLM
8. **Dataset selection**: ETTh1 para teste, ETTh2/m1 para validação, Electricity para escala

---

**Última atualização**: 5 de março de 2026
