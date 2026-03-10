# train_gpt2.ps1
# Script para treinar Time-LLM com GPT-2 usando uv

Write-Host "================================"
Write-Host "Time-LLM Training com GPT-2"
Write-Host "================================"

# Variáveis de configuração
$MODEL_NAME = "TimeLLM-GPT2"
$TRAIN_EPOCHS = 50
$LEARNING_RATE = 0.0001
$BATCH_SIZE = 16
$D_MODEL = 32
$D_FF = 128

# Comando principal com uv
Write-Host "Iniciando treinamento..." -ForegroundColor Green

uv run python run_main.py `
  --task_name long_term_forecast `
  --is_training 1 `
  --root_path ./dataset/ETT-small/ `
  --data_path ETTh1.csv `
  --model_id ETTh1_512_96_gpt2 `
  --model TimeLLM `
  --data ETTh1 `
  --features M `
  --seq_len 512 `
  --label_len 48 `
  --pred_len 96 `
  --factor 3 `
  --enc_in 7 `
  --dec_in 7 `
  --c_out 7 `
  --des "Exp" `
  --itr 1 `
  --d_model $D_MODEL `
  --d_ff $D_FF `
  --batch_size $BATCH_SIZE `
  --learning_rate $LEARNING_RATE `
  --llm_model GPT2 `
  --llm_dim 768 `
  --llm_layers 12 `
  --train_epochs $TRAIN_EPOCHS `
  --model_comment $MODEL_NAME

Write-Host "Treinamento concluído!" -ForegroundColor Green