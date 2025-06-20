source .env
torchserve --start --model-store artefacts/models/model_store --models $MODEL_NAME=$MODEL_NAME.mar --ncs --disable-token-auth