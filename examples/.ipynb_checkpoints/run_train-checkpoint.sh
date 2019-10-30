BERT_BASE_DIR="/home1/liushaoweihua/pretrained_lm/roberta_chinese"
DATA_DIR="data"
OUTPUT_DIR="models"

python run_train.py \
    -train_data=${DATA_DIR}/train.txt \
    -dev_data=${DATA_DIR}/train.txt \
    -save_path=${OUTPUT_DIR} \
    -bert_config=${BERT_BASE_DIR}/bert_config.json \
    -bert_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt \
    -bert_vocab=${BERT_BASE_DIR}/vocab.txt \
    -device_map="0" \
    -hard_epochs=5 \
    -max_epochs=256 \
    -early_stop_patience=3 \
    -reduce_lr_patience=2 \
    -reduce_lr_factor=0.5 \
    -batch_size=64 \
    -max_len=64 \
    -learning_rate=5e-6 \
    -model_type="cnn" \
    -cell_type="idcnn" \
    -cnn_filters=128 \
    -cnn_kernel_size=3 \
    -cnn_blocks=4 \
    -dropout_rate=0.0