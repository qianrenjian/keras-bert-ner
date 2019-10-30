PRETRAINED_LM_DIR="/home1/liushaoweihua/pretrained_lm/albert_tiny_250k"
DATA_DIR="../data"
MODEL_DIR="../models"
OUTPUT_DIR="test_outputs"

python run_test.py \
    -test_data=${DATA_DIR}/test.txt \
    -model_path=${MODEL_DIR} \
    -model_name="ALBERT-IDCNN-CRF.h5" \
    -bert_vocab=${PRETRAINED_LM_DIR}/vocab.txt \
    -output_path=${OUTPUT_DIR} \
    -max_len=512 \
    -device_map="cpu"
