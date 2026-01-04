Our code is implemented within Fairseq and has already been integrated into Fairseq.
Preprocess:
python preprocess/prep_mustc_data_joint.py \
  --tgt-lang ${LANG} --data-root ${MUSTC_ROOT} \
  --task st --yaml-filename config_st_raw_joint.yaml \
  --vocab-type unigram --vocab-size 10000 \
  --use-audio-input

MT预训练
python fairseq/fairseq_cli/train.py ${DATA} \
    --no-progress-bar --fp16 --memory-efficient-fp16 \
    --arch transformer --share-decoder-input-output-embed \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 8192 --max-update 250000 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 7e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.1 --weight-decay 0.0 \
    --seed 1 --update-freq 1 \
    --log-interval 10 \
    --validate-interval 1 --save-interval 1 \
    --save-interval-updates 1000 --keep-interval-updates 10 \
    --save-dir ${MT_SAVE_DIR} --tensorboard-logdir ${LOG_DIR} \
    --skip-invalid-size-inputs-valid-test \
    --ddp-backend=legacy_ddp \
    |& tee -a ${LOG_DIR}/train.log

ST训练
prob=0.2
kl_weight=2
python fairseq/fairseq_cli/train.py ${MUSTC_ROOT}/en-${LANG} \
    --no-progress-bar --fp16 --memory-efficient-fp16 \
    --config-yaml config_st_raw_joint.yaml --train-subset train_st_raw_joint --valid-subset dev_st_raw \
    --save-dir ${ST_SAVE_DIR} \
    --max-tokens 2000000 --max-source-positions 900000 --batch-size 32 --max-target-positions 1024  --max-tokens-text 4096 \
    --max-update 60000 --log-interval 10 --num-workers 4 \
    --task speech_and_text --criterion label_smoothed_cross_entropy_otmix \
    --use-kl --kl-st --kl-mt --kl-weight ${kl_weight} \
    --use-ot --ot-type L2 --ot-position encoder_out --ot-window --ot-window-size 10 --mix-prob ${prob} \
    --label-smoothing 0.1 --report-accuracy \
    --arch hubert_ot_post --layernorm-embedding --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr 1e-4 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
    --hubert-model-path ${HUBERT_MODEL} --mt-model-path ${MT_MODEL} \
    --clip-norm 0.0 --seed 1 --update-freq 2 \
    --tensorboard-logdir ${LOG_DIR} \
    --ddp-backend=legacy_ddp \
    --skip-invalid-size-inputs-valid-test \
    |& tee -a $LOG_DIR/train.log
