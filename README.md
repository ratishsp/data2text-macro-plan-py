# data2text-macro-plan-py
This repo contains code for [Data-to-text Generation with Macro Planning](https://arxiv.org/abs/2102.02723) (Ratish Puduppully and Mirella Lapata;  To appear: In Transactions of the Association for Computational Linguistics (TACL)); this code is based on an earlier (version 0.9.2) fork of [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).

## Requirements

All dependencies can be installed via:

```bash
pip install -r requirements.txt
```

## Code Details
The `main` branch contains code to generate macro plans from input verbalization. The code for training summary generation is in `summary_gen` branch.

The test outputs and trained models can be downloaded from the google drive link https://drive.google.com/drive/folders/1jJjq5IvuBKNLTAe7fuwlDYParrxpK-WD?usp=sharing

## Steps for training and inference for RotoWire dataset
1: Retokenize the RotoWire dataset:
The input json files can be downloaded from https://github.com/harvardnlp/boxscore-data
```
python retokenize_roto.py -json_root <folder containing json files> -output_folder <output folder> -dataset_type train/valid/test
```
The retokenized json files can also be downloaded from https://drive.google.com/drive/folders/1ECL-ffmonAeFtxzXE-pnw4zcWbvtdB0e?usp=sharing

2: Script to create macro planning dataset
```
JSON_ROOT=<Folder containing retokenized json files>
DATASET_TYPE=train/valid/test
python create_roto_target_data.py -json_root ${JSON_ROOT} \
-output_folder ${OUTPUT_FOLDER} -dataset_type ${DATASET_TYPE}
```
3: Run bpe tokenization
```
TRAIN_FILE_1=$BASE/rotowire/train.pp  
CODE=$BASE/rotowire-tokenized/code  
MERGES=2500
mkdir -p "${BASE}"/rotowire-tokenized  
python learn_bpe.py -s ${MERGES} <$TRAIN_FILE_1 >$CODE  
  
TRAIN_BPE_FILE_1=$BASE/rotowire-tokenized/train.bpe.pp  
  
python apply_bpe.py -c $CODE --glossaries "WON-[0-9]+" "LOST-[0-9]+" --vocabulary-threshold 10 <$TRAIN_FILE_1 >$TRAIN_BPE_FILE_1  
  
VALID_FILE_1=$BASE/rotowire/valid.pp  
VALID_BPE_FILE_1=$BASE/rotowire-tokenized/valid.bpe.pp  
  
python apply_bpe.py -c $CODE --glossaries "WON-[0-9]+" "LOST-[0-9]+" --vocabulary-threshold 10 <$VALID_FILE_1 >$VALID_BPE_FILE_1
```
4. Preprocess the data
```
python preprocess.py -train_src $BASE/rotowire-tokenized/train.bpe.pp -train_tgt \
$BASE/rotowire/train.macroplan -valid_src $BASE/rotowire-tokenized/valid.bpe.pp \
-valid_tgt $BASE/rotowire/valid.macroplan -save_data $BASE/preprocess/roto \
-src_seq_length 1000000 -tgt_seq_length 1000000 -shard_size 10000
```

5. Train the model
```
python train.py -data $BASE/preprocess/roto -save_model $BASE/gen_model/$IDENTIFIER/rotowire -encoder_type macroplan -layers 1 \
-decoder_type pointer \
-word_vec_size 384 -rnn_size 384 -seed 1234 -optim adagrad -learning_rate 0.15 -adagrad_accumulator_init 0.1 \
-content_selection_attn_hidden 64 \
-report_every 10 \
-batch_size 5 -valid_batch_size 5 -train_steps 30000 \
-valid_steps 300 -save_checkpoint_steps 300 -start_decay_steps 30000 -decay_steps 30000 --early_stopping 10 \
--early_stopping_criteria accuracy -world_size 1 -gpu_ranks 0 --keep_checkpoint 15
```
6. Construct inference time plan input
```
OUTPUT_FOLDER=$BASE/rotowire/
python construct_inference_roto_plan.py -json_root ${JSON_ROOT} \
-output_folder ${OUTPUT_FOLDER} -dataset_type ${DATASET_TYPE} -for_macroplanning \
-suffix infer
```

Apply bpe
```
PRED=infer
CODE=$BASE/rotowire-tokenized/code
VALID_FILE_1=$BASE/rotowire/valid.${PRED}.pp
VALID_BPE_FILE_1=$BASE/rotowire-tokenized/valid.bpe.${PRED}.pp

python apply_bpe.py -c $CODE --glossaries "WON-[0-9]+" "LOST-[0-9]+" --vocabulary-threshold 10 <$VALID_FILE_1 >$VALID_BPE_FILE_1
```

7. Run inference for macro planning
```
FILENAME=valid.bpe.infer.pp
mkdir $BASE/gen
python translate.py -model $MODEL_PATH -src $BASE/rotowire-tokenized/${FILENAME} \
-output $BASE/gen/roto_$IDENTIFIER-beam5_gens.txt -batch_size 10 -max_length 20 -gpu ${GPUID} \
-min_length 8 -beam_size 5 \
-length_penalty avg -block_ngram_repeat 2
```

8. Generate the macro plan from indices
```
SRC_FILE_NAME=valid.infer.pp
SRC_FILE=${BASE}/rotowire/${SRC_FILE_NAME}
python create_macro_plan_from_index.py -src_file ${SRC_FILE} \
-macro_plan_indices $BASE/gen/roto_$IDENTIFIER-beam5_gens.txt \
-output_file $BASE/gen/roto_$IDENTIFIER-plan-beam5_gens.txt
```

9. Run script to create paragraph plans conformant for generation
```
OUTPUT_FOLDER=${BASE}/rotowire/
python construct_inference_roto_plan.py -json_root ${JSON_ROOT} \  
-output_folder ${OUTPUT_FOLDER} -dataset_type ${DATASET_TYPE} -suffix stage2
``` 
Note: here we omit the ```for_macroplanning``` flag

10. Create the macro plan conformant for generation
```
SRC_FILE_NAME=valid.stage2.pp
SRC_FILE=${BASE}/rotowire/${SRC_FILE_NAME}
python create_macro_plan_from_index.py -src_file ${SRC_FILE} \
-macro_plan_indices $BASE/gen/roto_$IDENTIFIER-beam5_gens.txt \
-output_file $BASE/gen/roto_$IDENTIFIER-plan-summary-beam5_gens.txt
```
11. Add segment indices to macro plan
```
BASE_ROTO_PLAN=$BASE/gen/roto_$IDENTIFIER-plan-summary-beam5_gens.txt
BASE_OUTPUT_FILE=~/docgen/rotowire/roto_${IDENTIFIER}-plan-beam5_gens.te
python convert_roto_plan.py -roto_plan ${BASE_ROTO_PLAN} \  
-output_file ${BASE_OUTPUT_FILE}
```
12. Script to create plan-to-summary generation dataset
```
JSON_ROOT=<Folder containing retokenized json files>
DATASET_TYPE=train/valid/test
OUTPUT_FOLDER=~/docgen/rotowire
python create_roto_target_data_gen.py -json_root ${JSON_ROOT} \
-output_folder ${OUTPUT_FOLDER} -dataset_type ${DATASET_TYPE}
```
13. Run bpe tokenization
```
BASE=~/docgen
MERGES=6000
TRAIN_FILE_1=$BASE/rotowire/train.te
TRAIN_FILE_2=$BASE/rotowire/train.su
COMBINED=$BASE/rotowire/combined
CODE=$BASE/rotowire-tokenized/code
mkdir $BASE/rotowire-tokenized
cat $TRAIN_FILE_1 $TRAIN_FILE_2 > $COMBINED
python learn_bpe.py -s 6000 < $COMBINED > $CODE 

TRAIN_BPE_FILE_1=$BASE/rotowire-tokenized/train.bpe.te
TRAIN_BPE_FILE_2=$BASE/rotowire-tokenized/train.bpe.su

python apply_bpe.py -c $CODE --vocabulary-threshold 10 --glossaries "<segment[0-30]>"  < $TRAIN_FILE_1 > $TRAIN_BPE_FILE_1
python apply_bpe.py -c $CODE --vocabulary-threshold 10 --glossaries "<segment[0-30]>"  < $TRAIN_FILE_2 > $TRAIN_BPE_FILE_2


VALID_FILE_1=$BASE/rotowire/valid.te
VALID_FILE_2=$BASE/rotowire/valid.su
VALID_BPE_FILE_1=$BASE/rotowire-tokenized/valid.bpe.te
VALID_BPE_FILE_2=$BASE/rotowire-tokenized/valid.bpe.su
python apply_bpe.py -c $CODE --vocabulary-threshold 10 --glossaries "<segment[0-30]>"  < $VALID_FILE_1 > $VALID_BPE_FILE_1
python apply_bpe.py -c $CODE --vocabulary-threshold 10 --glossaries "<segment[0-30]>"  < $VALID_FILE_2 > $VALID_BPE_FILE_2
```
14. Preprocess the plan-to-summary dataset
```
git checkout summary_gen
BASE=~/docgen/
mkdir $BASE/preprocess
python preprocess.py -train_src $BASE/rotowire-tokenized/train.bpe.te -train_tgt $BASE/rotowire-tokenized/train.bpe.su -valid_src $BASE/rotowire-tokenized/valid.bpe.te -valid_tgt $BASE/rotowire-tokenized/valid.bpe.su -save_data $BASE/preprocess/roto -src_seq_length 10000 -tgt_seq_length 10000 -dynamic_dict
```

15. Train summary-gen model
```
python train.py -data $BASE/preprocess/roto -save_model $BASE/gen_model/$IDENTIFIER/roto -encoder_type brnn -layers 1 \
-word_vec_size 300 -rnn_size 600 -seed 1234 -optim adagrad -learning_rate 0.15 -adagrad_accumulator_init 0.1 \
-report_every 10 \
-batch_size 5 -valid_batch_size 5 -truncated_decoder 100 -copy_attn -reuse_copy_attn -train_steps 30000 \
-valid_steps 400 -save_checkpoint_steps 400 -start_decay_steps 2720 -decay_steps 680 --early_stopping 10 \
--early_stopping_criteria accuracy -world_size 1 -gpu_ranks 0 --keep_checkpoint 15 --learning_rate_decay 0.97
```
16. Apply bpe to plan obtained in Step 11
```
BASE=~/docgen
BASE_OUTPUT_FILE=~/docgen/rotowire/roto_${IDENTIFIER}-plan-beam5_gens.te
CODE=$BASE/rotowire-tokenized/code  # This BPE code is obtained in Step 14 below
VALID_FILE_1=${BASE_OUTPUT_FILE}  
VALID_BPE_FILE_1=$BASE/rotowire-tokenized/roto_${IDENTIFIER}-plan.bpe-beam5_gens.te    
python apply_bpe.py -c $CODE --vocabulary-threshold 10 <$VALID_FILE_1 >$VALID_BPE_FILE_1
```

17. Run inference for the macro plan
```
MODEL_PATH=~/docgen/gen_model/$IDENTIFIER/roto_step_15200.pt
FILENAME=roto_${IDENTIFIER}-plan.bpe-beam5_gens.te
python translate.py -model $MODEL_PATH -src $BASE/rotowire-tokenized/${FILENAME} \  
-output $BASE/gen/roto_$IDENTIFIER-bpe_beam5_gens.txt -batch_size 10 -max_length 850 -gpu ${GPUID} \  
-min_length 150 -beam_size 5
```
18. Strip the ```@@@``` characters
```
sed -r 's/(@@ )|(@@ ?$)//g; s/<segment[0-9]+> //g' $BASE/gen/roto_$IDENTIFIER-bpe_beam5_gens.txt \  
>  $BASE/gen/roto_$IDENTIFIER-beam5_gens.txt
```
19. Compute BLEU
```
REFERENCE=$BASE/rotowire/test.strip.su  # contains reference without <segment> tags
perl ~/mosesdecoder/scripts/generic/multi-bleu.perl  ${REFERENCE}\  
< $BASE/gen/roto_$IDENTIFIER-beam5_gens.txt
```

20. Preprocessing for IE
```
python data_utils.py -mode prep_gen_data -gen_fi $BASE/gen/roto_$IDENTIFIER-beam5_gens.txt \  
-dict_pfx "$IE_ROOT/data/roto-ie" -output_fi $BASE/transform_gen/roto_$IDENTIFIER-beam5_gens.h5 \  
-input_path "$IE_ROOT/json"
```
21. Run RG evaluation
```
th extractor.lua -gpuid  0 -datafile $IE_ROOT/data/roto-ie.h5 \  
-preddata $BASE/transform_gen/roto_$IDENTIFIER-beam5_gens.h5 -dict_pfx "$IE_ROOT/data/roto-ie" -just_eval \  
-folder_root $FOLDER_ROOT
```
22. Run evaluation for non rg metrics 
```
python non_rg_metrics.py $BASE/transform_gen/roto_val-beam5_gens.h5-tuples.txt \  
$BASE/transform_gen/roto_$IDENTIFIER-beam5_gens.h5-tuples.txt
```


