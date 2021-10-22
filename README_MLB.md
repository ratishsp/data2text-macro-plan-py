
# Training and Inference on the MLB Dataset

This repo is organized in two branches:

- `main`: macro-planning training and inference
- `summary_gen`: text generation training and inference

To leverage these two branches as best as possible, clone this repo twice, and
name the copies `macro_plan` and `doc_gen`. Then, set env. variables to the
corresponding path:

```bash
MACRO_PLAN=<path_to_macro_plan_repo>
DOC_GEN=<path_to_doc_gen_repo>
```

Do not forget to set the correct branch in each:

```bash
cd $MACRO_PLAN
git checkout main

cd $DOC_GEN
git checkout summary_gen
```

Note that sections I and III are for training, and sections II and IV are for inference.  
As such, you can pretty much run sections I and III in parallel if you wish (just run I.1 first).

### Download the MLB data

The input json files can be downloaded from https://drive.google.com/drive/folders/1G4iIE-02icAU2-5skvLlTEPWDQQj1ss4?usp=sharing

```bash
cd $MACRO_PLAN
mkdir -p json/mlb
# copy the json files into mlb folder
MLB_JSONS=$MACRO_PLAN/json/mlb/
```


## __I.__ Training the macro planning module

The following steps will guide you through the training phase of the macro planning module.
At all times, you should be in `$MACRO_PLAN`, on branch `main`

1: Create the macro planning dataset:

```
MLB=$MACRO_PLAN/mlb/
mkdir $MLB
ORDINAL_ADJECTIVE_MAP_FOLDER=$MACRO_PLAN/data

cd $MACRO_PLAN/scripts
python create_mlb_target_data.py -json_root $MLB_JSONS -output_folder $MLB -dataset_type train -ordinal_adjective_map_file \
${ORDINAL_ADJECTIVE_MAP_FOLDER}/traintokens-ordinaladjective-inning-identifier
python create_mlb_target_data.py -json_root $MLB_JSONS -output_folder $MLB -dataset_type valid -ordinal_adjective_map_file \
${ORDINAL_ADJECTIVE_MAP_FOLDER}/validtokens-ordinaladjective-inning-identifier
python create_mlb_target_data.py -json_root $MLB_JSONS -output_folder $MLB -dataset_type test -ordinal_adjective_map_file \
${ORDINAL_ADJECTIVE_MAP_FOLDER}/testtokens-ordinaladjective-inning-identifier
```

3: Run bpe tokenization
```
MLB_TOKENIZED=$MACRO_PLAN/mlb-tokenized
mkdir $MLB_TOKENIZED

TRAIN_FILE_1=$MLB/train.pp  
CODE=$MLB_TOKENIZED/code  
MERGES=6000

cd $MACRO_PLAN/tools
python learn_bpe.py -s ${MERGES} <$TRAIN_FILE_1 >$CODE  
  
TRAIN_BPE_FILE_1=$MLB_TOKENIZED/train.bpe.pp  
  
python apply_bpe.py -c $CODE --vocabulary-threshold 10 <$TRAIN_FILE_1 >$TRAIN_BPE_FILE_1  
  
VALID_FILE_1=$MLB/valid.pp  
VALID_BPE_FILE_1=$MLB_TOKENIZED/valid.bpe.pp  
  
python apply_bpe.py -c $CODE --vocabulary-threshold 10 <$VALID_FILE_1 >$VALID_BPE_FILE_1
```

4. Preprocess the data

```
PREPROCESS=$MACRO_PLAN/preprocess
mkdir $PREPROCESS

IDENTIFIER=mlb

cd $MACRO_PLAN
python preprocess.py -train_src $MLB_TOKENIZED/train.bpe.pp \
                     -train_tgt $MLB/train.macroplan \
                     -valid_src $MLB_TOKENIZED/valid.bpe.pp \
                     -valid_tgt $MLB/valid.macroplan \
                     -save_data $PREPROCESS/$IDENTIFIER \
                     -src_seq_length 1000000 -tgt_seq_length 1000000 -shard_size 1000
```

5. Train the model  
(Note that onmt can be buggy when training on a different gpu than 0. 
You can add `CUDA_VISIBLE_DEVICES=1` to train on gpu1, rather than changing
the `--gpu_ranks` flag)

```
MODELS=$MACRO_PLAN/models

mkdir $MODELS
mkdir $MODELS/$IDENTIFIER

python train.py -data $PREPROCESS/$IDENTIFIER \
                -save_model $MODELS/$IDENTIFIER/model \
                -encoder_type macroplan -layers 1 \
                -decoder_type pointer \
                -word_vec_size 512 \
                -rnn_size 512 \
                -seed 1234 \
                -optim adagrad \
                -learning_rate 0.15 \
                -adagrad_accumulator_init 0.1 \
                -content_selection_attn_hidden 64 \
                -report_every 10 \
                -batch_size 12 \
                -valid_batch_size 12 \
                -train_steps 30000 \
                -valid_steps 400 \
                -save_checkpoint_steps 400 \
                -start_decay_steps 30000 \
                -decay_steps 30000 \
                --early_stopping 10 \
                --early_stopping_criteria accuracy \
                -world_size 1 \
                -gpu_ranks 0 \
                --keep_checkpoint 15
```

## __II.__ Using the macro planning module for inference

1. Construct inference time plan input

```
DATASET_TYPE=<one_of_{valid|test}>
SUFFIX=infer

cd $MACRO_PLAN/scripts
python construct_inference_mlb_plan.py -json_root $MLB_JSONS \
                                        -output_folder $MLB \
                                        -dataset_type $DATASET_TYPE \
                                        -for_macroplanning \
                                        -suffix $SUFFIX
```

2. Apply bpe
```
CODE=$MLB_TOKENIZED/code  
VALID_FILE_1=$MLB/$DATASET_TYPE.$SUFFIX.pp
VALID_BPE_FILE_1=$MLB_TOKENIZED/$DATASET_TYPE.bpe.$SUFFIX.pp

cd $MACRO_PLAN/tools
python apply_bpe.py -c $CODE --vocabulary-threshold 10 <$VALID_FILE_1 >$VALID_BPE_FILE_1
```

3. Run inference for macro planning
```
MODEL_PATH=$MODELS/$IDENTIFIER/<best_checkpoint>
FILENAME=$DATASET_TYPE.bpe.$SUFFIX.pp
GEN=$MACRO_PLAN/gen
mkdir $GEN

cd $MACRO_PLAN
python translate.py -model $MODEL_PATH \
                    -src $MLB_TOKENIZED/${FILENAME} \
                    -output $GEN/$IDENTIFIER-beam5_gens.txt \
                    -batch_size 5 \
                    -max_length 20 \
                    -gpu 0 \
                    -min_length 12 \
                    -beam_size 10 \
                    -length_penalty avg \
                    -block_ngram_repeat 2 \
                    -block_repetitions 3
```

4. Run script to create paragraph plans conformant for generation

```
cd $MACRO_PLAN/scripts
python construct_inference_mlb_plan.py -json_root $MLB_JSONS \
                                        -output_folder $MLB \
                                        -dataset_type ${DATASET_TYPE} \
                                        -suffix stage2
``` 
Note: here we omit the ```for_macroplanning``` flag

5. Create the macro plan conformant for generation

```
SRC_FILE=$MLB/$DATASET_TYPE.stage2.pp

python create_macro_plan_from_index.py -src_file $SRC_FILE \
                                       -macro_plan_indices $GEN/$IDENTIFIER-beam5_gens.txt \
                                       -output_file $GEN/$IDENTIFIER-plan-summary-beam5_gens.txt
```

6. Strip ```</s>``` from the end of the plan (and save the results in `$DOC_GEN` so that 
it can be used by the summary generation model once it is trained)

```
BASE_MLB_PLAN=$GEN/$IDENTIFIER-plan-summary-beam5_gens.txt
BASE_OUTPUT_FILE=$DOC_GEN/mlb/${IDENTIFIER}-plan-beam5_gens.te
mkdir $DOC_GEN/mlb

cd $MACRO_PLAN/scripts
python convert_mlb_plan.py -mlb_plan ${BASE_MLB_PLAN} -output_file ${BASE_OUTPUT_FILE}
```

## __III.__ Train the NLG system, that generates summaries from plans

We now switch to `$DOC_GEN`. Make sure you are on the correct branch:  
(Note that we will reuse most of the env. variable names, so be careful)

```bash
cd $DOC_GEN
git checkout summary_gen
```


1. Script to create plan-to-summary generation dataset

```
MLB=$DOC_GEN/mlb
cd $MACRO_PLAN/scripts
ORDINAL_ADJECTIVE_MAP_FOLDER=$MACRO_PLAN/data

python create_mlb_target_data_gen.py -json_root $MLB_JSONS \
                                      -output_folder $MLB \
                                      -dataset_type train \
                                      -ordinal_adjective_map_file ${ORDINAL_ADJECTIVE_MAP_FOLDER}/traintokens-ordinaladjective-inning-identifier
                                      
python create_mlb_target_data_gen.py -json_root $MLB_JSONS \
                                      -output_folder $MLB \
                                      -dataset_type valid \
                                      -ordinal_adjective_map_file ${ORDINAL_ADJECTIVE_MAP_FOLDER}/validtokens-ordinaladjective-inning-identifier
                                      
python create_mlb_target_data_gen.py -json_root $MLB_JSONS \
                                      -output_folder $MLB \
                                      -dataset_type test \
                                      -ordinal_adjective_map_file ${ORDINAL_ADJECTIVE_MAP_FOLDER}/testtokens-ordinaladjective-inning-identifier
```

2. Run bpe tokenization

```
MERGES=16000
TRAIN_FILE_1=$MLB/train.te
TRAIN_FILE_2=$MLB/train.su
COMBINED=$MLB/combined

MLB_TOKENIZED=$DOC_GEN/mlb_tokenized
mkdir $MLB_TOKENIZED

CODE=$MLB_TOKENIZED/code
cat $TRAIN_FILE_1 $TRAIN_FILE_2 > $COMBINED

cd $DOC_GEN/tools
python learn_bpe.py -s 6000 < $COMBINED > $CODE 

TRAIN_BPE_FILE_1=$MLB_TOKENIZED/train.bpe.te
TRAIN_BPE_FILE_2=$MLB_TOKENIZED/train.bpe.su

python apply_bpe.py -c $CODE --vocabulary-threshold 10 < $TRAIN_FILE_1 > $TRAIN_BPE_FILE_1
python apply_bpe.py -c $CODE --vocabulary-threshold 10 < $TRAIN_FILE_2 > $TRAIN_BPE_FILE_2


VALID_FILE_1=$MLB/valid.te
VALID_FILE_2=$MLB/valid.su
VALID_BPE_FILE_1=$MLB_TOKENIZED/valid.bpe.te
VALID_BPE_FILE_2=$MLB_TOKENIZED/valid.bpe.su
python apply_bpe.py -c $CODE --vocabulary-threshold 10 < $VALID_FILE_1 > $VALID_BPE_FILE_1
python apply_bpe.py -c $CODE --vocabulary-threshold 10 < $VALID_FILE_2 > $VALID_BPE_FILE_2
```

3. Preprocess the plan-to-summary dataset
```
PREPROCESS=$DOC_GEN/preprocess
mkdir $PREPROCESS

IDENTIFIER=mlb

cd $DOC_GEN
python preprocess.py -train_src $MLB_TOKENIZED/train.bpe.te \
                     -train_tgt $MLB_TOKENIZED/train.bpe.su \
                     -valid_src $MLB_TOKENIZED/valid.bpe.te \
                     -valid_tgt $MLB_TOKENIZED/valid.bpe.su \
                     -save_data $PREPROCESS/$IDENTIFIER \
                     -src_seq_length 100000 \
                     -tgt_seq_length 100000 \
                     -dynamic_dict
```

4. Train summary-gen model 
```
MODELS=$DOC_GEN/models
mkdir $MODELS
mkdir $MODELS/$IDENTIFIER

cd $DOC_GEN
python train.py -data $PREPROCESS/$IDENTIFIER \
                -save_model $MODELS/$IDENTIFIER/model \
                -encoder_type brnn \
                -layers 1 \
                -word_vec_size 512 \
                -rnn_size 1024 \
                -seed 1234 \
                -optim adagrad \
                -learning_rate 0.15 \
                -adagrad_accumulator_init 0.1 \
                -report_every 10 \
                -batch_size 8 \
                -valid_batch_size 5 \
                -truncated_decoder 100 \
                -copy_attn \
                -reuse_copy_attn \
                -train_steps 30000 \
                -valid_steps 400 \
                -save_checkpoint_steps 400 \
                -start_decay_steps 30000 \
                -decay_steps 30000 \
                --early_stopping 10 \
                --early_stopping_criteria accuracy \
                -world_size 4 \
                -gpu_ranks 0 1 2 3 \
                --keep_checkpoint 15
```

## __IV.__ Generate a summary, using a plan from step II.

1. Apply bpe to plan obtained in Step II.6. 
```
FILENAME=$MLB/${IDENTIFIER}-plan-beam5_gens.te
BPE_FILENAME=$MLB_TOKENIZED/${IDENTIFIER}-plan.bpe-beam5_gens.te

cd $DOC_GEN/tools
python apply_bpe.py -c $CODE --vocabulary-threshold 10 <$FILENAME >$BPE_FILENAME
```

2. Run inference for the summary gen model
```

MODEL_PATH=$MODELS/$IDENTIFIER/<best_checkpoint>
GEN=$DOC_GEN/gen
mkdir $GEN

cd $DOC_GEN
python translate.py -model $MODEL_PATH \
                    -src $BPE_FILENAME \
                    -output $GEN/$IDENTIFIER-bpe_beam5_gens.txt \
                    -batch_size 10 \
                    -max_length 1000 \
                    -gpu 0 \
                    -min_length 200 \
                    -beam_size 5
```

3. Strip the ```@@@``` characters.
```
sed -r 's/(@@ )|(@@ ?$)| <segment>//g' $GEN/$IDENTIFIER-bpe_beam5_gens.txt >  $GEN/$IDENTIFIER-beam5_gens.txt
```


## Evaluation
Note: The IE evaluation has a preprocessing step of identifying mentions of innings. For this, we make use of GPT-2 to check if an ordinal adjective is an inning or not. The details are mentioned in the paper. You can install the required version of HuggingFace Transformers library using the below command.
```
pip install transformers==2.3
```
The detailed evaluation steps are below:

1. Compute BLEU
```
REFERENCE=$MLB/test.strip.su  # contains reference without <segment> tags
perl ~/mosesdecoder/scripts/generic/multi-bleu.perl  $REFERENCE < $GENS/$IDENTIFIER-beam5_gens.txt
```

2. Preprocessing for IE

```
cd $MACRO_PLAN/scripts
```
```
python add_segment_marker.py -input_file $GEN/$IDENTIFIER-beam5_gens.txt -output_file \  
$GEN/$IDENTIFIER-segment-beam5_gens.txt
```
```
python inning_prediction_offline.py -input_file $GEN/$IDENTIFIER-segment-beam5_gens.txt \
-output_file $GEN/$IDENTIFIER-inning-map-beam5_gens.txt
```

```
IE_ROOT=~/ie_root
python mlb_data_utils.py -mode prep_gen_data -gen_fi $GEN/$IDENTIFIER-segment-beam5_gens.txt \
-dict_pfx "$IE_ROOT/data/mlb-ie" -output_fi $DOC_GEN/transform_gen/$IDENTIFIER-beam5_gens.h5 \
-input_path "$IE_ROOT/json" \
-ordinal_inning_map_file $GEN/$IDENTIFIER-inning-map-beam5_gens.txt \
-test
```
4. Run RG evaluation
```
IE_ROOT=~/ie_root
th extractor.lua -gpuid  0 -datafile $IE_ROOT/data/mlb-ie.h5 \
-preddata $DOC_GEN/transform_gen/$IDENTIFIER-beam5_gens.h5 -dict_pfx \
"$IE_ROOT/data/mlb-ie" -just_eval -ignore_idx 14 -test
```
5. Run evaluation for non rg metrics 
```
python non_rg_metrics.py $DOC_GEN/transform_gen/test_inn_mlb-beam5_gens.h5-tuples.txt \
$DOC_GEN/transform_gen/$IDENTIFIER-beam5_gens.h5-tuples.txt
```



