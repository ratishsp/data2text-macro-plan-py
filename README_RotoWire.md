# Training and Inference on the RotoWire Dataset

This repo is organized in two branches:

- `main`: macro-planning training and inference
- `summary_gen_roto`: text generation training and inference

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
git checkout summary_gen_roto
```

Note that sections I and III are for training, and sections II and IV are for inference.  
As such, you can pretty much run sections I and III in parallel if you wish (just run I.1 first).

### Download the RotoWire data

The input json files can be downloaded from https://github.com/harvardnlp/boxscore-data

```bash
cd $MACRO_PLAN
git clone https://github.com/harvardnlp/boxscore-data.git

cd boxscore-data
tar -xvf rotowire.tar.bz2

RAW_DATA=$MACRO_PLAN/boxscore-data/rotowire/
```


## __I.__ Training the macro planning module

The following steps will guide you through the training phase of the macro planning module.
At all times, you should be in `$MACRO_PLAN`, on branch `main`

1: Retokenize the RotoWire dataset (retokenized files are available [here][1]):
[1]: https://drive.google.com/drive/folders/1ECL-ffmonAeFtxzXE-pnw4zcWbvtdB0e?usp=sharing

```
ROTOWIRE_JSONS=$MACRO_PLAN/rotowire_jsons/
mkdir $ROTOWIRE_JSONS

cd $MACRO_PLAN/scripts
python retokenize_roto.py -json_root $RAW_DATA -output_folder $ROTOWIRE_JSONS -dataset_type train
python retokenize_roto.py -json_root $RAW_DATA -output_folder $ROTOWIRE_JSONS -dataset_type valid
python retokenize_roto.py -json_root $RAW_DATA -output_folder $ROTOWIRE_JSONS -dataset_type test
```

2: Create the macro planning dataset (files can be downloaded [here][2]):
[2]: https://drive.google.com/drive/folders/1b_BK6lfNuBf89GmUDtSu_8Mw_kP9t09z?usp=sharing

```
ROTOWIRE=$MACRO_PLAN/rotowire/
mkdir $ROTOWIRE

cd $MACRO_PLAN/scripts
python create_roto_target_data.py -json_root $ROTOWIRE_JSONS -output_folder $ROTOWIRE -dataset_type train
python create_roto_target_data.py -json_root $ROTOWIRE_JSONS -output_folder $ROTOWIRE -dataset_type valid
python create_roto_target_data.py -json_root $ROTOWIRE_JSONS -output_folder $ROTOWIRE -dataset_type test
```

3: Run bpe tokenization
```
ROTOWIRE_TOKENIZED=$MACRO_PLAN/rotowire-tokenized
mkdir $ROTOWIRE_TOKENIZED

TRAIN_FILE_1=$ROTOWIRE/train.pp  
CODE=$ROTOWIRE_TOKENIZED/code  
MERGES=2500

cd $MACRO_PLAN/tools
python learn_bpe.py -s ${MERGES} <$TRAIN_FILE_1 >$CODE  
  
TRAIN_BPE_FILE_1=$ROTOWIRE_TOKENIZED/train.bpe.pp  
  
python apply_bpe.py -c $CODE --glossaries "WON-[0-9]+" "LOST-[0-9]+" --vocabulary-threshold 10 <$TRAIN_FILE_1 >$TRAIN_BPE_FILE_1  
  
VALID_FILE_1=$ROTOWIRE/valid.pp  
VALID_BPE_FILE_1=$ROTOWIRE_TOKENIZED/valid.bpe.pp  
  
python apply_bpe.py -c $CODE --glossaries "WON-[0-9]+" "LOST-[0-9]+" --vocabulary-threshold 10 <$VALID_FILE_1 >$VALID_BPE_FILE_1
```

4. Preprocess the data

```
PREPROCESS=$MACRO_PLAN/preprocess
mkdir $PREPROCESS

IDENTIFIER=rotowire

cd $MACRO_PLAN
python preprocess.py -train_src $ROTOWIRE_TOKENIZED/train.bpe.pp \
                     -train_tgt $ROTOWIRE/train.macroplan \
                     -valid_src $ROTOWIRE_TOKENIZED/valid.bpe.pp \
                     -valid_tgt $ROTOWIRE/valid.macroplan \
                     -save_data $PREPROCESS/$IDENTIFIER \
                     -src_seq_length 1000000 -tgt_seq_length 1000000 -shard_size 10000
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
                -word_vec_size 384 \
                -rnn_size 384 \
                -seed 1234 \
                -optim adagrad \
                -learning_rate 0.15 \
                -adagrad_accumulator_init 0.1 \
                -content_selection_attn_hidden 64 \
                -report_every 10 \
                -batch_size 5 \
                -valid_batch_size 5 \
                -train_steps 30000 \
                -valid_steps 300 \
                -save_checkpoint_steps 300 \
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
python construct_inference_roto_plan.py -json_root $ROTOWIRE_JSONS \
                                        -output_folder $ROTOWIRE \
                                        -dataset_type $DATASET_TYPE \
                                        -for_macroplanning \
                                        -suffix $SUFFIX
```

2. Apply bpe
```
CODE=$ROTOWIRE_TOKENIZED/code  
VALID_FILE_1=$ROTOWIRE/$DATASET_TYPE.$SUFFIX.pp
VALID_BPE_FILE_1=$ROTOWIRE_TOKENIZED/$DATASET_TYPE.bpe.$SUFFIX.pp

cd $MACRO_PLAN/tools
python apply_bpe.py -c $CODE --glossaries "WON-[0-9]+" "LOST-[0-9]+" --vocabulary-threshold 10 <$VALID_FILE_1 >$VALID_BPE_FILE_1
```

3. Run inference for macro planning
```
MODEL_PATH=$MODELS/$IDENTIFIER/<best_checkpoint>  # In case of RotoWire, we choose the checkpoint which maximizes the CO on text generation.
FILENAME=$DATASET_TYPE.bpe.$SUFFIX.pp
GEN=$MACRO_PLAN/gen
mkdir $GEN

cd $MACRO_PLAN
python translate.py -model $MODEL_PATH \
                    -src $ROTOWIRE_TOKENIZED/${FILENAME} \
                    -output $GEN/$IDENTIFIER-beam5_gens.txt \
                    -batch_size 10 \
                    -max_length 20 \
                    -gpu 0 \
                    -min_length 8 \
                    -beam_size 5 \
                    -length_penalty avg \
                    -block_ngram_repeat 2
```


4. Run script to create paragraph plans conformant for generation

```
cd $MACRO_PLAN.scripts
python construct_inference_roto_plan.py -json_root $ROTOWIRE_JSONS \
                                        -output_folder $ROTOWIRE \
                                        -dataset_type ${DATASET_TYPE} \
                                        -suffix stage2
``` 
Note: here we omit the ```for_macroplanning``` flag

5. Create the macro plan conformant for generation

```
SRC_FILE=$ROTOWIRE/$DATASET_TYPE.stage2.pp

python create_macro_plan_from_index.py -src_file $SRC_FILE \
                                       -macro_plan_indices $GEN/$IDENTIFIER-beam5_gens.txt \
                                       -output_file $GEN/$IDENTIFIER-plan-summary-beam5_gens.txt
```

6. Add segment indices to macro plan (and save the results in `$DOC_GEN` so that 
it can be used by the summary generation model onces it is trained)

```
BASE_ROTO_PLAN=$GEN/$IDENTIFIER-plan-summary-beam5_gens.txt
BASE_OUTPUT_FILE=$DOC_GEN/rotowire/${IDENTIFIER}-plan-beam5_gens.te
mkdir $DOC_GEN/rotowire

cd $MACRO_PLAN/scripts
python convert_roto_plan.py -roto_plan ${BASE_ROTO_PLAN} -output_file ${BASE_OUTPUT_FILE}
```

## __III.__ Train the NLG system, that generates summaries from plans

We now switch to `$DOC_GEN`. Make sure you are on the correc branch:  
(Note that we will reuse most of the env. variable names, so be careful)

```bash
cd $DOC_GEN
git checkout summary_gen_roto
```


1. Script to create plan-to-summary generation dataset

```
ROTOWIRE=$DOC_GEN/rotowire
cd $MACRO_PLAN/scripts

python create_roto_target_data_gen.py -json_root $ROTOWIRE_JSONS \
                                      -output_folder $ROTOWIRE \
                                      -dataset_type train
                                      
python create_roto_target_data_gen.py -json_root $ROTOWIRE_JSONS \
                                      -output_folder $ROTOWIRE \
                                      -dataset_type valid
                                      
python create_roto_target_data_gen.py -json_root $ROTOWIRE_JSONS \
                                      -output_folder $ROTOWIRE \
                                      -dataset_type test
```

2. Run bpe tokenization

```
MERGES=6000
TRAIN_FILE_1=$ROTOWIRE/train.te
TRAIN_FILE_2=$ROTOWIRE/train.su
COMBINED=$ROTOWIRE/combined

ROTOWIRE_TOKENIZED=$DOC_GEN/rotowire_tokenized
mkdir $ROTOWIRE_TOKENIZED

CODE=$ROTOWIRE_TOKENIZED/code
cat $TRAIN_FILE_1 $TRAIN_FILE_2 > $COMBINED

cd $DOC_GEN/tools
python learn_bpe.py -s 6000 < $COMBINED > $CODE 

TRAIN_BPE_FILE_1=$ROTOWIRE_TOKENIZED/train.bpe.te
TRAIN_BPE_FILE_2=$ROTOWIRE_TOKENIZED/train.bpe.su

python apply_bpe.py -c $CODE --vocabulary-threshold 10 --glossaries "<segment[0-30]>"  < $TRAIN_FILE_1 > $TRAIN_BPE_FILE_1
python apply_bpe.py -c $CODE --vocabulary-threshold 10 --glossaries "<segment[0-30]>"  < $TRAIN_FILE_2 > $TRAIN_BPE_FILE_2


VALID_FILE_1=$ROTOWIRE/valid.te
VALID_FILE_2=$ROTOWIRE/valid.su
VALID_BPE_FILE_1=$ROTOWIRE_TOKENIZED/valid.bpe.te
VALID_BPE_FILE_2=$ROTOWIRE_TOKENIZED/valid.bpe.su
python apply_bpe.py -c $CODE --vocabulary-threshold 10 --glossaries "<segment[0-30]>"  < $VALID_FILE_1 > $VALID_BPE_FILE_1
python apply_bpe.py -c $CODE --vocabulary-threshold 10 --glossaries "<segment[0-30]>"  < $VALID_FILE_2 > $VALID_BPE_FILE_2
```

3. Preprocess the plan-to-summary dataset
```
PREPROCESS=$DOC_GEN/preprocess
mkdir $PREPROCESS

IDENTIFIER=rotowire

cd $DOC_GEN
python preprocess.py -train_src $ROTOWIRE_TOKENIZED/train.bpe.te \
                     -train_tgt $ROTOWIRE_TOKENIZED/train.bpe.su \
                     -valid_src $ROTOWIRE_TOKENIZED/valid.bpe.te \
                     -valid_tgt $ROTOWIRE_TOKENIZED/valid.bpe.su \
                     -save_data $PREPROCESS/$IDENTIFIER \
                     -src_seq_length 10000 \
                     -tgt_seq_length 10000 \
                     -dynamic_dict
```

4. Train summary-gen model  
(Again, prefer to use `CUDA_VISIBLE_DEVICE=1` if you want to use another gpu)

```
MODELS=$DOC_GEN/models
mkdir $MODELS
mkdir $MODELS/$IDENTIFIER

cd $DOC_GEN
python train.py -data $PREPROCESS/$IDENTIFIER \
                -save_model $MODELS/$IDENTIFIER/model \
                -encoder_type brnn \
                -layers 1 \
                -word_vec_size 300 \
                -rnn_size 600 \
                -seed 1234 \
                -optim adagrad \
                -learning_rate 0.15 \
                -adagrad_accumulator_init 0.1 \
                -report_every 10 \
                -batch_size 5 \
                -valid_batch_size 5 \
                -truncated_decoder 100 \
                -copy_attn \
                -reuse_copy_attn \
                -train_steps 30000 \
                -valid_steps 400 \
                -save_checkpoint_steps 400 \
                -start_decay_steps 2720 \
                -decay_steps 680 \
                --early_stopping 10 \
                --early_stopping_criteria accuracy \
                -world_size 1 \
                -gpu_ranks 0 \
                --keep_checkpoint 15 \
                --learning_rate_decay 0.97
```

## __IV.__ Generate a summary, using a plan from step II.

1. Apply bpe to plan obtained in Step II.6. 
```
FILENAME=$ROTOWIRE/${IDENTIFIER}-plan-beam5_gens.te
BPE_FILENAME=$ROTOWIRE_TOKENIZED/${IDENTIFIER}-plan.bpe-beam5_gens.te

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
                    -max_length 850 \
                    -gpu 0 \
                    -min_length 150 \
                    -beam_size 5
```

3. Strip the ```@@@``` characters and <segmentN> tags.
```
sed -r 's/(@@ )|(@@ ?$)//g; s/<segment[0-9]+> //g' $GEN/$IDENTIFIER-bpe_beam5_gens.txt >  $GEN/$IDENTIFIER-beam5_gens.txt
```


## Evaluation

1. Compute BLEU
```
REFERENCE=$ROTOWIRE/test.strip.su  # contains reference without <segment> tags
perl ~/mosesdecoder/scripts/generic/multi-bleu.perl  $REFERENCE < $GENS/$IDENTIFIER-beam5_gens.txt
```

2. Preprocessing for IE
In this paper, we make use of IE including duplicate records too. 
The branch ```include_all_records``` of https://github.com/ratishsp/data2text-1 
contains the relevant code changes. 
```
IE_ROOT=~/ie_root
git checkout include_all_records
python data_utils.py -mode prep_gen_data -gen_fi $GEN/$IDENTIFIER-beam5_gens.txt \  
-dict_pfx "$IE_ROOT/data/roto-ie" -output_fi $DOC_GEN/transform_gen/$IDENTIFIER-beam5_gens.h5 \  
-input_path "$IE_ROOT/json"
```
3. Run RG evaluation
```
IE_ROOT=~/ie_root
git checkout include_all_records
th extractor.lua -gpuid  0 -datafile $IE_ROOT/data/roto-ie.h5 \  
-preddata $DOC_GEN/transform_gen/$IDENTIFIER-beam5_gens.h5 -dict_pfx "$IE_ROOT/data/roto-ie" -just_eval  
```
4. Run evaluation for non rg metrics 
```
git checkout include_all_records
python non_rg_metrics.py $DOC_GEN/transform_gen/roto_val-beam5_gens.h5-tuples.txt \  
$DOC_GEN/transform_gen/$IDENTIFIER-beam5_gens.h5-tuples.txt
```


