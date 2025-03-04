#!/bin/bash

WORKDIR=`pwd`
export PYTHONPATH=$WORKDIR;
export PYTHONIOENCODING=utf-8;

if [ "$#" -ne 5 ]; then
  echo "Usage: $0 <source_lang> <target_lang> <dataset> <model> <gpu_id>"
  echo "Supported models: bm25, codebert, unixcoder, graphcodebert, starencoder, text-embedding-3"
  exit 1
fi

SOURCE_LANG=$1
TARGET_LANG=$2
DATASET=$3
MODEL=$4
GPU_ID=$5

if [ "$MODEL" == "bm25" ]; then
  python3 retriever/bm25.py --source_lang "$SOURCE_LANG" --target_lang "$TARGET_LANG" --dataset "$DATASET"

elif [ "$MODEL" == "codebert" ]; then
  python3 retriever/codebert.py --source_lang "$SOURCE_LANG" --target_lang "$TARGET_LANG" --dataset "$DATASET" --gpu_id "$GPU_ID"

elif [ "$MODEL" == "unixcoder" ]; then
  python3 retriever/unixcoder.py --source_lang "$SOURCE_LANG" --target_lang "$TARGET_LANG" --dataset "$DATASET" --gpu_id "$GPU_ID"

elif [ "$MODEL" == "graphcodebert" ]; then
  python3 retriever/codebert.py --source_lang "$SOURCE_LANG" --target_lang "$TARGET_LANG" --dataset "$DATASET" --output_dir "../output/graphcodebert" --model_path "../models/graphcodebert" --gpu_id "$GPU_ID"

elif [ "$MODEL" == "starencoder" ]; then
  python3 retriever/starencoder.py --source_lang "$SOURCE_LANG" --target_lang "$TARGET_LANG" --dataset "$DATASET" --gpu_id "$GPU_ID"

elif [ "$MODEL" == "text-embedding-3" ]; then
  python3 retriever/text-embedding-v3.py --source_lang "$SOURCE_LANG" --target_lang "$TARGET_LANG" --dataset "$DATASET"

else
  echo "Error: Unsupported model '$MODEL'."
  echo "Supported models: bm25, codebert, unixcoder, graphcodebert, starencoder, text-embedding-3"
  exit 1
fi