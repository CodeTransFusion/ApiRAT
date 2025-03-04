#!/bin/bash

WORKDIR=`pwd`
export PYTHONPATH=$WORKDIR;
export PYTHONIOENCODING=utf-8;

if [ "$#" -ne 5 ]; then
  echo "Usage: $0 <source_lang> <target_lang> <dataset> <model> <gpu_id>"
  echo "Supported models: BM25, CodeBert, UniXCoder, GraphCodeBert, StarEncoder, Text-Embedding-3"
  exit 1
fi

MODEL=$1
DATASET=$2
SOURCE_LANG=$3
TARGET_LANG=$4
GPU_ID=$5

if [ "$MODEL" == "BM25" ]; then
  python3 retriever/bm25.py --source_lang "$SOURCE_LANG" --target_lang "$TARGET_LANG" --dataset "$DATASET"

elif [ "$MODEL" == "CodeBert" ]; then
  python3 retriever/codebert.py --source_lang "$SOURCE_LANG" --target_lang "$TARGET_LANG" --dataset "$DATASET" --gpu_id "$GPU_ID"

elif [ "$MODEL" == "UniXCoder" ]; then
  python3 retriever/unixcoder.py --source_lang "$SOURCE_LANG" --target_lang "$TARGET_LANG" --dataset "$DATASET" --gpu_id "$GPU_ID"

elif [ "$MODEL" == "GraphCodeBert" ]; then
  python3 retriever/codebert.py --source_lang "$SOURCE_LANG" --target_lang "$TARGET_LANG" --dataset "$DATASET" --output_dir "../output/graphcodebert" --model_path "../models/graphcodebert" --gpu_id "$GPU_ID"

elif [ "$MODEL" == "StarEncoder" ]; then
  python3 retriever/starencoder.py --source_lang "$SOURCE_LANG" --target_lang "$TARGET_LANG" --dataset "$DATASET" --gpu_id "$GPU_ID"

elif [ "$MODEL" == "Text-Embedding-3" ]; then
  python3 retriever/text-embedding-v3.py --source_lang "$SOURCE_LANG" --target_lang "$TARGET_LANG" --dataset "$DATASET"