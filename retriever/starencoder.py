from transformers import AutoTokenizer, RobertaModel, BertForPreTraining
import os
from tqdm import tqdm
from tqdm.contrib import tenumerate
import argparse
import torch
import json
from typing import *
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def pool_and_normalize(
    features_sequence: torch.Tensor,
    attention_masks: torch.Tensor,
    return_norms: bool = False,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Temporal ooling of sequences of vectors and projection onto the unit sphere.

    Args:
        features_sequence (torch.Tensor): Inpute features with shape [B, T, F].
        attention_masks (torch.Tensor): Pooling masks with shape [B, T, F].
        return_norms (bool, optional): Whether to additionally return the norms. Defaults to False.

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]: Pooled and normalized vectors with shape [B, F].
    """

    pooled_embeddings = pooling(features_sequence, attention_masks)
    embedding_norms = pooled_embeddings.norm(dim=1)

    normalizing_factor = torch.where(  # Only normalize embeddings with norm > 1.0.
        embedding_norms > 1.0, embedding_norms, torch.ones_like(embedding_norms)
    )

    pooled_normalized_embeddings = pooled_embeddings / normalizing_factor[:, None]

    if return_norms:
        return pooled_normalized_embeddings, embedding_norms
    else:
        return pooled_normalized_embeddings


def pooling(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pools a batch of vector sequences into a batch of vector global representations.
    It does so by taking the last vector in the sequence, as indicated by the mask.

    Args:
        x (torch.Tensor): Batch of vector sequences with shape [B, T, F].
        mask (torch.Tensor): Batch of masks with shape [B, T].

    Returns:
        torch.Tensor: Pooled version of the input batch with shape [B, F].
    """

    # eos_token
    eos_idx = mask.sum(1) - 1
    batch_idx = torch.arange(len(eos_idx), device=x.device)

    mu = x[batch_idx, eos_idx, :]

    return mu

    # pooling
    # x_masked = x * mask.unsqueeze(-1).float()
    # sum_vector = x_masked.sum(dim=1)
    # valid_element_count = mask.sum(dim=1, keepdim=True)
    # average_vector = sum_vector / valid_element_count
    # return average_vector

def set_device(inputs: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    output_data = {}
    for k, v in inputs.items():
        output_data[k] = v.to(device)

    return output_data

def get_embedding(text, tokenizer, model, device):
    inputs = tokenizer(
        f"{text}<|endoftext|>",
        padding="longest",
        max_length=1024,
        truncation=True,
        return_tensors="pt"
        ).to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    print(f"words;{tokens}")
    print(f"input_ids:{inputs['input_ids']}")
    print(f"att_mask:{inputs['attention_mask']}")
    outputs = model(**set_device(inputs, device))
    embedding = pool_and_normalize(outputs.hidden_states[-1], inputs.attention_mask)
    return embedding[0].cpu().detach().numpy().tolist()

def get_apis(txtdir):
    if isinstance(txtdir, str):
        txtdir = [txtdir]

    data_lines = []
    apis = []
    indics = []
    length = []

    for file_name in txtdir:
        with open(file_name, 'r', encoding='utf-8') as file:
            for line in file.readlines():
                arrs_split = line.split('<CODESPLIT>')
                data_lines.append(line)
                apis.append(arrs_split[1])
                indics.append(arrs_split[0])
                lla = arrs_split[1].split(' ; ')
                length.append(len(lla)-1)
        file.close()
    return apis, indics, length


def calculate_similarity(source_apis, target_apis, src_lang, tgt_lang, tokenizer, model, device, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    source_embeddings = []
    for t in tqdm(source_apis):
        source_embeddings.append(get_embedding(t, tokenizer, model, device))

    target_embeddings = []
    for t in tqdm(target_apis):
        target_embeddings.append(get_embedding(t, tokenizer, model, device))

    similarities = cosine_similarity(source_embeddings, target_embeddings)

    results = []
    correct_matches = 0
    for idx, sim_row in enumerate(similarities):
        best_match_idx = np.argmax(sim_row)
        if best_match_idx == idx:
            correct_matches += 1
        results.append(
            f"Source Index: {idx}, Best Match Index: {best_match_idx}, Score: {sim_row[best_match_idx]:.4f}"
        )

    accuracy = correct_matches / len(source_embeddings)
    results.insert(0, f"UniXCoder Accuracy: {accuracy:.4f}\n")
    results.insert(1, "-" * 50)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"unixcoder_{src_lang}_{tgt_lang}_results.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(results))

def main():
    parser = argparse.ArgumentParser(description="API Similarity Calculation")
    parser.add_argument(
        "--source_lang",
        required=True,
        help="Source language (Python or Java)",
    )
    parser.add_argument(
        "--target_lang",
        required=True,
        help="Target language (Java or Python)",
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the pretrained model."
    )
    parser.add_argument(
        "--output_dir",
        default="./output/starencoder",
        help="Directory to save the output results."
    )
    args = parser.parse_args()

    if args.source_lang == 'Python' and args.target_lang == 'Java':
        source_path = '../datasets/HCLASM/HCLASM-Python.txt'
        target_path = '../datasets/HCLASM/HCLASM-Java.txt'
    elif args.source_lang == 'Java' and args.target_lang == 'Python':
        source_path = '../datasets/HCLASM/HCLASM-Java.txt'
        target_path = '../datasets/HCLASM/HCLASM-Python.txt'
    else:
        raise ValueError("Unsupported language pair: {} -> {}".format(args.source_lang, args.target_lang))

    MASK_TOKEN = "<mask>"
    SEPARATOR_TOKEN = "<|endoftext|>"
    PAD_TOKEN = "<pad>"
    CLS_TOKEN = "<|endoftext|>"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = "<pad>"
    model = BertForPreTraining.from_pretrained(args.model_path, output_hidden_states=True)
    model.to(device)

    source_apis, source_indics, src_len = get_apis(source_path)
    target_apis, target_indics, tgt_len = get_apis(target_path)

    calculate_similarity(source_apis, target_apis, args.source_lang, args.target_lang, tokenizer, model, device, args.output_dir)


if __name__ == "__main__":
    main()