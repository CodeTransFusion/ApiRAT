from simcse import SimCSE
import torch
import numpy as np
import os
import argparse
from sklearn.metrics.pairwise import cosine_similarity

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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

def calculate_similarity(src_lang, tgt_lang, source_sentences, target_sentences, model_path, output_dir, batch_size=30000):
    os.makedirs(output_dir, exist_ok=True)

    if len(source_sentences) > batch_size:
        source_batches = [source_sentences[i:i + batch_size] for i in range(0, len(source_sentences), batch_size)]
    else:
        source_batches = [source_sentences]

    model = SimCSE(model_path)

    key_vecs = model.encode(target_sentences, return_numpy=True, max_length=512)  # suppose M keys

    results = []
    for batch_id, source_batch in enumerate(source_batches):
        query_vecs = model.encode(source_batch, return_numpy=True, max_length=512)

        similarities = cosine_similarity(query_vecs, key_vecs)

        correct_matches = 0
        for idx, sim_row in enumerate(similarities):
            best_match_idx = np.argmax(sim_row)
            if best_match_idx == idx:
                correct_matches += 1
            results.append(
                f"Source Index: {idx}, Best Match Index: {best_match_idx}, Score: {sim_row[best_match_idx]:.4f}"
            )

        accuracy = correct_matches / len(source_batch)
        results.insert(0, f"UniXCoder Accuracy: {accuracy:.4f}\n")
        results.insert(1, "-" * 50)
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, f"{model_path}_{src_lang}_{tgt_lang}_results.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(results))

def main():
    parser = argparse.ArgumentParser(description="UniXcoder")
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
        "--output_dir",
        default="../output/codebert",
    )
    parser.add_argument(
        "--model_path",
        required=True,
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

    source_apis, source_indics, source_length = get_apis(source_path)
    target_apis, target_indics, target_length = get_apis(target_path)

    calculate_similarity(args.source_lang, args.target_lang, source_apis, target_apis, args.model_path, args.output_dir)


if __name__ == "__main__":
    main()
