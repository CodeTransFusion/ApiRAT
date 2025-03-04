import argparse
import os
from rank_bm25 import BM25Okapi, BM25L, BM25Plus
from nltk.tokenize import word_tokenize
import numpy as np

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

def bm25(src_lang, tgt_lang, source_txtdir, target_txtdir, output_dir):
    source_apis, source_indics, source_length = get_apis(source_txtdir)
    target_apis, target_indics, target_length = get_apis(target_txtdir)

    tokenized_src_corpus = [word_tokenize(doc) for doc in source_apis]
    tokenized_tgt_corpus = [word_tokenize(doc) for doc in target_apis]

    bm25_okapi = BM25Okapi(tokenized_tgt_corpus)
    bm25_l = BM25L(tokenized_tgt_corpus)
    bm25_plus = BM25Plus(tokenized_tgt_corpus)

    def calculate_accuracy(bm25_model, model_name):
        correct_matches = 0
        results = []

        for idx, src_tokens in enumerate(tokenized_src_corpus):
            scores = bm25_model.get_scores(src_tokens)
            best_match_idx = np.argmax(scores)
            if best_match_idx == idx:
                correct_matches += 1
            results.append(
                f"Source Index: {idx}, Best Match Index: {best_match_idx}, Score: {scores[best_match_idx]:.4f}"
            )

        accuracy = float(correct_matches / len(tokenized_src_corpus))
        results.insert(0, f"{model_name} Accuracy: {accuracy:.4f}\n")
        results.insert(1, "-" * 50)
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, f"{model_name}_{src_lang}_{tgt_lang}_results.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(results))

    calculate_accuracy(bm25_okapi, "BM25Okapi")
    calculate_accuracy(bm25_l, "BM25L")
    calculate_accuracy(bm25_plus, "BM25Plus")

def main():
    parser = argparse.ArgumentParser(description="BM25")
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
        default="../output/bm25",
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

    bm25(args.source_lang, args.target_lang, source_path, target_path, args.output_dir)

if __name__ == "__main__":
    main()