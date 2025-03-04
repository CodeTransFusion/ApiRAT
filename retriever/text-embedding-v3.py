import os
import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

os.environ["AZURE_OPENAI_API_KEY"] = ""
os.environ["AZURE_OPENAI_ENDPOINT"] = ""
os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = ""

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")


def get_apis(txtdir):
    if isinstance(txtdir, str):
        txtdir = [txtdir]

    apis = []
    indics = []
    length = []

    for file_name in txtdir:
        with open(file_name, 'r', encoding='utf-8') as file:
            for line in file.readlines():
                arrs_split = line.split('<CODESPLIT>')
                apis.append(arrs_split[1])
                indics.append(arrs_split[0])
                lla = arrs_split[1].split(' ; ')
                length.append(len(lla) - 1)
        file.close()
    return apis, indics, length

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def generate_embeddings(text, model):
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def list_split(items, n):
    return [items[i:i + n] for i in range(0, len(items), n)]

def calculate_similarity(src_lang, tgt_lang, source_apis, target_apis, model, output_dir, batch_size=30000):
    os.makedirs(output_dir, exist_ok=True)

    print("Generating target embeddings...")
    key_vecs = [generate_embeddings(x, model=model) for x in target_apis]
    key_vecs = np.array(key_vecs)

    if len(source_apis) > batch_size:
        source_batches = list_split(source_apis, batch_size)
    else:
        source_batches = [source_apis]

    results = []
    acc_total = 0
    total_samples = 0

    for batch_id, source_batch in enumerate(source_batches):
        query_vecs = [generate_embeddings(x, model=model) for x in source_batch]
        query_vecs = np.array(query_vecs)

        similarities = cosine_similarity(query_vecs, key_vecs)

        max_sim_indices = np.argmax(similarities, axis=1)
        correct_matches = sum(1 for i, idx in enumerate(max_sim_indices) if i == idx)
        accuracy = correct_matches / len(source_batch)

        acc_total += correct_matches
        total_samples += len(source_batch)

        results.append(f"Batch {batch_id + 1} Accuracy: {accuracy:.4f}")
        for idx, sim_row in enumerate(similarities):
            best_match_idx = np.argmax(sim_row)
            results.append(
                f"Source Index: {idx}, Best Match Index: {best_match_idx}, Score: {sim_row[best_match_idx]:.4f}"
            )

    overall_accuracy = acc_total / total_samples
    results.insert(0, f"Text-embedding-v3 Accuracy: {overall_accuracy:.4f}\n")
    results.insert(1, "-" * 50)

    output_file = os.path.join(output_dir, f"azure_{src_lang}_{tgt_lang}_results.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(results))

def main():
    parser = argparse.ArgumentParser()
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
        default="../output/azure_openai",
    )
    parser.add_argument(
        "--model",
        default="text-embedding-3-large",
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

    calculate_similarity(args.source_lang, args.target_lang, source_apis, target_apis, args.model, args.output_dir)

if __name__ == "__main__":
    main()
