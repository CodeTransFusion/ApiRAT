## APIRAT: Integrating Multi-source API Knowledge for Enhanced Code Translation with LLMs
The repository for the paper "APIRAT: Integrating Multi-source API Knowledge for Enhanced Code Translation with LLMs".

### Artifacts

All artifacts in the paper can be found in artifacts.zip. Please download the `artifacts.zip` file from our [Google Drive](). After downloading, unzip the file into the repository's root directory.

### Installation
Follow the steps below to set up your environment.
Clone this repository:
```
git clone https://github.com/CodeTransFusion/ApiRAT/```

Create a virtual environment using the following command:
```
conda create -n apirat python=3.10
```

Activate it using the following command:
```
conda activate apirat
```

Verify your Python and pip versions:
```
python3 --version && pip3 --version
```

Install all required dependencies:
```
pip3 install -r requirements.txt
```

### Datasets

The datasets are organized as follows:

1. [CodeNet](https://github.com/IBM/Project_CodeNet)
2. [AVATAR](https://github.com/wasiahmad/AVATAR)
3. [CodeSearchNet](https://github.com/github/CodeSearchNet)
4. [HCLASM]()
5. [ApiRetrieval]()

We also provide direct download of the dataset. Please download the `datasets.zip` file from our [Google Drive](). After downloading datasets.zip, unzip it into the root directory of the repository. The directory structure should look like this:


```
ApiRAT
├── datasets
    ├── codenet
    ├── avatar
    ├── CodeSearchNet
    ├── HCLASM
    ├── ApiRetrieval
    └── README.md
├── ...
```
### Retrieval

To retrieve relevant API knowledge for a specific dataset, use the following command:
```
bash scripts/retrieve.sh <model_name> <dataset_name> <source_language> <target_language> <gpu_id>
```

Example:
```
bash scripts/retrieve.sh UniXCoder HCLASM Java Python 0
```

### Translation
To translate code snippets between programming languages, use the following script:
```
bash scripts/translate.sh <model_name> <dataset_name> <source_language> <target_language> <top_k> <top_p> <temperature> <gpu_id>
```

**Example 1: Translation with GPT** 

You can run the following command to translate all `Python -> Java` code snippets in `codenet` dataset with the `GPT-4o-mini` while top-k sampling is `k=50`, top-p sampling is `p=0.95`, and `temperature=0.7`:
```
bash scripts/translate.sh Gpt-3.5-turbo codenet Python Java 50 0.95 0.7 0
```


**Example 2: Translation with StarCoder**

You can execute the following command to translate all `Python -> Java` code snippets in `codenet` dataset with the `StarCoder` while top-k sampling is `k=50`, top-p sampling is `p=0.95`, and `temperature=0.2` on GPU `gpu_id=0`:
```
bash scripts/translate.sh StarCoder codenet Python Java 50 0.95 0.2 0
```

### Evaluation
Note: Evaluation requires Python 3.10 and Java 11. Please use `java -version` and `python3 --version` to verify that you are using the correct versions.


Run the following commands to evaluate the results:
```
bash scripts/test_retrieve.sh <source_language> <target_language> <model_name> <output_dir> <gpu_id>
bash scripts/test_avatar.sh <source_language> <target_language> <model_name> <output_dir> <gpu_id>
bash scripts/test_codenet.sh <source_language> <target_language> <model_name> <output_dir> <gpu_id>
```

**Example 1: Evaluate retrieve results** 
```
bash scripts/test_avatar.sh Python Java StarCoder reports/StarCoder 1
```

**Example 2: Evaluate translation results** 
```
bash scripts/test_avatar.sh Python Java StarCoder reports/StarCoder 1
```

