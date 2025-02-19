"""
Convert SimCSE's checkpoints to Huggingface style.
"""

import argparse
import torch
import os
import json


def main():
    path = ''

    path_list=[]
    for file in os.listdir(path):
        path_str={}
        d = os.path.join(path, file)
        if os.path.isdir(d):
            str=d.split("-")
            path_str['path'] = d
            path_str['checkpoint'] = int(str[2])
            path_list.append(path_str)
    path_list=sorted(path_list,key=lambda x:x['checkpoint'])
    print(path_list)

    #do transformation
    for i in path_list:
        model_path=i['path']
        check_point=i['checkpoint']

        print("SimCSE checkpoint -> Huggingface checkpoint for {}".format(model_path))

        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location=torch.device("cpu"))
        new_state_dict = {}
        for key, param in state_dict.items():
            # Replace "mlp" to "pooler"
            if "mlp" in key:
                key = key.replace("mlp", "pooler")

            # Delete "bert" or "roberta" prefix
            if "bert." in key:
                key = key.replace("bert.", "")
            if "roberta." in key:
                key = key.replace("roberta.", "")

            new_state_dict[key] = param

        torch.save(new_state_dict, os.path.join(model_path, "pytorch_model.bin"))

        # Change architectures in config.json
        config = json.load(open(os.path.join(model_path, "config.json")))
        for i in range(len(config["architectures"])):
            config["architectures"][i] = config["architectures"][i].replace("ForCL", "Model")
        json.dump(config, open(os.path.join(model_path, "config.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
