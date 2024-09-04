import fire
import pandas as pd
import json
from tqdm import tqdm

def merge(input_path, output_path):
    data = []
    for i in tqdm(range(8)):
        with open(f'{input_path}/{i}.json', 'r') as f:
            data.extend(json.load(f))
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    fire.Fire(merge)
