
import os,sys,json
root = os.path.abspath(os.path.join(os.getcwd(), '.'))
sys.path.append(root)
from utils import utils

if __name__ == "__main__": 
    clean_eval, full_eval = utils.build_eval_sets("dataset/labelled_eval_dataset.json")

    with open("dataset/clean_eval.json", "w") as f:
        json.dump(clean_eval, f, indent=2)

    with open("dataset/full_eval.json", "w") as f:
        json.dump(full_eval, f, indent=2)

    print("Done writing json for full_eval and clean_eval")