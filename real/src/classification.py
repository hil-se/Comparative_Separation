from exp import Exp
import pandas as pd
import numpy as np
import time
from pdb import set_trace


if __name__ == "__main__":
    np.random.seed(0)
    r = 1000
    datasets = ["compas", "german"]
    treatments = ["None", "FairBalance", "Reweighing"]
    result = {"Treatment":[]}
    for treatment in treatments:
        result["Treatment"].append(treatment + "-" + "Separation")
        result["Treatment"].append(treatment + "-" + "Comparative Separation")
    for data in datasets:
        print(data)
        for treatment in treatments:
            exp = Exp(data, treatment=treatment)
            keys = []
            violate_sep = {}
            violate_comp = {}
            for protected in exp.A:
                violate_sep[protected] = 0
                violate_comp[protected] = 0
            for i in range(r):
                exp.random_exp(violate_sep, violate_comp)
            print(treatment)
            print("Separation")
            print(violate_sep)
            print("Comparative Separation")
            print(violate_comp)
            for protected in exp.A:
                key = data+"-"+protected
                if key not in result:
                    result[key] = []
                result[key].append(violate_sep[protected]/r)
                result[key].append(violate_comp[protected]/r)
    df = pd.DataFrame(result)
    df.to_csv("../result/classification.csv", index=False)
