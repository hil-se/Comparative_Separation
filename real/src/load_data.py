import pandas as pd

def load(data):
    # data: name of the dataset, can be {"compas", "german"}
    datasets = {"compas": load_compas, "german": load_german}
    if data not in datasets:
        raise Exception("Unknown dataset name.")
    return datasets[data]()


def load_compas():
    df = pd.read_csv("../data/compas-scores-two-years.csv")
    features_to_keep = ['sex', 'age', 'age_cat', 'race',
                        'juv_fel_count', 'juv_misd_count', 'juv_other_count',
                        'priors_count', 'c_charge_degree', 'c_charge_desc',
                        'two_year_recid']
    df = df[features_to_keep]
    # sensitive attribute names
    A = ["sex", "race"]
    df['sex'] = df['sex'].apply(lambda x: 1 if x == "Male" else 0)
    # discretize race: Caucasian vs. non-Caucasian
    df['race'] = df['race'].apply(lambda x: 1 if x == "Caucasian" else 0)
    # prefer 0 (no recid) as label 1
    df['two_year_recid'] = df['two_year_recid'].apply(lambda x: 1 if x==0 else 0)
    return df, A


def load_german():
    column_names = ['status', 'month', 'credit_history',
                    'purpose', 'credit_amount', 'savings', 'employment',
                    'investment_as_income_percentage', 'sex',
                    'other_debtors', 'residence_since', 'property', 'age',
                    'installment_plans', 'housing', 'number_of_credits',
                    'skill_level', 'people_liable_for', 'telephone',
                    'foreign_worker', 'credit']
    df = pd.read_csv("../data/german.data", sep=' ', header=None, names=column_names)
    # sensitive attribute names
    A = ["age", "sex"]
    # discretize age: x>25
    df["age"] = df["age"].apply(lambda x: 1 if x > 25 else 0)
    # transform personal_status into sex
    df["sex"] = df["sex"].apply(lambda x: 1 if x in {"A91", "A93", "A94"} else 0)
    # prefer 1 (good credit) as label 1
    df['credit'] = df['credit'].apply(lambda x: 1 if x==1 else 0)
    return df, A
