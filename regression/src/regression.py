import numpy as np
import os
from sentence_transformers import SentenceTransformer
import pandas as pd
import tensorflow as tf
from metrics import Metrics
from density_balance import DensityBalance
from pdb import set_trace

def loadData(dataName="jirasoftware_filtered"):
    path = "../data/"
    df = pd.read_csv(path+dataName+".csv")
    return df

def process(dataName="jirasoftware_filtered", sensitive="is_internal"):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    data = loadData(dataName=dataName)
    embeddings = model.encode(data["text"].to_list())
    embedded = pd.DataFrame({"X": embeddings.tolist(), "Y": data["storypoint"], "A": data[sensitive], "split_mark": data["split_mark"]})
    return embedded


def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dim,)),

        # tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Dropout(0.3),
        #
        # tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Dropout(0.2),
        #
        # tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        # tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(1, activation="linear")
    ])

    model.compile(
        optimizer='adam',
        loss="mae",
        # loss=tf.keras.losses.Huber(delta=1.0),
        metrics=['mae']
    )

    return model

def train_and_test(dataname, treatment = "None"):

    data = process(dataname, "is_internal")
    train_x = np.array(data[data["split_mark"]!="test"]["X"].tolist())
    train_y = np.array(data[data["split_mark"]!="test"]["Y"].tolist())
    test_x = np.array(data[data["split_mark"] == "test"]["X"].tolist())
    test_y = np.array(data[data["split_mark"] == "test"]["Y"].tolist())

    if treatment=="FairReweighing":
        db = DensityBalance(model='Neighbor')
        weight = db.weight(np.array(data[data["split_mark"] != "test"]["A"]), train_y)
    else:
        weight = None


    model = build_model((train_x.shape[1]))

    checkpoint_path = "checkpoint/STD.keras"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=100, factor=0.3, min_lr=1e-6, verbose=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="loss", save_best_only=True,
                                                    save_weights_only=False, verbose=1)

    history = model.fit(
        train_x, train_y,
        sample_weight=weight,
        validation_data=(test_x, test_y),
        batch_size=32,
        epochs=600,
        callbacks=[reduce_lr, checkpoint],
        verbose=1
    )

    print("\nLoading best checkpoint model...")
    model = tf.keras.models.load_model(checkpoint_path)
    preds_test = model.predict(test_x).flatten()
    m_test = Metrics(test_y, preds_test)
    s = np.array(data[data["split_mark"] == "test"]["A"])
    N = len(test_y)
    alpha = 0.05
    selected1 = np.random.choice(N, size=N, replace=True)
    selected2 = np.random.choice(N, size=N, replace=True)
    comp_y = test_y[selected1] - test_y[selected2]
    comp_pred = preds_test[selected1] - preds_test[selected2]
    m_comp = Metrics(comp_y, comp_pred)
    pc, dc, pw, dw = m_comp.comparative_separation(s[selected1], s[selected2], stats=False)

    result_test = {"Data": dataname, "Treatment": treatment, "MAE": m_test.mae(),
                   "Pearson": m_test.pearsonr().statistic, "Spearman": m_test.spearmanr().statistic,
                   "Isep": m_test.Isep(s), "pc": pc, "dc": dc, "pw": pw, "dw": dw}
    return result_test

if __name__ == "__main__":
    np.random.seed(0)
    data = "jirasoftware_filtered"
    treatments = ["None", "FairReweighing"]
    results_test = []
    for treatment in treatments:
        for _ in range(1):
            result_test = train_and_test(data, treatment=treatment)
            results_test.append(result_test)
            df = pd.DataFrame(results_test)
            print(df)
            os.makedirs("../output", exist_ok=True)
            df.to_csv("../output/regression.csv", index=False)



