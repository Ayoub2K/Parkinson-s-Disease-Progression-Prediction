import tensorflow as tf
from tensorflow import keras
from keras import optimizers
from keras import losses
from keras import layers
from keras import Sequential

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def plot_udprs(patient_id: int, df: pd.DataFrame):
    df = df[df["patient_id"] == patient_id]
    fig, ax = plt.subplots(1, 1)

    ax.plot(df["visit_month"], df["updrs_1"], marker="o", color="blue", label="updrs_1", linestyle="-")
    ax.plot(df["visit_month"], df["updrs_2"], marker="o", color="red", label="updrs_2", linestyle="-")
    ax.plot(df["visit_month"], df["updrs_3"], marker="o", color="green", label="updrs_3", linestyle="-")
    ax.plot(df["visit_month"], df["updrs_4"], marker="o", color="orange", label="updrs_4", linestyle="-")

    ax.legend()
    plt.show()

def prepare_dataset(train_proteins: pd.DataFrame, train_peptides: pd.DataFrame):

    # Grouping 
    df_protein_grouped = train_proteins.groupby(["visit_id","UniProt"])["NPX"].mean().reset_index()
    df_peptide_grouped = train_peptides.groupby(["visit_id","Peptide"])["PeptideAbundance"].mean().reset_index()

    # Pivoting
    df_protein = df_protein_grouped.pivot(index="visit_id", columns="UniProt", values="NPX").rename_axis(columns=None).reset_index()
    df_peptide = df_peptide_grouped.pivot(index="visit_id", columns="Peptide", values="PeptideAbundance").rename_axis(columns=None).reset_index()
    
    # Merging
    pro_pep_df = df_protein.merge(df_peptide, on=["visit_id"], how="left")
    return pro_pep_df

def prepare_dataset_protein(train_proteins: pd.DataFrame):

    # Grouping 
    df_protein_grouped = train_proteins.groupby(["visit_id","UniProt"])["NPX"].mean().reset_index()

    # Pivoting
    df_protein = df_protein_grouped.pivot(index="visit_id", columns="UniProt", values="NPX").rename_axis(columns=None).reset_index()
    
    # Merging
    return df_protein

def prepare_dataset_peptide(train_peptides: pd.DataFrame):

    # Grouping 
    df_peptide_grouped = train_peptides.groupby(["visit_id","Peptide"])["PeptideAbundance"].mean().reset_index()

    # Pivoting
    df_peptide = df_peptide_grouped.pivot(index="visit_id", columns="Peptide", values="PeptideAbundance").rename_axis(columns=None).reset_index()
    
    # Merging
    return df_peptide

def prepare_features(df: pd.DataFrame):
    features = [i for i in df.columns if i not in ["visit_id"]]
    features.append("visit_month")
    return features

def kaggle_score_smape(test_label_values: list, prediction_label_values: list):
    return 100/len(test_label_values) * np.sum(2 * np.abs(prediction_label_values - test_label_values) / (np.abs(test_label_values) + np.abs(prediction_label_values)))

def train_test_data(dataset, test_ratio=0.30):
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]

if __name__ == "__main__":
    train_proteins = pd.read_csv("./train_proteins.csv")
    train_peptides = pd.read_csv("./train_peptides.csv")
    train_clinical = pd.read_csv("./train_clinical_data.csv")

    df_protein_pepitide = prepare_dataset(train_proteins=train_proteins, train_peptides=train_peptides)
    df_ml_dataset = df_protein_pepitide.merge(train_clinical, on=["visit_id"], how="left")
   

    FEATURES = prepare_features(df_protein_pepitide)
    feature_list = FEATURES.copy()


    df_ml_dataset = df_ml_dataset.dropna(subset=["updrs_1"])
    df_ml_dataset = df_ml_dataset.fillna(0)

    x = df_ml_dataset[feature_list]
    y = df_ml_dataset["updrs_1"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = Sequential(
    [
        layers.Dense(64, activation="tanh", name="layer1"),   
        layers.Dense(64, activation="tanh", name="layer2"),   
        layers.Dense(64, activation="tanh", name="layer3"),   
        layers.Dense(64, activation="tanh", name="layer4"),   
        layers.Dense(1),
    ])
    
    model.compile(
        optimizer=optimizers.Adam(),
        loss=losses.MeanSquaredError().name,
        metrics=[losses.MeanSquaredError().name])
    
    model.fit(
        x_train,
        y_train,
        epochs=10, 
        validation_data=(x_test, y_test)
        )

    print(model.evaluate(x_test, y_test))
    predictions = model.predict(x_test)
    smape = kaggle_score_smape(y_test.values.tolist(), predictions.flatten())

    print(f"smape: {smape}")


