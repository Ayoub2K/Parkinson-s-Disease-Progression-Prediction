import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import tensorflow_decision_forests as tfdf
import math
from scipy import stats
import random

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
    df_protein_pepitide.info(verbose=True)
   
    # print(df_ml_dataset.head())
    print(len(df_ml_dataset))

    df_ml_dataset = df_ml_dataset.dropna(subset=["updrs_1"])

    FEATURES = prepare_features(df_protein_pepitide)

    MSE_SCORES = []
    LOWEST_MSE_FEATURES = []

    for i in range(100):
        feature_list = FEATURES.copy()
        feature_list = random.sample(feature_list, round(len(feature_list) / 8))
        feature_list.append("updrs_1")

        test_df, train_df = train_test_data(df_ml_dataset[feature_list])

        train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label="updrs_1", task = tfdf.keras.Task.REGRESSION)
        test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label="updrs_1", task = tfdf.keras.Task.REGRESSION)


        random_forest = tfdf.keras.GradientBoostedTreesModel(
            task = tfdf.keras.Task.REGRESSION, 
            verbose=0, 
            max_depth=10, 
            num_trees=300, 
            num_threads=32,
            growing_strategy="BEST_FIRST_GLOBAL", 
            max_num_nodes=10**6)
        
        random_forest.compile(metrics=["mse"])
        
        # Train the model.
        random_forest.fit(x=train_ds)

        inspector = random_forest.make_inspector()
        inspector.evaluation()
        evaluation = random_forest.evaluate(x=test_ds,return_dict=True)

        print(f"mse: {evaluation['mse']}")
        if len(MSE_SCORES) > 0 and evaluation['mse'] < min(MSE_SCORES):
            LOWEST_MSE_FEATURES = feature_list
            
        MSE_SCORES.append(evaluation['mse'])
        print(LOWEST_MSE_FEATURES)


        preds = random_forest.predict(test_ds)

        smape = kaggle_score_smape(test_df["updrs_1"].values.tolist(), preds.flatten())

        print(f"smape: {smape}")
    
    print(MSE_SCORES)
    print(f" lowest mse: {min(MSE_SCORES)}")
    print(LOWEST_MSE_FEATURES)

