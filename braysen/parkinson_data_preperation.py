import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
from scipy import stats

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

def prepare_features(df: pd.DataFrame):
    features = [i for i in df.columns if i not in ["visit_id"]]
    features.append("visit_month")



if __name__ == "__main__":
    train_proteins = pd.read_csv("./train_proteins.csv")
    train_peptides = pd.read_csv("./train_peptides.csv")
    train_clinical = pd.read_csv("./train_clinical_data.csv")

    df_protein_pepitide = prepare_dataset(train_proteins=train_proteins, train_peptides=train_peptides)
    df_ml_dataset = df_protein_pepitide.merge(train_clinical, on=["visit_id"], how="left")
    df_ml_dataset.info(verbose=True)
    print(df_ml_dataset.head())

    FEATURES = prepare_features(df_protein_pepitide)

    plot_udprs(patient_id=4172, df=train_clinical)

