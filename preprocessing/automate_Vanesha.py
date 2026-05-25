import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    print(f"[INFO] Hapus duplikat: {before - len(df)} baris dihapus")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols_missing = ['ca', 'thal']
    for col in cat_cols_missing:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
            print(f"[INFO] Isi missing values kolom '{col}' dengan modus: {df[col].mode()[0]}")

    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            print(f"[INFO] Isi missing values kolom '{col}' dengan median")

    return df


def remove_outliers_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    before = len(df)
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    print(f"[INFO] Hapus outlier: {before - len(df)} baris dihapus")
    return df


def encode_categorical(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    print(f"[INFO] Encoding selesai. Shape baru: {df.shape}")
    return df


def normalize_features(df: pd.DataFrame, num_cols: list) -> pd.DataFrame:
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    print(f"[INFO] Normalisasi selesai untuk: {num_cols}")
    return df


def preprocess(input_path: str, output_path: str) -> pd.DataFrame:
    df = load_data(input_path)
    df = remove_duplicates(df)
    df = handle_missing_values(df)

    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    cat_cols = ['cp', 'restecg', 'slope', 'ca', 'thal']

    df = remove_outliers_iqr(df, num_cols)
    df = encode_categorical(df, cat_cols)
    df = normalize_features(df, num_cols)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Dataset siap disimpan ke: {output_path}")
    return df


if __name__ == "__main__":
    preprocess(
        input_path="heart_disease_raw.csv",
        output_path="preprocessing/heart_disease_preprocessing.csv"
    )