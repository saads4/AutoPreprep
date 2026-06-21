from pathlib import Path

import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer


BASE_DIR = Path(__file__).parent


class AutoPreprocessor:
    def __init__(self):
        self.preprocessor = None
        self.report_ = None

    def fit(self, data):
        SIMILARITY_THRESHOLD = 0.5
        DEFAULT_ENCODING = "onehot"

        ###output cols
        mean_imputer = []
        median_imputer = []
        mode_imputer = []
        label_encoder = []
        onehot_encoder = []
        ordinal_encoder =[]
        standard_scaler = []
        robust_scaler = []
        no_scaling = []
        drop_cols = []

        model = SentenceTransformer("all-MiniLM-L6-v2")
        client = chromadb.PersistentClient(
            path=str(BASE_DIR / "chroma_db")
        )
        collection = client.get_collection("rag_collection")

        for col in tqdm(data.columns, desc="Classifying Columns"):
            if data[col].isnull().mean() > 0.5:
                drop_cols.append(col)
                continue
            if data[col].dtype == object:
                if data[col].isnull().sum() > 0:
                    mode_imputer.append(col)
                if data[col].nunique() == 2:
                    label_encoder.append(col)
                else:
                    onehot_encoder.append(col)
            else:
                if data[col].isnull().sum() > 0:
                    if abs(data[col].skew()) > 1:
                        median_imputer.append(col)
                    else:
                        mean_imputer.append(col)
                if data[col].nunique()/len(data) < 0.05:
                    no_scaling.append(col)
                elif abs(data[col].skew()) > 1:
                    robust_scaler.append(col)
                else:
                    standard_scaler.append(col)

        for j in tqdm(onehot_encoder, desc="RAG Encoding Prediction"):
            values = pd.Series(data[j].dropna().unique()).astype(str).str.lower().tolist()
            query = " ".join(values)
            query_embedding = model.encode(query)
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=3
            )
            r_documents = results["documents"][0]
            r_distances = results["distances"][0]
            r_metadatas = results["metadatas"][0]
            r_ids = results["ids"][0]
            predictions = []

            for i, (doc, distance, meta, rid) in enumerate(zip(r_documents, r_distances, r_metadatas, r_ids)):
                similarity_score = 1 - distance
                prediction = meta.get("encoding", DEFAULT_ENCODING) if similarity_score > SIMILARITY_THRESHOLD else DEFAULT_ENCODING

                predictions.append({
                    "rank": i + 1,
                    "id": rid,
                    "similarity_score": similarity_score,
                    "metadata": meta,
                    "document": doc,
                    "prediction": prediction
                })
            best = max(predictions, key=lambda x: x["similarity_score"])
            if best["prediction"] == "ordinal":
                ordinal_encoder.append(j)
        onehot_encoder = [col for col in onehot_encoder if col not in ordinal_encoder]
        print("Ordinal encoding columns:\n")
        for k in ordinal_encoder.copy():
            print(f"Column: {k}")
            print("Unique values:")
            print(data[k].dropna().unique())

            while True:
                proceed = input("Keep as Ordinal? (Y/N): ").strip().upper()

                if proceed == "Y":
                    break

                elif proceed == "N":
                    onehot_encoder.append(k)
                    ordinal_encoder.remove(k)
                    break

                else:
                    print("Only Y and N are expected.")
        all_lists = [
            mean_imputer,
            median_imputer,
            mode_imputer,
            label_encoder,
            onehot_encoder,
            standard_scaler,
            robust_scaler,
            no_scaling,
            ordinal_encoder
        ]
        for lst in tqdm(all_lists, desc="Cleaning Drop Columns"):
            lst[:] = [col for col in lst if col not in drop_cols]

        mode_label = [col for col in mode_imputer if col in label_encoder]
        mode_onehot = [col for col in mode_imputer if col in onehot_encoder]
        mode_ordinal = [col for col in mode_imputer if col in ordinal_encoder]
        mean_standard = [col for col in mean_imputer if col in standard_scaler]
        mean_robust = [col for col in mean_imputer if col in robust_scaler]
        mean_noscale = [col for col in mean_imputer if col in no_scaling]
        median_standard = [col for col in median_imputer if col in standard_scaler]
        median_robust = [col for col in median_imputer if col in robust_scaler]
        median_noscale = [col for col in median_imputer if col in no_scaling]
        label_only = [col for col in label_encoder if col not in mode_imputer]
        onehot_only = [col for col in onehot_encoder if col not in mode_imputer]
        ordinal_only = [col for col in ordinal_encoder if col not in mode_imputer]
        standard_only = [col for col in standard_scaler if col not in mean_imputer + median_imputer]
        robust_only = [col for col in robust_scaler if col not in mean_imputer + median_imputer]
        noscale_only = [col for col in no_scaling if col not in mean_imputer + median_imputer]

        mode_label_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OrdinalEncoder())])
        mode_onehot_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))])
        mode_ordinal_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])
        mean_standard_pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
        mean_robust_pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', RobustScaler())])
        median_standard_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
        median_robust_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', RobustScaler())])
        mean_noscale_pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean'))])
        median_noscale_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median'))])
        label_pipeline = Pipeline([('encoder', OrdinalEncoder())])
        onehot_pipeline = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore'))])
        ordinal_pipeline = Pipeline([('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])
        standard_pipeline = Pipeline([('scaler', StandardScaler())])
        robust_pipeline = Pipeline([('scaler', RobustScaler())])

        transformers = []
        if mode_label:
            transformers.append(('mode_label', mode_label_pipeline, mode_label))
        if mode_onehot:
            transformers.append(('mode_onehot', mode_onehot_pipeline, mode_onehot))
        if mode_ordinal:
            transformers.append(('mode_ordinal', mode_ordinal_pipeline, mode_ordinal))
        if mean_standard:
            transformers.append(('mean_standard', mean_standard_pipeline, mean_standard))
        if mean_robust:
            transformers.append(('mean_robust', mean_robust_pipeline, mean_robust))
        if mean_noscale:
            transformers.append(('mean_noscale', mean_noscale_pipeline, mean_noscale))
        if median_standard:
            transformers.append(('median_standard', median_standard_pipeline, median_standard))
        if median_robust:
            transformers.append(('median_robust', median_robust_pipeline, median_robust))
        if median_noscale:
            transformers.append(('median_noscale', median_noscale_pipeline, median_noscale))
        if label_only:
            transformers.append(('label', label_pipeline, label_only))
        if onehot_only:
            transformers.append(('onehot', onehot_pipeline, onehot_only))
        if ordinal_only:
            transformers.append(('ordinal', ordinal_pipeline, ordinal_only))
        if robust_only:
            transformers.append(('robust', robust_pipeline, robust_only))
        if standard_only:
            transformers.append(('standard', standard_pipeline, standard_only))
        if noscale_only:
            transformers.append(('no_scaling', 'passthrough', noscale_only))

        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )
        self.preprocessor.fit(data)

        self.report_ = {
            "Mean Imputer": mean_imputer,
            "Median Imputer": median_imputer,
            "Mode Imputer": mode_imputer,
            "Label Encoder": label_encoder,
            "OneHot Encoder": onehot_encoder,
            "Ordinal Encoder": ordinal_encoder,
            "Standard Scaler": standard_scaler,
            "Robust Scaler": robust_scaler,
            "No Scaling Required": no_scaling,
            "Drop Columns": drop_cols
        }

        return self

    def transform(self, data):
        return self.preprocessor.transform(data)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
