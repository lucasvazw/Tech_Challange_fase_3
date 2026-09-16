import os
import sys
import json
import gzip
import logging
import subprocess
import urllib.request
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dependencias no Databricks Runtime basico
for pkg in ["shap", "xgboost"]:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import shap
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, roc_curve
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pipeline_alfabetizacao")

sns.set_theme(style="whitegrid")


def buscar_estados_ibge():
    url = "https://servicodados.ibge.gov.br/api/v1/localidades/estados"
    req = urllib.request.Request(url, headers={"User-Agent": "DataPipeline/1.0", "Accept-Encoding": "gzip"})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data_raw = response.read()
            if response.info().get("Content-Encoding") == "gzip" or data_raw[:2] == b"\x1f\x8b":
                data_raw = gzip.decompress(data_raw)
            payload = json.loads(data_raw.decode("utf-8"))
            
        df = pd.DataFrame(payload)[["id", "sigla", "nome", "regiao"]]
        df["regiao_nome"] = df["regiao"].apply(lambda x: x.get("nome") if isinstance(x, dict) else str(x))
        return df.rename(columns={"id": "co_uf", "sigla": "sg_uf", "nome": "no_uf"}).drop(columns=["regiao"])
    except Exception as exc:
        logger.warning(f"Falha ao consultar API IBGE: {exc}. Prosseguindo sem enriquecimento externo.")
        return pd.DataFrame(columns=["co_uf", "sg_uf", "no_uf", "regiao_nome"])


def carregar_dados():
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
        from pyspark.sql.window import Window

        spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
        df_aluno = spark.table("workspace.default.gold_aluno")
        df_ind = spark.table("workspace.default.gold_indicador_municipio")
        
        join_keys = ["nu_ano_avaliacao", "co_uf", "co_municipio", "tp_dependencia"]
        cols_duplicadas = [c for c in df_ind.columns if c in df_aluno.columns and c not in join_keys]
        df_ind_clean = df_ind.drop(*cols_duplicadas)

        df_joined = df_aluno.join(df_ind_clean, on=join_keys, how="left")

        # Engenharia de contexto intraescolar
        if "id_escola" in df_joined.columns:
            w_escola = Window.partitionBy("id_escola")
            df_joined = df_joined.withColumn(
                "tx_alfabetizacao_escola",
                (F.sum("in_alfabetizado").over(w_escola) + 5.0) / (F.count("in_alfabetizado").over(w_escola) + 10.0)
            ).withColumn(
                "vol_alunos_escola",
                F.count("in_alfabetizado").over(w_escola)
            )

        # Amostragem para evitar estouro de RAM (OOM) no driver
        total_rows = df_joined.count()
        target_rows = 250000
        if total_rows > target_rows:
            logger.info(f"Base volumosa detectada ({total_rows} linhas). Amostrando estratificadamente {target_rows} amostras...")
            fraction = float(target_rows) / total_rows
            df_joined = df_joined.sampleBy("in_alfabetizado", fractions={0: fraction, 1: fraction}, seed=42)

        pdf = df_joined.toPandas()
        return pdf.loc[:, ~pdf.columns.duplicated()].copy()
    except Exception:
        pdf = pd.read_parquet("data/gold_dados_consolidados.parquet")
        return pdf.loc[:, ~pdf.columns.duplicated()].copy()


def criar_features_avancadas(df):
    df_clean = df.loc[:, ~df.columns.duplicated()].copy()
    
    # 1. Ratios socioeconomicos municipais
    if "vl_renda_per_capita_ano_atual" in df_clean.columns and "pct_vulnerabilidade_social_ano_atual" in df_clean.columns:
        df_clean["ratio_renda_vulnerabilidade"] = (
            df_clean["vl_renda_per_capita_ano_atual"] / (df_clean["pct_vulnerabilidade_social_ano_atual"] + 1.0)
        )
        df_clean["log_renda_per_capita"] = np.log1p(np.maximum(0, df_clean["vl_renda_per_capita_ano_atual"]))

    if "vl_idhm_ano_atual" in df_clean.columns and "pct_vulnerabilidade_social_ano_atual" in df_clean.columns:
        df_clean["indice_pressao_social"] = df_clean["pct_vulnerabilidade_social_ano_atual"] / (df_clean["vl_idhm_ano_atual"] + 1e-4)

    # 2. Deltas temporais
    if "vl_idhm_ano_atual" in df_clean.columns and "vl_idhm_ano_anterior" in df_clean.columns:
        df_clean["delta_idhm"] = df_clean["vl_idhm_ano_atual"] - df_clean["vl_idhm_ano_anterior"]

    if "percentual_alfabetizados_ano_atual" in df_clean.columns and "percentual_alfabetizados_ano_anterior" in df_clean.columns:
        df_clean["delta_alfabetizacao"] = df_clean["percentual_alfabetizados_ano_atual"] - df_clean["percentual_alfabetizados_ano_anterior"]

    return df_clean


def split_features(df):
    df_clean = criar_features_avancadas(df)
    
    # Remocao de vazamento direto (notas/status finais) e constantes
    leakage_cols = [
        "vl_proficiencia_lp", "status_alfabetizacao", "media_proficiencia_lp",
        "percentual_alfabetizados", "percentual_nao_alfabetizados",
        "id_aluno", "nu_ano_avaliacao", "co_municipio", "no_municipio",
        "in_presenca_lp", "status_presenca_lp"
    ]
    features = [c for c in df_clean.columns if c not in leakage_cols and c != "in_alfabetizado"]
    return df_clean[features], df_clean["in_alfabetizado"]


def get_preprocessor(X):
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    if "id_escola" in num_cols:
        num_cols.remove("id_escola")
        cat_cols.append("id_escola")

    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", TargetEncoder(smooth="auto", cv=5, random_state=42))
        ]), cat_cols)
    ])


def benchmark_modelos(X_train, y_train, X_val, y_val, preprocessor):
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_weight = float(n_neg) / max(1, n_pos)

    candidatos = {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "rf": RandomForestClassifier(n_estimators=120, max_depth=14, class_weight="balanced", max_samples=0.75, random_state=42, n_jobs=-1),
        "xgb": XGBClassifier(n_estimators=160, learning_rate=0.06, max_depth=6, scale_pos_weight=scale_weight, eval_metric="logloss", random_state=42, n_jobs=2)
    }

    resultados = {}
    pipes = {}

    for tag, clf in candidatos.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        pipe.fit(X_train, y_train)
        
        preds = pipe.predict(X_val)
        probs = pipe.predict_proba(X_val)[:, 1]
        
        auc = roc_auc_score(y_val, probs)
        f1_macro = f1_score(y_val, preds, average="macro")
        rec_nao_alfab = recall_score(y_val, preds, pos_label=0)
        
        resultados[tag] = {
            "acc": accuracy_score(y_val, preds),
            "f1": f1_score(y_val, preds),
            "f1_macro": f1_macro,
            "roc_auc": auc,
            "recall_risco": rec_nao_alfab
        }
        pipes[tag] = pipe
        logger.info(f"Modelo: {tag:<8} | ROC-AUC: {auc:.4f} | F1-Macro: {f1_macro:.4f} | Recall(Nao-Alfab): {rec_nao_alfab:.4f}")

    best_tag = max(resultados, key=lambda k: resultados[k]["roc_auc"])
    logger.info(f"Melhor baseline selecionado: {best_tag}")
    return pipes[best_tag], best_tag, resultados


def otimizar_modelo(best_pipe, X_train, y_train):
    clf = best_pipe.named_steps["clf"]
    
    if isinstance(clf, XGBClassifier):
        grid = {
            "clf__n_estimators": [120, 200],
            "clf__max_depth": [5, 7, 9],
            "clf__learning_rate": [0.03, 0.06],
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.7, 0.9]
        }
    elif isinstance(clf, RandomForestClassifier):
        grid = {
            "clf__n_estimators": [100, 160],
            "clf__max_depth": [12, 18],
            "clf__min_samples_split": [5, 10]
        }
    else:
        grid = {"clf__C": [0.1, 1.0, 5.0]}

    search = RandomizedSearchCV(
        best_pipe,
        param_distributions=grid,
        n_iter=3,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        random_state=42,
        n_jobs=1
    )
    search.fit(X_train, y_train)
    logger.info(f"ROC-AUC apos otimizacao de hiperparametros: {search.best_score_:.4f}")
    return search.best_estimator_


def otimizar_limiar_decisao(pipeline, X_val, y_val):
    probs_1 = pipeline.predict_proba(X_val)[:, 1]
    melhor_f1_risco = 0.0
    melhor_threshold = 0.50
    
    for thr in np.linspace(0.40, 0.75, 36):
        preds = (probs_1 >= thr).astype(int)
        f1_risco = f1_score(y_val, preds, pos_label=0, zero_division=0)
        if f1_risco > melhor_f1_risco:
            melhor_f1_risco = f1_risco
            melhor_threshold = thr
            
    logger.info(f"Limiar calibrado para Classe 0: {melhor_threshold:.3f} (F1 Risco Max: {melhor_f1_risco:.4f})")
    return melhor_threshold


def gerar_artefatos_avaliacao(pipeline, X_test, y_test, threshold=0.5, output_dir="images"):
    os.makedirs(output_dir, exist_ok=True)
    
    probs = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (probs >= threshold).astype(int)
    
    logger.info("\n=== RELATORIO DE CLASSIFICACAO FINAL (TESTE) ===")
    logger.info("\n" + classification_report(y_test, y_pred, target_names=["Nao Alfabetizado", "Alfabetizado"]))
    
    # Matriz de Confusao
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_ylabel("Real")
    ax.set_xlabel(f"Predito (Threshold: {threshold:.2f})")
    ax.set_title(f"Matriz de Confusao (Limiar = {threshold:.2f})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "matriz_confusao.png"), dpi=200)
    plt.close()

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, probs):.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.7)
    ax.legend(loc="lower right")
    ax.set_title("Curva ROC ")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "curva_roc.png"), dpi=200)
    plt.close()

    return probs, y_pred


def gerar_shap_plots(pipeline, X_sample, output_dir="images"):
    os.makedirs(output_dir, exist_ok=True)
    prep = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]

    X_tx = prep.transform(X_sample)
    num_features = prep.transformers_[0][2]
    cat_features = prep.transformers_[1][2]
    features_out = num_features + cat_features
    X_tx_df = pd.DataFrame(X_tx, columns=features_out)

    try:
        explainer = shap.TreeExplainer(clf)
        values = explainer.shap_values(X_tx_df)
    except Exception:
        explainer = shap.Explainer(clf, X_tx_df)
        values = explainer(X_tx_df).values

    if isinstance(values, list):
        values = values[1]

    plt.figure(figsize=(10, 6))
    shap.summary_plot(values, X_tx_df, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary.png"), dpi=200, bbox_inches="tight")
    plt.close()


def exportar_camada_consumo(df_gold, pipeline, X_test, y_test, resultados, threshold=0.5, out_path="data/power_bi"):
    os.makedirs(out_path, exist_ok=True)
    
    # 1. Métricas dos modelos
    df_metrics = pd.DataFrame(resultados).T.reset_index().rename(columns={"index": "modelo"})
    df_metrics.to_csv(os.path.join(out_path, "gold_metricas_modelos.csv"), index=False)

    # 2. Base escorada com threshold calibrado
    probs = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (probs >= threshold).astype(int)
    
    df_scores = df_gold.loc[X_test.index].copy()
    df_scores["in_alfabetizado_real"] = y_test.values
    df_scores["in_alfabetizado_pred"] = y_pred
    df_scores["score_prob"] = (probs * 100).round(2)
    
    conditions = [
        df_scores["score_prob"] < 45.0,
        (df_scores["score_prob"] >= 45.0) & (df_scores["score_prob"] < 70.0),
        df_scores["score_prob"] >= 70.0
    ]
    df_scores["cluster_risco"] = np.select(conditions, ["Alto", "Medio", "Baixo"], default="N/D")
    df_scores.to_csv(os.path.join(out_path, "gold_predicoes_risco.csv"), index=False)

    # 3. Dimensão IBGE
    df_ibge = buscar_estados_ibge()
    if not df_ibge.empty:
        df_ibge.to_csv(os.path.join(out_path, "dim_estados_ibge.csv"), index=False)

    # Sincronização Delta Lake no Databricks
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
        spark.createDataFrame(df_scores).write.format("delta").mode("overwrite").saveAsTable("workspace.default.gold_predicoes_risco_ml")
        logger.info("Tabelas Delta sincronizadas no metastore com sucesso.")
    except Exception:
        pass


def main():
    logger.info("Carregando base analitica...")
    df = carregar_dados()
    
    X, y = split_features(df)
    preprocessor = get_preprocessor(X)
    
    # Divisão: 70% treino, 15% validação, 15% teste
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.176, stratify=y_train_val, random_state=42)
    
    best_pipe, tag, resultados = benchmark_modelos(X_train, y_train, X_val, y_val, preprocessor)
    pipe_final = otimizar_modelo(best_pipe, X_train, y_train)
    
    threshold_otimizado = otimizar_limiar_decisao(pipe_final, X_val, y_val)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe_final, "models/modelo_alfabetizacao.joblib")
    
    gerar_artefatos_avaliacao(pipe_final, X_test, y_test, threshold=threshold_otimizado)
    gerar_shap_plots(pipe_final, X_test.sample(n=min(250, len(X_test)), random_state=42))
    exportar_camada_consumo(df, pipe_final, X_test, y_test, resultados, threshold=threshold_otimizado)
    logger.info("Processamento finalizado com sucesso.")


if __name__ == "__main__":
    main()