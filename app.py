import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Regression Streamlit App", layout="wide")
st.title("Часть 5 | Streamlit: EDA + Inference + Weights")

BASE = Path(__file__).parent

@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)

@st.cache_data
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    return df[cols]

def safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="ignore")
    return out

def extract_coef(model):
    if hasattr(model, "coef_"):
        return np.asarray(model.coef_).ravel()
    if hasattr(model, "named_steps"):
        for step_name in reversed(list(model.named_steps.keys())):
            est = model.named_steps[step_name]
            if hasattr(est, "coef_"):
                return np.asarray(est.coef_).ravel()
    return None

MODELS = {
    "LinearRegression (8 фич)": ("lin_reg.joblib", "features_8.json"),
    "Lasso best (8 фич)": ("best_lasso.joblib", "features_8.json"),
    "ElasticNet best (8 фич)": ("best_enet.joblib", "features_8.json"),
    "Ridge best (52 фич)": ("best_ridge.joblib", "features_52.json"),
}

st.sidebar.header("Настройки")
model_name = st.sidebar.selectbox("Выберите модель", list(MODELS.keys()))
model_file, feat_file = MODELS[model_name]

model_path = BASE / model_file
feat_path = BASE / feat_file

if not model_path.exists():
    st.error(f"Не найден файл модели: {model_file}")
    st.stop()

if not feat_path.exists():
    st.error(f"Не найден файл признаков: {feat_file}")
    st.stop()

model = load_model(model_path)
feat_cols = load_json(feat_path)

st.sidebar.caption(f"Файл модели: {model_file}")
st.sidebar.caption(f"Кол-во признаков: {len(feat_cols)}")

tab_eda, tab_infer, tab_weights = st.tabs(["1) EDA", "2) Inference", "3) Weights"])

with tab_eda:
    st.subheader("Основные графики/гистограммы (EDA)")

    default_eda = BASE / "train_for_eda.csv"
    if default_eda.exists():
        df_eda = pd.read_csv(default_eda)
        st.info("Использую файл train_for_eda.csv.")
    else:
        up_eda = st.file_uploader("Загрузить CSV для EDA", type=["csv"], key="eda_upload")
        df_eda = pd.read_csv(up_eda) if up_eda else None

    if df_eda is None:
        st.stop()

    df_eda = safe_numeric(df_eda)

    st.dataframe(df_eda.head(20), use_container_width=True)
    st.caption(f"Размер: {df_eda.shape[0]} × {df_eda.shape[1]}")

    num_cols = df_eda.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) == 0:
        st.stop()

    c1, c2 = st.columns(2)

    with c1:
        col = st.selectbox("Признак", num_cols)
        bins = st.slider("bins", 10, 100, 30)
        fig = plt.figure()
        plt.hist(df_eda[col].dropna().values, bins=bins)
        plt.xlabel(col)
        plt.ylabel("count")
        st.pyplot(fig)

        fig2 = plt.figure()
        plt.boxplot(df_eda[col].dropna().values, vert=True)
        plt.ylabel(col)
        st.pyplot(fig2)

    with c2:
        na = df_eda.isna().mean().sort_values(ascending=False)
        st.dataframe(pd.DataFrame({"na_share": na}), use_container_width=True)

        if len(num_cols) >= 2:
            corr = df_eda[num_cols].corr(numeric_only=True)
            pairs = []
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    pairs.append((num_cols[i], num_cols[j], float(corr.iloc[i, j])))
            top = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)[:10]
            st.dataframe(pd.DataFrame(top, columns=["feat_1", "feat_2", "corr"]), use_container_width=True)

with tab_infer:
    st.subheader("Инференс")

    mode = st.radio("Режим", ["Загрузить CSV", "Ввести один объект"], horizontal=True)

    if mode == "Загрузить CSV":
        up = st.file_uploader("CSV с признаками", type=["csv"], key="infer_csv")
        if up:
            df_in = pd.read_csv(up)
            df_in = safe_numeric(df_in)

            st.dataframe(df_in.head(20), use_container_width=True)

            X = ensure_columns(df_in, feat_cols)
            preds = model.predict(X)
            out = df_in.copy()
            out["prediction"] = preds

            st.dataframe(out.head(50), use_container_width=True)

            st.download_button(
                "Скачать predictions.csv",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )
    else:
        values = {}
        cols = st.columns(2)
        for i, f in enumerate(feat_cols):
            with cols[i % 2]:
                values[f] = st.number_input(f, value=0.0, format="%.6f")

        if st.button("Предсказать"):
            X_one = pd.DataFrame([values])
            pred = float(model.predict(X_one)[0])
            st.success(f"Prediction: {pred:.6f}")

with tab_weights:
    st.subheader("Веса модели")

    coefs = extract_coef(model)
    if coefs is None:
        st.stop()

    if len(coefs) != len(feat_cols):
        feat_names = [f"f{i}" for i in range(len(coefs))]
    else:
        feat_names = feat_cols

    df_w = pd.DataFrame({"feature": feat_names, "weight": coefs})
    df_w["abs_weight"] = df_w["weight"].abs()

    top_k = st.slider("Top-K", 5, min(50, len(df_w)), min(20, len(df_w)))
    df_top = df_w.sort_values("abs_weight", ascending=False).head(top_k)

    st.dataframe(df_top[["feature", "weight"]], use_container_width=True)

    fig = plt.figure()
    plt.barh(df_top["feature"][::-1], df_top["weight"][::-1])
    plt.xlabel("weight")
    plt.ylabel("feature")
    st.pyplot(fig)