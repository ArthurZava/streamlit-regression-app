import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Regression Streamlit App", layout="wide")
st.title("Часть 5 | Streamlit: EDA + Inference + Weights")

BASE = Path(__file__).parent

@st.cache_resource
def load_pipeline(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="ignore")
    return out

def extract_estimator(pipe):
    if hasattr(pipe, "named_steps"):
        return pipe.named_steps.get("model", None)
    return None

def extract_coef(pipe):
    est = extract_estimator(pipe)
    if est is not None and hasattr(est, "coef_"):
        return np.asarray(est.coef_).ravel()
    if hasattr(pipe, "coef_"):
        return np.asarray(pipe.coef_).ravel()
    return None

def get_feature_names_from_pipeline(pipe):
    if hasattr(pipe, "feature_names_in_"):
        return list(pipe.feature_names_in_)
    return None

PIPELINES = {
    "LinearRegression (8 фич)": "lin_reg_pipeline.pkl",
    "Lasso best (8 фич)": "best_lasso_pipeline.pkl",
    "ElasticNet best (8 фич)": "best_enet_pipeline.pkl",
    "Ridge best (52 фич)": "best_ridge_pipeline.pkl",
}

st.sidebar.header("Настройки")
model_name = st.sidebar.selectbox("Выберите модель", list(PIPELINES.keys()))
pipe_file = PIPELINES[model_name]
pipe_path = BASE / pipe_file

if not pipe_path.exists():
    st.error(f"Не найден файл пайплайна: {pipe_file}. Положите его рядом с app.py")
    st.stop()

pipe = load_pipeline(pipe_path)
feat_cols = get_feature_names_from_pipeline(pipe)

st.sidebar.caption(f"Файл пайплайна: {pipe_file}")
if feat_cols is not None:
    st.sidebar.caption(f"Кол-во признаков: {len(feat_cols)}")
else:
    st.sidebar.caption("Кол-во признаков: неизвестно (нет feature_names_in_)")

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

            if feat_cols is not None:
                missing = [c for c in feat_cols if c not in df_in.columns]
                extra = [c for c in df_in.columns if c not in feat_cols]
                if missing:
                    st.warning(f"В CSV не хватает {len(missing)} признаков. Пример: {missing[:5]}")
                if extra:
                    st.info(f"В CSV есть лишние колонки (будут проигнорированы): {extra[:5]}")

                X = df_in.reindex(columns=feat_cols)
            else:
                X = df_in

            preds = pipe.predict(X)
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
        if feat_cols is None:
            st.error("Не удалось определить список признаков из пайплайна. Для ручного ввода нужны feature_names_in_.")
            st.stop()

        values = {}
        cols = st.columns(2)
        for i, f in enumerate(feat_cols):
            with cols[i % 2]:
                values[f] = st.number_input(f, value=0.0, format="%.6f")

        if st.button("Предсказать"):
            X_one = pd.DataFrame([values])
            pred = float(pipe.predict(X_one)[0])
            st.success(f"Prediction: {pred:.6f}")

with tab_weights:
    st.subheader("Веса модели")

    coefs = extract_coef(pipe)
    if coefs is None:
        st.error("У выбранной модели не удалось извлечь coef_.")
        st.stop()

    if feat_cols is None or len(coefs) != len(feat_cols):
        feat_names = [f"f{i}" for i in range(len(coefs))]
    else:
        feat_names = feat_cols

    df_w = pd.DataFrame({"feature": feat_names, "weight": coefs})
    df_w["abs_weight"] = df_w["weight"].abs()

    top_k = st.slider("Top-K", 5, len(df_w), min(20, len(df_w)))
    df_top = df_w.sort_values("abs_weight", ascending=False).head(top_k)

    st.dataframe(df_top[["feature", "weight"]], use_container_width=True)

    fig = plt.figure()
    plt.barh(df_top["feature"][::-1], df_top["weight"][::-1])
    plt.xlabel("weight")
    plt.ylabel("feature")
    st.pyplot(fig)