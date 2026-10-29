import io
import json
import os
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

# ----------------------------------------------------------------------------
# Configuración general (set_page_config debe ser el primer comando de Streamlit)
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Detector de Diabetes", page_icon="🩺", layout="wide")

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"   # app en src/, modelos en models/
MODEL_PATH = MODELS_DIR / "xgboost_diabetes_model.pkl"
MEDIANS_PATH = MODELS_DIR / "medians_train.json"

# mismo orden que en el entrenamiento del notebook
PREDICTORAS = [
    "Glucose", "BMI", "Age", "Pregnancies", "SkinThickness", "Insulin",
    "BloodPressure", "DiabetesPedigreeFunction",
]
# en estas columnas un 0 es fisiológicamente imposible: se trata como dato faltante y se imputa con la mediana de train
COLS_CON_CEROS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
COLS_ID_POSIBLES = ["patientid", "patient_id", "id", "customerid", "paciente"]
COL_TARGET = "Outcome"

COLOR_ALTO, COLOR_MEDIO, COLOR_BAJO = "#B42318", "#C98A1B", "#2F7D5B"

# ----------------------------------------------------------------------------
# Estilo
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
    html, body, [class*="css"], .stMarkdown, button, input { font-family: 'IBM Plex Sans', system-ui, sans-serif; }
    .block-container { max-width: 1180px; padding-top: 3.5rem; }
    footer { visibility: hidden; }

    .titulo { font-size: 2rem; line-height: 1.3; font-weight: 600; letter-spacing: -0.02em; margin: 0; }
    .subtitulo { color: #5b6673; margin: .25rem 0 1.6rem 0; max-width: 62ch; line-height: 1.5; }

    [data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #e3e7ec; border-radius: 6px; padding: 14px 18px;
    }
    [data-testid="stMetricLabel"] p { color: #5b6673; font-size: .85rem; }
    [data-testid="stMetricValue"] { font-weight: 600; letter-spacing: -0.02em; }

    .barra { display: flex; height: 14px; border-radius: 3px; overflow: hidden; margin: .4rem 0 .5rem 0; }
    .leyenda { display: flex; gap: 1.4rem; font-size: .85rem; color: #3b4652; }
    .punto { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Funciones
# ----------------------------------------------------------------------------
@st.cache_resource
def cargar_modelo():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def cargar_medianas():
    """Medianas de TRAIN guardadas desde el notebook (None si el archivo no existe)."""
    if MEDIANS_PATH.exists():
        with open(MEDIANS_PATH) as f:
            return json.load(f)
    return None


def leer_archivo(archivo):
    extension = os.path.splitext(archivo.name)[1].lower()
    if extension == ".csv":
        return pd.read_csv(archivo)
    return pd.read_excel(archivo)


def columna_id(df):
    """Devuelve el nombre de la columna identificadora si existe."""
    for c in df.columns:
        if str(c).strip().lower() in COLS_ID_POSIBLES:
            return c
    return None


def validar(df):
    """Devuelve una lista de problemas encontrados (vacía si el archivo es válido)."""
    problemas = []
    faltan = [c for c in PREDICTORAS if c not in df.columns]
    if faltan:
        problemas.append(f"Faltan estas columnas: {', '.join(faltan)}.")
        return problemas

    numericos = df[PREDICTORAS].apply(pd.to_numeric, errors="coerce")
    # Las columnas con ceros imposibles se imputan; el resto no admite vacíos ni texto
    obligatorias = [c for c in PREDICTORAS if c not in COLS_CON_CEROS]
    con_nulos = [c for c in obligatorias if numericos[c].isnull().any()]
    if con_nulos:
        problemas.append(
            f"Hay valores vacíos o no numéricos en: {', '.join(con_nulos)}. "
            "Completa esos datos y vuelve a subir el archivo."
        )
    return problemas


def preparar(df, medianas):
    """Replica el preprocesado del notebook: ceros -> NaN -> mediana de train."""
    datos = df[PREDICTORAS].apply(pd.to_numeric, errors="coerce").copy()
    datos[COLS_CON_CEROS] = datos[COLS_CON_CEROS].replace(0, float("nan"))
    n_imputados = int(datos[COLS_CON_CEROS].isnull().sum().sum())
    datos[COLS_CON_CEROS] = datos[COLS_CON_CEROS].fillna(pd.Series(medianas))
    return datos, n_imputados


def predecir(df, medianas):
    """Devuelve un DataFrame con el ID del paciente y su probabilidad de diabetes (0-100)."""
    modelo = cargar_modelo()
    datos, n_imputados = preparar(df, medianas)
    orden = list(getattr(modelo, "feature_names_in_", PREDICTORAS))
    probas = modelo.predict_proba(datos[orden])
    # clase 1 = "Diabetes" (Outcome = 1)
    col_id = columna_id(df)
    ids = df[col_id].values if col_id else range(1, len(df) + 1)
    res = pd.DataFrame({"Paciente": ids, "Probabilidad": probas[:, 1] * 100})
    return res, n_imputados


def nivel_riesgo(prob, medio_min, alto_min):
    return "Alto" if prob >= alto_min else ("Medio" if prob >= medio_min else "Bajo")


# ----------------------------------------------------------------------------
# Barra lateral
# ----------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Cómo funciona")
    st.markdown(
        "1. Sube el archivo con los datos de tus pacientes.\n"
        "2. Pulsa **Calcular predicción**.\n"
        "3. Revisa el riesgo de cada paciente y descarga el resultado."
    )
    st.divider()
    st.subheader("Columnas necesarias")
    st.caption(", ".join(PREDICTORAS))
    st.caption("Opcionales: una columna de ID del paciente y `Outcome` (para medir el acierto del modelo).")
    st.divider()
    st.subheader("Niveles de riesgo")
    medio_min, alto_min = st.slider(
        "Tramo de riesgo medio (%)", 0, 100, (30, 50),
        help="Por debajo del tramo, el riesgo es bajo. Por encima, alto.",
    )
    st.caption(f"Bajo: menos de {medio_min}% · Medio: {medio_min}%–{alto_min}% · Alto: {alto_min}% o más")

# ----------------------------------------------------------------------------
# Cabecera y carga de datos
# ----------------------------------------------------------------------------
st.markdown('<p class="titulo">Detector de Diabetes</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitulo">Estima qué pacientes tienen más probabilidad de ser diabéticos a partir de sus '
    "medidas diagnósticas, para priorizar su revisión médica. Es una herramienta de apoyo, no un diagnóstico.</p>",
    unsafe_allow_html=True,
)

archivo = st.file_uploader(
    "Archivo de pacientes", type=["csv", "xlsx", "xls"], help="Formatos admitidos: CSV, XLSX y XLS."
)

if archivo is None:
    st.session_state.pop("resultados", None)
    st.info("Sube un archivo para empezar.")
    plantilla = pd.DataFrame(columns=["PatientID"] + PREDICTORAS)
    st.download_button(
        "Descargar plantilla (CSV)",
        data=plantilla.to_csv(index=False).encode("utf-8-sig"),
        file_name="plantilla_pacientes.csv",
        mime="text/csv",
    )
    st.stop()

try:
    dataframe = leer_archivo(archivo)
except Exception as e:
    st.error(f"No se pudo leer el archivo: {e}")
    st.stop()

problemas = validar(dataframe)
if problemas:
    for p in problemas:
        st.error(p)
    st.stop()

# Medianas de train: archivo del notebook, o (como alternativa) las del archivo subido
medianas = cargar_medianas()
if medianas is None:
    medianas = dataframe[COLS_CON_CEROS].apply(pd.to_numeric, errors="coerce").replace(0, float("nan")).median().to_dict()
    st.warning(
        "No se encontró `models/medians_train.json`. Para imputar los datos faltantes se usarán las medianas "
        "del archivo subido, que pueden diferir de las usadas en el entrenamiento."
    )

st.success(f"Archivo cargado: {len(dataframe):,} pacientes.")
with st.expander("Ver vista previa de los datos"):
    st.dataframe(dataframe.head(10), use_container_width=True)

# Reiniciar resultados si el usuario cambia de archivo
firma = (archivo.name, archivo.size)
if st.session_state.get("firma") != firma:
    st.session_state.pop("resultados", None)
    st.session_state["firma"] = firma

if st.button("Calcular predicción", type="primary"):
    with st.spinner("Calculando..."):
        try:
            st.session_state["resultados"], st.session_state["n_imputados"] = predecir(dataframe, medianas)
        except FileNotFoundError:
            st.error(f"No se encontró el modelo en {MODEL_PATH}.")
            st.stop()

if "resultados" not in st.session_state:
    st.stop()

# ----------------------------------------------------------------------------
# Resultados
# ----------------------------------------------------------------------------
res = st.session_state["resultados"].copy()
res["Nivel de riesgo"] = res["Probabilidad"].apply(lambda p: nivel_riesgo(p, medio_min, alto_min))
res["Diabetes prevista"] = res["Probabilidad"].apply(lambda p: "Sí" if p >= 50 else "No")

total = len(res)
n_alto = int((res["Nivel de riesgo"] == "Alto").sum())
n_medio = int((res["Nivel de riesgo"] == "Medio").sum())
n_bajo = total - n_alto - n_medio

st.subheader("Resultados")
if st.session_state.get("n_imputados", 0) > 0:
    st.caption(
        f"Se imputaron {st.session_state['n_imputados']:,} valores vacíos o iguales a 0 "
        "(Glucose, BloodPressure, SkinThickness, Insulin, BMI) con la mediana del entrenamiento."
    )

c1, c2, c3, c4 = st.columns(4)
c1.metric("Pacientes analizados", f"{total:,}")
c2.metric("Riesgo alto", f"{n_alto:,}")
c3.metric("Porcentaje en riesgo alto", f"{n_alto / total:.1%}")
c4.metric("Probabilidad media", f"{res['Probabilidad'].mean():.1f}%")

st.markdown(
    f"""
    <div class="barra">
      <div style="width:{n_alto / total * 100}%;background:{COLOR_ALTO}"></div>
      <div style="width:{n_medio / total * 100}%;background:{COLOR_MEDIO}"></div>
      <div style="width:{n_bajo / total * 100}%;background:{COLOR_BAJO}"></div>
    </div>
    <div class="leyenda">
      <span><span class="punto" style="background:{COLOR_ALTO}"></span>Alto: {n_alto}</span>
      <span><span class="punto" style="background:{COLOR_MEDIO}"></span>Medio: {n_medio}</span>
      <span><span class="punto" style="background:{COLOR_BAJO}"></span>Bajo: {n_bajo}</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

# Si el archivo trae Outcome, medimos qué tan bien acierta el modelo
if COL_TARGET in dataframe.columns:
    y_real = pd.to_numeric(dataframe[COL_TARGET], errors="coerce")
    if y_real.isin([0, 1]).all():
        y_pred = (st.session_state["resultados"]["Probabilidad"] >= 50).astype(int).values
        y_real = y_real.astype(int).values
        with st.expander("Rendimiento del modelo sobre este archivo (usa la columna Outcome)", expanded=True):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{accuracy_score(y_real, y_pred):.1%}")
            m2.metric("Precisión (diabetes)", f"{precision_score(y_real, y_pred, zero_division=0):.1%}")
            m3.metric("Recall (diabetes)", f"{recall_score(y_real, y_pred, zero_division=0):.1%}")
            m4.metric("F1 (diabetes)", f"{f1_score(y_real, y_pred, zero_division=0):.1%}")
            cm = confusion_matrix(y_real, y_pred, labels=[0, 1])
            st.dataframe(
                pd.DataFrame(cm, index=["Real: No diabetes", "Real: Diabetes"], columns=["Pred: No diabetes", "Pred: Diabetes"]),
                use_container_width=True,
            )
            st.caption("Si estos pacientes ya formaron parte del entrenamiento, las métricas serán demasiado optimistas.")

f1_, f2_ = st.columns([2, 1])
filtro = f1_.radio("Mostrar", ["Todos", "Riesgo alto", "Riesgo alto y medio"], horizontal=True)
busqueda = f2_.text_input("Buscar paciente", placeholder="ID de paciente")

vista = res
if filtro == "Riesgo alto":
    vista = vista[vista["Nivel de riesgo"] == "Alto"]
elif filtro == "Riesgo alto y medio":
    vista = vista[vista["Nivel de riesgo"] != "Bajo"]
if busqueda.strip():
    vista = vista[vista["Paciente"].astype(str).str.contains(busqueda.strip())]
vista = vista.sort_values("Probabilidad", ascending=False).reset_index(drop=True)
vista = vista[["Paciente", "Nivel de riesgo", "Probabilidad", "Diabetes prevista"]]

colores = {"Alto": COLOR_ALTO, "Medio": COLOR_MEDIO, "Bajo": COLOR_BAJO}
estilo = vista.style
aplicar = getattr(estilo, "map", None) or estilo.applymap   # 'map' en pandas >= 2.1
estilo = aplicar(lambda v: f"color: {colores.get(v, 'inherit')}; font-weight: 600", subset=["Nivel de riesgo"])

st.caption(f"{len(vista):,} pacientes, ordenados de mayor a menor probabilidad.")
st.dataframe(
    estilo,
    use_container_width=True,
    hide_index=True,
    height=420,
    column_config={
        "Paciente": st.column_config.TextColumn("Paciente"),
        "Probabilidad": st.column_config.ProgressColumn(
            "Probabilidad de diabetes", format="%.1f%%", min_value=0, max_value=100
        ),
    },
)

descarga = vista.rename(columns={"Probabilidad": "Probabilidad de diabetes (%)"}).round(1)

d1, d2 = st.columns([1, 3])
formato = d1.selectbox("Formato de descarga", ["CSV", "Excel"], label_visibility="collapsed")

if formato == "CSV":
    datos_descarga = descarga.to_csv(index=False).encode("utf-8-sig")
    nombre_archivo = "predicciones_diabetes.csv"
    tipo_mime = "text/csv"
else:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        descarga.to_excel(writer, index=False, sheet_name="Predicciones")
    datos_descarga = buffer.getvalue()
    nombre_archivo = "predicciones_diabetes.xlsx"
    tipo_mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

d2.download_button(
    f"Descargar resultados ({formato})",
    data=datos_descarga,
    file_name=nombre_archivo,
    mime=tipo_mime,
)