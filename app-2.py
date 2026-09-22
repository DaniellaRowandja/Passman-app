"""
Dashboard Streamlit — Prévision de la charge des points Wi-Fi Paris
=====================================================================
Application 100% autonome : ne dépend d'AUCUN fichier généré par le notebook.
Elle charge directement un CSV hébergé sur GitHub (votre extraction des
données Paris Wi-Fi), puis fait tout le pipeline en direct (agrégation,
feature engineering, entraînement, prévision) — mis en cache pour rester
rapide au fil des interactions.

COMMENT PUBLIER CETTE APPLICATION
----------------------------------
1. Déposez votre fichier "data-passman-paris.csv" dans un repo GitHub (public,
   ou privé + token si besoin).
2. Sur GitHub, ouvrez le fichier puis cliquez sur "Raw" : copiez l'URL
   (ex : https://raw.githubusercontent.com/<user>/<repo>/main/data-passman-paris.csv)
3. Collez cette URL dans DEFAULT_CSV_URL ci-dessous (ou dans le champ de la
   barre latérale au premier lancement, en local).
4. Placez ce fichier `app.py` + `requirements_app.txt` (renommé `requirements.txt`)
   dans un repo GitHub, puis déployez sur https://share.streamlit.io
   (Streamlit Community Cloud) en pointant vers ce repo.

FORMAT DE CSV ATTENDU (deux formats acceptés, détectés automatiquement)
-------------------------------------------------------------------------
A) Données BRUTES (1 ligne = 1 session Wi-Fi), colonnes typiques :
   - une colonne date/heure de connexion (ex: "date", "connectiondate"...)
   - une colonne identifiant le site (ex: "sitename", "site"...)
   - optionnellement : bytesin, bytesout, duration...
   -> l'appli agrège elle-même en (site, heure).

B) Données déjà AGRÉGÉES (1 ligne = 1 site x 1 heure), colonnes typiques :
   - une colonne date/heure ("datetime_hour", "date"...)
   - une colonne site
   - une colonne de charge ("nb_connexions", "nb_sessions", "count"...)
   -> l'appli les utilise directement (plus rapide à charger).

Si vos noms de colonnes ne sont détectés par aucun des mots-clés ci-dessous,
ajustez la fonction `guess_column()` ou renseignez COLUMN_OVERRIDES en bas
de la section CONFIGURATION.

Lancer en local : streamlit run app.py
"""

import hashlib
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor

st.set_page_config(page_title="Charge Wi-Fi Paris — Prévision", layout="wide")

# ============================================================================
# CONFIGURATION — à adapter
# ============================================================================
DEFAULT_CSV_URL = "https://github.com/DaniellaRowandja/Passman-app/blob/main/data-passman-paris.csv"
DEFAULT_GEO_URL = "https://github.com/DaniellaRowandja/Passman-app/blob/main/data-passman-paris-geo.csv"  # optionnel : CSV avec colonnes site,lat,lon pour la carte

# Si l'auto-détection des colonnes se trompe, forcez les noms exacts ici, ex :
# COLUMN_OVERRIDES = {"date": "connectiondate", "site": "sitename", "count": None, "bytesin": "bytesin"}
COLUMN_OVERRIDES = {"date": None, "site": None, "count": None, "bytesin": None}

FORECAST_MAX_DAYS = 14
RANDOM_STATE = 42
FEATURES = ["hour", "dayofweek", "is_weekend", "month", "is_holiday",
            "lag_1h", "lag_24h", "lag_168h", "roll_mean_7d", "roll_std_7d", "site_code"]
FR_HOLIDAYS_FIXED = {"01-01", "05-01", "05-08", "07-14", "08-15", "11-01", "11-11", "12-25"}


# ============================================================================
# CHARGEMENT & DÉTECTION DE SCHÉMA
# ============================================================================
def guess_column(columns, keywords, override=None):
    if override:
        return override
    for kw in keywords:
        for c in columns:
            if kw in str(c).lower():
                return c
    return None


@st.cache_data(show_spinner="Téléchargement du CSV…", ttl=3600)
def load_csv(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    if df.empty:
        raise ValueError("Le fichier CSV est vide.")
    return df


@st.cache_data(show_spinner="Préparation de la série temporelle…", ttl=3600)
def prepare_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    col_date = guess_column(cols, ["datetime_hour", "date", "connect", "start"], COLUMN_OVERRIDES.get("date"))
    col_site = guess_column(cols, ["sitename", "site_name", "site", "zonelabel"], COLUMN_OVERRIDES.get("site"))
    col_count = guess_column(cols, ["nb_connexions", "nb_sessions", "count", "charge"], COLUMN_OVERRIDES.get("count"))
    col_bytesin = guess_column(cols, ["bytesin"], COLUMN_OVERRIDES.get("bytesin"))

    if col_date is None or col_site is None:
        raise ValueError(
            f"Impossible de détecter automatiquement les colonnes date/site.\n"
            f"Colonnes disponibles dans le CSV : {cols}\n"
            f"-> Renseignez COLUMN_OVERRIDES en haut du script avec les noms exacts."
        )

    work = df.copy()
    work[col_date] = pd.to_datetime(work[col_date], errors="coerce", utc=True).dt.tz_localize(None)
    work = work.dropna(subset=[col_date, col_site])

    if col_count is not None:
        # --- Cas B : données déjà agrégées (site x heure) ---
        work["datetime_hour"] = work[col_date].dt.floor("h")
        agg = {col_count: "sum"}
        if col_bytesin:
            agg[col_bytesin] = "sum"
        ts = (work.groupby(["datetime_hour", col_site], as_index=False)
                   .agg(agg)
                   .rename(columns={col_site: "site", col_count: "nb_connexions"}))
        if col_bytesin:
            ts = ts.rename(columns={col_bytesin: "bytesin_total"})
        else:
            ts["bytesin_total"] = np.nan
    else:
        # --- Cas A : données brutes (1 ligne = 1 session) -> on agrège nous-mêmes ---
        work["datetime_hour"] = work[col_date].dt.floor("h")
        agg = {col_date: "count"}
        if col_bytesin:
            agg[col_bytesin] = "sum"
        ts = work.groupby(["datetime_hour", col_site]).agg(agg).reset_index()
        rename_map = {col_site: "site", col_date: "nb_connexions"}
        if col_bytesin:
            rename_map[col_bytesin] = "bytesin_total"
        ts = ts.rename(columns=rename_map)
        if not col_bytesin:
            ts["bytesin_total"] = np.nan

    # Grille complète site x heure d'ouverture (7h-23h), pour ne pas biaiser
    # les statistiques avec des trous silencieux (pas de session != donnée manquante).
    all_sites = ts["site"].unique()
    full_range = pd.date_range(ts["datetime_hour"].min(), ts["datetime_hour"].max(), freq="h")
    full_range = full_range[(full_range.hour >= 7) & (full_range.hour <= 23)]
    full_index = pd.MultiIndex.from_product([all_sites, full_range], names=["site", "datetime_hour"])
    ts_full = (ts.set_index(["site", "datetime_hour"])
                 .reindex(full_index, fill_value=0)
                 .reset_index())
    return ts_full


# ============================================================================
# FEATURE ENGINEERING (identique à la logique du notebook d'analyse)
# ============================================================================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["site", "datetime_hour"]).copy()
    df["hour"] = df["datetime_hour"].dt.hour
    df["dayofweek"] = df["datetime_hour"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["month"] = df["datetime_hour"].dt.month
    df["is_holiday"] = df["datetime_hour"].dt.strftime("%m-%d").isin(FR_HOLIDAYS_FIXED).astype(int)

    g = df.groupby("site")["nb_connexions"]
    df["lag_1h"] = g.shift(1)
    df["lag_24h"] = g.shift(17)
    df["lag_168h"] = g.shift(17 * 7)
    df["roll_mean_7d"] = g.transform(lambda s: s.shift(1).rolling(17 * 7, min_periods=17).mean())
    df["roll_std_7d"] = g.transform(lambda s: s.shift(1).rolling(17 * 7, min_periods=17).std())

    df["site_code"] = df["site"].astype("category").cat.codes
    return df


@st.cache_resource(show_spinner="Entraînement du modèle (Gradient Boosting)…")
def train_model(df_model_hash: str, df_model: pd.DataFrame):
    # NB : on choisit Gradient Boosting (et non un réseau de neurones) pour
    # l'app déployée : entraînement en quelques secondes, aucune dépendance
    # lourde (pas de TensorFlow) -> conforme à la conclusion du notebook
    # d'analyse (§9.2) qui recommande ce modèle pour la mise en production.
    model = HistGradientBoostingRegressor(max_depth=8, learning_rate=0.08, max_iter=300,
                                           random_state=RANDOM_STATE)
    model.fit(df_model[FEATURES], df_model["nb_connexions"])
    return model


def _df_hash(df: pd.DataFrame) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()


@st.cache_data(show_spinner="Calcul des prévisions…", ttl=3600)
def generate_forecast(_model, df_ts: pd.DataFrame, forecast_hours: int, data_hash: str) -> pd.DataFrame:
    all_sites = df_ts["site"].unique()
    last_ts = df_ts["datetime_hour"].max()
    future_hours = pd.date_range(last_ts + pd.Timedelta(hours=1), periods=forecast_hours, freq="h")
    future_hours = future_hours[(future_hours.hour >= 7) & (future_hours.hour <= 23)]

    future_rows = pd.MultiIndex.from_product([all_sites, future_hours], names=["site", "datetime_hour"]).to_frame(index=False)
    history_plus_future = pd.concat([
        df_ts[["site", "datetime_hour", "nb_connexions"]],
        future_rows.assign(nb_connexions=np.nan),
    ], ignore_index=True).sort_values(["site", "datetime_hour"])

    preds = {}
    for ts in sorted(future_rows["datetime_hour"].unique()):
        feat_now = build_features(history_plus_future[history_plus_future["datetime_hour"] <= ts])
        row_now = feat_now[feat_now["datetime_hour"] == ts].dropna(subset=FEATURES)
        if row_now.empty:
            continue
        yhat = np.clip(_model.predict(row_now[FEATURES]), 0, None)
        for site, val in zip(row_now["site"], yhat):
            preds[(site, ts)] = val
            history_plus_future.loc[
                (history_plus_future["site"] == site) & (history_plus_future["datetime_hour"] == ts),
                "nb_connexions"
            ] = val

    forecast_df = pd.Series(preds).rename_axis(["site", "datetime_hour"]).reset_index(name="charge_prevue")
    return forecast_df


@st.cache_data(show_spinner=False)
def compute_thresholds(df_ts: pd.DataFrame):
    high = df_ts.groupby("site")["nb_connexions"].quantile(0.90)
    med = df_ts.groupby("site")["nb_connexions"].quantile(0.70)
    return high, med


def classify(row, high, med):
    hi = high.get(row["site"], np.inf)
    m = med.get(row["site"], np.inf)
    if row["charge_prevue"] >= hi:
        return "Saturation probable"
    elif row["charge_prevue"] >= m:
        return "Vigilance"
    return "Normal"


@st.cache_data(show_spinner=False)
def pseudo_geo(sites):
    """Coordonnées factices mais STABLES (dérivées d'un hash du nom de site) pour
    afficher une carte quand aucun CSV de géolocalisation n'est fourni."""
    rows = []
    for s in sites:
        h = int(hashlib.md5(str(s).encode()).hexdigest(), 16)
        lat = 48.8566 + ((h % 2000) / 2000 - 0.5) * 0.10
        lon = 2.3522 + ((h // 2000 % 2000) / 2000 - 0.5) * 0.14
        rows.append({"site": s, "lat": lat, "lon": lon})
    return pd.DataFrame(rows)


# ============================================================================
# SIDEBAR — configuration & filtres
# ============================================================================
with st.sidebar:
    st.header("Charge Wi-Fi Paris")

    with st.expander("Source des données", expanded=False):
        csv_url = st.text_input("URL du CSV (GitHub raw)", value=DEFAULT_CSV_URL)
        geo_url = st.text_input("URL CSV géoloc (optionnel : site,lat,lon)", value=DEFAULT_GEO_URL)
        forecast_days = st.slider("Horizon de prévision (jours)", 1, FORECAST_MAX_DAYS, 7)

# ============================================================================
# PIPELINE (chargement -> features -> entraînement -> prévision), en cache
# ============================================================================
if not csv_url or "<votre-utilisateur>" in csv_url:
    st.warning("Renseignez l'URL de votre CSV GitHub (Raw) dans la barre latérale, "
               "ou modifiez `DEFAULT_CSV_URL` en haut du script.")
    st.stop()

try:
    raw_df = load_csv(csv_url)
    df_ts = prepare_timeseries(raw_df)
except Exception as e:
    st.error(f"Erreur de chargement/préparation des données : {e}")
    st.stop()

df_feat = build_features(df_ts)
df_model = df_feat.dropna().reset_index(drop=True)

if len(df_model) < 200:
    st.error("Pas assez d'historique exploitable après feature engineering (besoin d'au moins ~2-3 "
              "semaines de données par site pour calculer les retards/moyennes glissantes).")
    st.stop()

model = train_model(_df_hash(df_model), df_model)
forecast_hours = forecast_days * 17  # ~17h d'ouverture/jour
forecast_df = generate_forecast(model, df_ts, forecast_hours, _df_hash(df_ts))

if forecast_df.empty:
    st.error("La prévision n'a produit aucun résultat — vérifiez le format et la fraîcheur de vos données.")
    st.stop()

THRESHOLD_HIGH, THRESHOLD_MED = compute_thresholds(df_ts)
forecast_df["alerte"] = forecast_df.apply(lambda r: classify(r, THRESHOLD_HIGH, THRESHOLD_MED), axis=1)
forecast_df["recommandation"] = forecast_df["alerte"].map({
    "Saturation probable": "Renforcer bande passante / présence humaine, prioriser la maintenance préventive",
    "Vigilance": "Surveiller, prévoir un renfort si la tendance se confirme",
    "Normal": "RAS — allocation standard",
})

# Géolocalisation
if geo_url:
    try:
        geo_df = load_csv(geo_url)
        geo_cols = list(geo_df.columns)
        g_site = guess_column(geo_cols, ["site"])
        g_lat = guess_column(geo_cols, ["lat"])
        g_lon = guess_column(geo_cols, ["lon", "lng"])
        sites_meta = geo_df.rename(columns={g_site: "site", g_lat: "lat", g_lon: "lon"})[["site", "lat", "lon"]]
    except Exception:
        st.sidebar.warning("CSV de géolocalisation illisible — coordonnées factices utilisées à la place.")
        sites_meta = pseudo_geo(df_ts["site"].unique())
else:
    sites_meta = pseudo_geo(df_ts["site"].unique())

# ============================================================================
# DASHBOARD
# ============================================================================
st.title("Prévision de la charge des points Wi-Fi — Paris")
st.caption("Aide à la décision pour l'allocation des ressources humaines, matérielles et commerciales")

dates_dispo = sorted(forecast_df["datetime_hour"].dt.date.unique())
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    date_sel = st.selectbox("Date", dates_dispo, index=0)
with c2:
    heures_dispo = sorted(forecast_df.loc[forecast_df["datetime_hour"].dt.date == date_sel, "datetime_hour"].dt.hour.unique())
    heure_sel = st.select_slider("Heure", options=heures_dispo, value=heures_dispo[0])
with c3:
    sites_sel = st.multiselect("Filtrer par site (vide = tous)", sorted(forecast_df["site"].unique()))

mask_time = (forecast_df["datetime_hour"].dt.date == date_sel) & (forecast_df["datetime_hour"].dt.hour == heure_sel)
snapshot = forecast_df[mask_time].copy()
if sites_sel:
    snapshot = snapshot[snapshot["site"].isin(sites_sel)]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Charge totale prévue", f"{snapshot['charge_prevue'].sum():.0f} connexions")
col2.metric("Sites en alerte", int((snapshot["alerte"] != "Normal").sum()))
if not snapshot.empty:
    top_site = snapshot.loc[snapshot["charge_prevue"].idxmax()]
    col3.metric("Site le plus chargé", top_site["site"], f"{top_site['charge_prevue']:.0f}")
else:
    col3.metric("Site le plus chargé", "—")
col4.metric("Créneau", f"{date_sel} — {heure_sel}h")

st.divider()

left, right = st.columns([2, 1])
with left:
    st.subheader("Où la charge augmente-t-elle ?")
    map_df = snapshot.merge(sites_meta, on="site", how="left")
    if not map_df.empty and map_df["lat"].notna().any():
        fig_map = px.scatter_mapbox(
            map_df, lat="lat", lon="lon", size="charge_prevue", color="alerte",
            color_discrete_map={"Normal": "green", "Vigilance": "orange", "Saturation probable": "red"},
            hover_name="site", hover_data={"charge_prevue": True, "lat": False, "lon": False},
            zoom=11, height=500,
        )
        fig_map.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_map, use_container_width=True)
        if not geo_url:
            st.caption("Coordonnées approximatives (aucun CSV de géolocalisation fourni) — "
                       "renseignez `URL CSV géoloc` dans la barre latérale pour une carte précise.")
    else:
        st.info("Coordonnées géographiques indisponibles pour cette sélection.")

with right:
    st.subheader("Sites à risque")
    alert_table = (snapshot[snapshot["alerte"] != "Normal"]
                   .sort_values("charge_prevue", ascending=False)
                   [["site", "charge_prevue", "alerte", "recommandation"]])
    st.dataframe(alert_table, use_container_width=True, hide_index=True)
    st.download_button("Exporter les alertes (CSV)", alert_table.to_csv(index=False), "alertes_wifi.csv")

st.divider()

st.subheader("Quand la demande va-t-elle augmenter ? — Évolution sur l'horizon choisi")
site_detail = st.selectbox("Choisir un site", sorted(forecast_df["site"].unique()))
detail_df = forecast_df[forecast_df["site"] == site_detail]
fig_line = go.Figure()
fig_line.add_trace(go.Scatter(x=detail_df["datetime_hour"], y=detail_df["charge_prevue"],
                               mode="lines+markers", marker=dict(size=4), name="Charge prévue"))
fig_line.add_hline(y=THRESHOLD_HIGH.get(site_detail, np.nan), line_dash="dash", line_color="red",
                    annotation_text="Seuil haut")
fig_line.add_hline(y=THRESHOLD_MED.get(site_detail, np.nan), line_dash="dash", line_color="orange",
                    annotation_text="Seuil moyen")
fig_line.update_layout(title=f"Charge prévue — {site_detail}", xaxis_title="Date/heure",
                        yaxis_title="Nb connexions prévu")
st.plotly_chart(fig_line, use_container_width=True)

st.caption("Application autonome : les données sont chargées directement depuis le CSV GitHub configuré "
           "et le modèle (Gradient Boosting) est ré-entraîné à chaque changement de données (mis en cache). "
           "Les seuils d'alerte sont calibrés sur les quantiles 70 % / 90 % de l'historique par site et "
           "doivent être validés avec les équipes opérationnelles.")
