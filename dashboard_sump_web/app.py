"""
Backend API — Dashboard Prediksi Elevasi Muka Air Sump
=========================================================

Menyediakan 4 endpoint yang dipakai oleh static/script.js:
  GET  /api/model-info          -> metrik performa model (RMSE, MAE, R², dsb)
  GET  /api/historical          -> data historis untuk grafik tren
  POST /api/predict             -> prediksi elevasi besok
  GET  /api/feature-importance  -> kontribusi tiap fitur ke prediksi

PERUBAHAN UTAMA dari versi sebelumnya:
  - "Debit Air Limpasan" TIDAK diinput manual lagi. Dihitung otomatis dari
    Curah Hujan, Durasi Hujan, Koefisien Limpasan (C), dan Luas Catchment
    Area, memakai METODE RASIONAL: Q = C x I x A / 3.6
    (I = intensitas hujan = curah_hujan / durasi_hujan, dalam mm/jam;
     A dalam km²; Q dalam m³/detik). Dicocokkan ke data historis Anda,
     akurasinya >99.9%.
  - "Volume Sump" TIDAK diinput manual lagi. Dihitung otomatis dari Elevasi
    Muka Air Sump hari ini, memakai regresi kuadratik yang di-fit dari
    545 data historis Anda (R² = 0.9998) — karena Volume Sump ternyata
    murni fungsi dari elevasinya (bentuk cekungan sump).
  - "Akumulasi 3/5/7 Hari" dijadikan opsional/lanjutan. Kalau tidak diisi
    manual, diestimasi otomatis dari Curah Hujan hari ini (curah x 3/5/7)
    -- estimasi kasar, tapi pengaruh fitur ini ke prediksi sangat kecil
    (<1.5% gabungan), jadi dampaknya ke akurasi prediksi minimal.
  - Status AMAN/WASPADA/KRITIS ditambahkan berdasar ambang batas elevasi
    yang bisa diatur (default kritis di -14 m, sesuai permintaan).
"""
import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model_rf_elevasi_sump.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "model", "feature_columns.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset_final.xlsx")

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)

# ---------------------------------------------------------------------------
# Muat data historis sekali saat server start (dipakai untuk grafik & metrik)
# ---------------------------------------------------------------------------
_df = pd.read_excel(DATA_PATH)
_df = _df.sort_values("Tanggal").reset_index(drop=True)

# --- Fit kurva Volume Sump = f(Elevasi) dari data historis (kuadratik) -----
_vol_coeffs = np.polyfit(
    _df["Elevasi Muka Air Sump (m)"], _df["Volume Sump (m³)"], deg=2
)

def hitung_volume_sump(elevasi_m: float) -> float:
    """Volume sump (m³) diestimasi dari elevasi, pakai kurva hasil fit data historis."""
    return float(np.polyval(_vol_coeffs, elevasi_m))


def hitung_debit_limpasan(curah_hujan, durasi_hujan, koef_c, luas_km2) -> float:
    """Metode Rasional: Q (m³/detik) = C x I x A / 3.6, I = curah/durasi (mm/jam)."""
    if not durasi_hujan or durasi_hujan <= 0:
        return 0.0
    intensitas = curah_hujan / durasi_hujan  # mm/jam
    return float(koef_c * intensitas * luas_km2 / 3.6)


# --- Evaluasi performa model (RMSE/MAE/R²) memakai target elevasi besok ----
def _evaluasi_model():
    df = _df.copy()
    df["target_besok"] = df["Elevasi Muka Air Sump (m)"].shift(-1)
    df["Delta Elevasi (m)"] = (
        df["Elevasi Muka Air Sump (m)"] - df["Elevasi Muka Air Sump Kemarin (m)"]
    )
    df = df.dropna(subset=["target_besok"])

    X = df[feature_columns]
    y_true = df["target_besok"].values
    y_pred = model.predict(X)

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot)
    return rmse, mae, r2, len(df)


_RMSE, _MAE, _R2, _N = _evaluasi_model()

# --- Ambang batas status (default; sama untuk semua pengguna dashboard) ----
DEFAULT_KRITIS = -14.0
DEFAULT_WASPADA = -15.0


def tentukan_status(elevasi: float, batas_waspada=DEFAULT_WASPADA, batas_kritis=DEFAULT_KRITIS) -> str:
    """Elevasi lebih besar (kurang negatif) = air lebih tinggi = lebih berisiko."""
    if elevasi >= batas_kritis:
        return "kritis"
    if elevasi >= batas_waspada:
        return "waspada"
    return "aman"


# ---------------------------------------------------------------------------
# ROUTES — static frontend
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# ---------------------------------------------------------------------------
# ROUTES — API
# ---------------------------------------------------------------------------
@app.route("/api/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "algoritma": "Random Forest Regressor",
        "n_estimators": getattr(model, "n_estimators", None),
        "rmse": _RMSE,
        "mae": _MAE,
        "r2": _R2,
        "jumlah_data": _N,
    })


@app.route("/api/historical", methods=["GET"])
def historical():
    days = request.args.get("days", default=0, type=int)
    df = _df if days <= 0 else _df.tail(days)
    return jsonify({
        "tanggal": df["Tanggal"].dt.strftime("%Y-%m-%d").tolist(),
        "elevasi": df["Elevasi Muka Air Sump (m)"].round(3).tolist(),
        "curah_hujan": df["Curah Hujan (mm)"].tolist(),
    })


@app.route("/api/feature-importance", methods=["GET"])
def feature_importance():
    items = [
        {"fitur": f, "importance": float(imp)}
        for f, imp in zip(feature_columns, model.feature_importances_)
    ]
    items.sort(key=lambda x: x["importance"], reverse=True)
    return jsonify(items)


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"sukses": False, "error": "Body request harus berupa JSON."}), 400

    # --- field wajib diisi manual oleh pengguna ---
    wajib = [
        "curah_hujan", "durasi_hujan", "elevasi_kemarin", "elevasi_hari_ini",
        "koefisien_limpasan", "luas_catchment_area",
    ]
    missing = [f for f in wajib if f not in payload or payload[f] in (None, "")]
    if missing:
        return jsonify({"sukses": False, "error": f"Field berikut belum diisi: {missing}"}), 400

    try:
        curah_hujan = float(payload["curah_hujan"])
        durasi_hujan = float(payload["durasi_hujan"])
        elevasi_kemarin = float(payload["elevasi_kemarin"])
        elevasi_hari_ini = float(payload["elevasi_hari_ini"])
        koef_c = float(payload["koefisien_limpasan"])
        luas_a = float(payload["luas_catchment_area"])
    except (TypeError, ValueError):
        return jsonify({"sukses": False, "error": "Semua nilai input harus berupa angka."}), 400

    if not (0.0 <= koef_c <= 1.0):
        return jsonify({"sukses": False, "error": "Koefisien Limpasan (C) harus di antara 0.0 - 1.0."}), 400

    # --- field lanjutan/opsional: akumulasi 3/5/7 hari ---
    # Kalau tidak dikirim (atau kosong), diestimasi otomatis dari curah hujan
    # hari ini x jumlah hari (estimasi kasar; pengaruh fitur ini ke prediksi
    # sangat kecil, <1.5% gabungan, sesuai analisis feature importance).
    def _get_or_estimate(key, n_hari):
        val = payload.get(key)
        if val in (None, ""):
            return curah_hujan * n_hari
        return float(val)

    akum_3 = _get_or_estimate("akumulasi_3", 3)
    akum_5 = _get_or_estimate("akumulasi_5", 5)
    akum_7 = _get_or_estimate("akumulasi_7", 7)

    # --- field yang dihitung otomatis ---
    debit_limpasan = hitung_debit_limpasan(curah_hujan, durasi_hujan, koef_c, luas_a)
    volume_sump = hitung_volume_sump(elevasi_hari_ini)
    delta_elevasi = elevasi_hari_ini - elevasi_kemarin

    fitur_values = {
        "Curah Hujan (mm)": curah_hujan,
        "Durasi Hujan (jam)": durasi_hujan,
        "Akumulasi 3 Hari (mm)": akum_3,
        "Akumulasi 5 Hari (mm)": akum_5,
        "Akumulasi 7 Hari (mm)": akum_7,
        "Debit Air Limpasan (m³/detik)": debit_limpasan,
        "Elevasi Muka Air Sump Kemarin (m)": elevasi_kemarin,
        "Elevasi Muka Air Sump (m)": elevasi_hari_ini,
        "Volume Sump (m³)": volume_sump,
        "Delta Elevasi (m)": delta_elevasi,
    }

    try:
        x_row = [fitur_values[col] for col in feature_columns]
    except KeyError as e:
        return jsonify({"sukses": False, "error": f"Kolom fitur model tidak dikenali: {e}"}), 500

    X = pd.DataFrame([x_row], columns=feature_columns)
    prediksi = float(model.predict(X)[0])

    # ambang batas status - boleh dikustomisasi lewat request, default -14/-15
    batas_waspada = float(payload.get("batas_waspada", DEFAULT_WASPADA))
    batas_kritis = float(payload.get("batas_kritis", DEFAULT_KRITIS))
    status = tentukan_status(prediksi, batas_waspada, batas_kritis)

    tree_preds = np.array([t.predict(X.values)[0] for t in model.estimators_])

    return jsonify({
        "sukses": True,
        "prediksi_elevasi_besok": round(prediksi, 3),
        "elevasi_hari_ini": round(elevasi_hari_ini, 3),
        "delta_elevasi_input": round(delta_elevasi, 3),
        "debit_air_limpasan_terhitung": round(debit_limpasan, 4),
        "volume_sump_terhitung": round(volume_sump, 2),
        "status": status,
        "std_dev_pohon": round(float(tree_preds.std()), 4),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
