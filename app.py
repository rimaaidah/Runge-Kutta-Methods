import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="RK4 Manual — COVID Kota Bandung", layout="wide")

# ----------------------------
# RK4 MANUAL (from scratch)
# ----------------------------
def deriv(state, t, params):
    # Logistic: dP/dt = r*P*(1 - P/K)
    r, K = params
    P = state[0]
    dP = r * P * (1.0 - P / K)
    return np.array([dP], float)

def rk4_simulate(params, t_start, t_final, h, y0):
    taxis = np.array([], float)
    yaxis = np.array([], float)

    s = np.array([y0], float)
    t = float(t_start)
    tf = float(t_final)

    while t <= tf:
        taxis = np.append(taxis, t)
        yaxis = np.append(yaxis, s[0])

        k1 = h * deriv(s, t, params)
        k2 = h * deriv(s + k1/2.0, t + h/2.0, params)
        k3 = h * deriv(s + k2/2.0, t + h/2.0, params)
        k4 = h * deriv(s + k3,     t + h,     params)

        s += (k1 + 2*k2 + 2*k3 + k4) / 6.0
        t += h

    return taxis, yaxis

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def safe_read_csv(uploaded_file, fallback_path):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.read_csv(fallback_path)

# ----------------------------
# UI
# ----------------------------
st.title("Simulasi Data-Driven dengan RK4 Manual (Logistic) Kota Bandung")

with st.sidebar:
    st.header("Input Data")
    uploaded = st.file_uploader("Upload CSV (opsional)", type=["csv"])
    fallback_path = st.text_input("Atau pakai file lokal:", "covid_jabar_perkembangan_harian.csv")

    st.header("Filter")
    wilayah = st.text_input("Wilayah (default)", "Kota Bandung")
    target_col = st.text_input("Kolom observasi", "konfirmasi_total")

    st.header("Periode (opsional)")
    use_date_filter = st.checkbox("Aktifkan filter tanggal", value=True)
    start_date = st.date_input("Start date", value=pd.to_datetime("2020-03-01"))
    end_date   = st.date_input("End date", value=pd.to_datetime("2021-03-01"))

    st.header("RK4 Settings")
    h = st.number_input("Step size h (hari)", min_value=0.01, max_value=10.0, value=1.0, step=0.25)

    st.header("Trial (Manual)")
    r_try = st.number_input("r (trial)", min_value=0.0, max_value=10.0, value=0.10, step=0.01)
    K_mult_try = st.number_input("K multiplier (trial) x y_last", min_value=1.0, max_value=50.0, value=1.5, step=0.1)

    st.header("Grid Search (Otomatis)")
    do_grid = st.checkbox("Jalankan Grid Search", value=True)
    r_min = st.number_input("r min", min_value=0.0, max_value=10.0, value=0.001, step=0.001, format="%.3f")
    r_max = st.number_input("r max", min_value=0.0, max_value=10.0, value=0.60, step=0.01)
    r_n   = st.slider("Jumlah titik r", min_value=10, max_value=200, value=90, step=5)

    K_min_mult = st.number_input("K min multiplier x y_last", min_value=1.0, max_value=200.0, value=1.02, step=0.01)
    K_max_mult = st.number_input("K max multiplier x y_last", min_value=1.0, max_value=200.0, value=3.00, step=0.10)
    K_n        = st.slider("Jumlah titik K", min_value=10, max_value=200, value=90, step=5)

    run_btn = st.button("Jalankan")

# ----------------------------
# Main logic
# ----------------------------
if run_btn:
    try:
        df = safe_read_csv(uploaded, fallback_path)
    except Exception as e:
        st.error(f"Gagal membaca CSV: {e}")
        st.stop()

    # Basic column checks
    required_cols = {"tanggal", "nama_kab_kota", target_col}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Kolom tidak ditemukan di CSV: {missing}")
        st.stop()

    # Parse date
    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
    df = df.dropna(subset=["tanggal", "nama_kab_kota", target_col])

    # Filter wilayah
    dff = df[df["nama_kab_kota"].astype(str).str.lower() == wilayah.lower()].copy()
    if dff.empty:
        st.error(f"Data wilayah '{wilayah}' tidak ditemukan. Coba cek penulisan nama_kab_kota.")
        st.stop()

    dff = dff.sort_values("tanggal")

    # Filter tanggal
    if use_date_filter:
        sd = pd.to_datetime(start_date)
        ed = pd.to_datetime(end_date)
        dff = dff[(dff["tanggal"] >= sd) & (dff["tanggal"] <= ed)].copy()

    dff = dff[["tanggal", target_col]].dropna()
    if len(dff) < 5:
        st.error("Data terlalu sedikit setelah filter. Perlu minimal beberapa titik data.")
        st.stop()

    # Build observation arrays
    t0 = dff["tanggal"].iloc[0]
    t_obs = (dff["tanggal"] - t0).dt.days.to_numpy(dtype=float)
    y_obs = pd.to_numeric(dff[target_col], errors="coerce").to_numpy(dtype=float)

    m = np.isfinite(t_obs) & np.isfinite(y_obs)
    t_obs, y_obs = t_obs[m], y_obs[m]

    if len(t_obs) < 5:
        st.error("Data valid terlalu sedikit (banyak NaN).")
        st.stop()

    y0 = float(y_obs[0])
    t_start = float(t_obs.min())
    t_final = float(t_obs.max())

    # ---------------- Trial
    K_try = float(y_obs[-1]) * float(K_mult_try)
    t_trial, y_trial_grid = rk4_simulate((r_try, K_try), t_start, t_final, float(h), y0)
    y_trial = np.interp(t_obs, t_trial, y_trial_grid)
    rmse_trial = rmse(y_obs, y_trial)

    # ---------------- Grid Search
    best = None
    if do_grid:
        r_grid = np.linspace(float(r_min), float(r_max), int(r_n))
        K_grid = np.linspace(float(y_obs[-1]) * float(K_min_mult), float(y_obs[-1]) * float(K_max_mult), int(K_n))

        best_rmse = np.inf
        best_r, best_K = None, None
        best_y = None

        prog = st.progress(0)
        total = len(r_grid) * len(K_grid)
        done = 0

        for r in r_grid:
            for K in K_grid:
                tt, yy = rk4_simulate((r, K), t_start, t_final, float(h), y0)
                y_sim = np.interp(t_obs, tt, yy)
                e = rmse(y_obs, y_sim)

                if e < best_rmse:
                    best_rmse = e
                    best_r = float(r)
                    best_K = float(K)
                    best_y = y_sim

                done += 1
            prog.progress(min(1.0, done / total))

        best = {"rmse": best_rmse, "r": best_r, "K": best_K, "y": best_y}

    # ---------------- Display
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Ringkasan Data")
        st.write(f"**Wilayah:** {wilayah}")
        if use_date_filter:
            st.write(f"**Periode:** {pd.to_datetime(start_date).date()} s/d {pd.to_datetime(end_date).date()}")
        st.write(f"**Jumlah data:** {len(t_obs)}")
        st.write(f"**y0:** {y0:,.2f}")
        st.write(f"**y_last:** {float(y_obs[-1]):,.2f}")
        st.dataframe(dff.head(10))

    with col2:
        st.subheader("Hasil Tuning")
        st.write(f"**Trial:** r={r_try:.4f}, K={K_try:,.2f}, RMSE={rmse_trial:,.4f}")
        if best is not None:
            st.write(f"**Best Grid:** r={best['r']:.4f}, K={best['K']:,.2f}, RMSE={best['rmse']:,.4f}")

    # Plot overlay
    st.subheader("Overlay Plot: Data vs Simulasi RK4 Manual")
    fig = plt.figure()
    plt.scatter(t_obs, y_obs, s=12, label="Data asli")
    plt.plot(t_obs, y_trial, label="RK4 manual (trial)")

    if best is not None:
        plt.plot(t_obs, best["y"], label="RK4 manual (best grid)")

    plt.xlabel("Hari ke-n")
    plt.ylabel(target_col)
    plt.title(f"Kota Bandung — Logistic Fit (RK4 manual)")
    plt.grid(True)
    plt.legend()
    st.pyplot(fig)

    # Tips interpretasi singkat
    st.info(
        "Interpretasi: r menunjukkan laju pertumbuhan awal; K adalah batas maksimum (carrying capacity). "
        "RMSE makin kecil → kurva simulasi makin mendekati data."
    )

else:
    st.write("Atur parameter di sidebar, lalu klik **Jalankan**.")
    st.caption("Pastikan file CSV ada (upload atau file lokal) dan kolom: tanggal, nama_kab_kota, konfirmasi_total.")
