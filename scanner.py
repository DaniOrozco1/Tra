import os
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ============================ CONFIGURACIÓN ============================
TZ = ZoneInfo("Europe/Madrid")
HORAS_AVISO = set(range(8, 24)) | {0}   # de 8:00 a 00:00 (hora de Madrid)
SEND_EMPTY = False                      # True: mensaje en cada ejecución aunque no haya señales

MA_FAST, MA_SLOW = 20, 60
VELA_MIN = 3                            # minutos por vela
MODO_VELA = "CIERRE"                    # "CUERPO": el cuerpo entero dentro de la zona
                                        # "CIERRE": basta con que el cierre esté dentro
MAX_EDAD_MIN = 30                       # ignora activos con datos más viejos (mercado cerrado)

# ============================== ACTIVOS ================================
FOREX_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD",
    "EURGBP", "EURJPY", "EURCHF", "EURCAD", "EURAUD", "EURNZD",
    "GBPJPY", "GBPCHF", "GBPCAD", "GBPAUD", "GBPNZD",
    "AUDJPY", "AUDCHF", "AUDCAD", "AUDNZD",
    "NZDJPY", "NZDCHF", "NZDCAD", "CADJPY", "CADCHF", "CHFJPY",
]
ASSETS = [(p, p + "=X") for p in FOREX_PAIRS]
ASSETS += [
    ("Petróleo WTI", "CL=F"), ("Petróleo Brent", "BZ=F"),
    ("Oro", "GC=F"), ("Plata", "SI=F"),
    ("Gas natural", "NG=F"), ("Cobre", "HG=F"), ("Platino", "PL=F"),
    ("S&P 500", "^GSPC"), ("Nasdaq 100", "^NDX"), ("Dow Jones", "^DJI"),
    ("DAX", "^GDAXI"), ("IBEX 35", "^IBEX"), ("FTSE 100", "^FTSE"),
    ("CAC 40", "^FCHI"), ("Euro Stoxx 50", "^STOXX50E"), ("Nikkei 225", "^N225"),
]
for _c in ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX", "TRX",
           "LINK", "DOT", "LTC", "BCH", "ATOM", "XLM", "NEAR"]:
    ASSETS.append((_c + "/USD", _c + "-USD"))


# ============================== TELEGRAM ===============================
def send_telegram(text):
    token = os.environ["TELEGRAM_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    parts, current = [], ""
    for block in text.split("\n\n"):
        if len(current) + len(block) + 2 > 3800:
            parts.append(current)
            current = ""
        current += block + "\n\n"
    parts.append(current)
    for p in parts:
        if not p.strip():
            continue
        r = requests.post(url, data={"chat_id": chat_id, "text": p.strip()}, timeout=30)
        if not r.ok:
            raise RuntimeError(f"Telegram devolvió error {r.status_code}: {r.text[:200]}")


# =============================== DATOS =================================
def fetch(ticker, retries=3):
    """Velas de 1 minuto (Yahoo no ofrece 3m); luego se agrupan."""
    for _ in range(retries):
        try:
            df = yf.Ticker(ticker).history(period="5d", interval="1m", auto_adjust=False)
            if df is not None and len(df) > 0:
                df = df[["Open", "High", "Low", "Close"]].dropna()
                df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
                if len(df) > 0:
                    return df
        except Exception:
            pass
        time.sleep(1.5)
    return None


def to_3m(df, now):
    v = df.resample(f"{VELA_MIN}min").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    ).dropna()
    ends = v.index + pd.Timedelta(minutes=VELA_MIN)
    return v[ends <= now]          # quita la vela que todavía se está formando


# ============================ INDICADORES ==============================
def heikin_ashi(df):
    o, h, l, c = (df[x].to_numpy(float) for x in ("Open", "High", "Low", "Close"))
    hc = (o + h + l + c) / 4
    ho = np.empty_like(hc)
    ho[0] = (o[0] + c[0]) / 2
    for i in range(1, len(hc)):
        ho[i] = (ho[i - 1] + hc[i - 1]) / 2
    hh = np.maximum.reduce([h, ho, hc])
    hl = np.minimum.reduce([l, ho, hc])
    return pd.DataFrame({"o": ho, "h": hh, "l": hl, "c": hc}, index=df.index)


# ============================== ANÁLISIS ===============================
def analyze(v, now):
    if len(v) < MA_SLOW + 20:
        return None

    # mercado cerrado / datos viejos
    edad = (now - (v.index[-1] + pd.Timedelta(minutes=VELA_MIN))).total_seconds() / 60
    if edad > MAX_EDAD_MIN:
        return None

    ha = heikin_ashi(v)
    ma20 = ha.c.rolling(MA_FAST).mean()
    ma60 = ha.c.rolling(MA_SLOW).mean()
    i = len(ha) - 1

    lo = min(ma20.iloc[i], ma60.iloc[i])
    hi = max(ma20.iloc[i], ma60.iloc[i])
    if np.isnan(lo) or np.isnan(hi):
        return None

    c = ha.c.iloc[i]
    if MODO_VELA == "CUERPO":
        body_lo = min(ha.o.iloc[i], c)
        body_hi = max(ha.o.iloc[i], c)
        dentro = body_lo >= lo and body_hi <= hi
    else:
        dentro = lo <= c <= hi
    if not dentro:
        return None

    cierre_vela = (v.index[-1] + pd.Timedelta(minutes=VELA_MIN)).tz_convert(TZ)
    return {"close": float(c), "ma20": float(ma20.iloc[i]),
            "ma60": float(ma60.iloc[i]), "hora": cierre_vela}


def fmt(x):
    return f"{x:,.5f}" if abs(x) < 10 else f"{x:,.2f}"


# ================================ MAIN =================================
def main():
    now = pd.Timestamp.now(tz="UTC")
    ahora_es = now.tz_convert(TZ)
    if os.getenv("GITHUB_EVENT_NAME") == "schedule" and ahora_es.hour not in HORAS_AVISO:
        print(f"Fuera de horario ({ahora_es:%H:%M} Madrid), no se ejecuta.")
        return

    avisos = []
    for name, ticker in ASSETS:
        try:
            df = fetch(ticker)
            if df is None:
                print(f"Sin datos: {name}")
                continue
            r = analyze(to_3m(df, now), now)
            if r:
                avisos.append(
                    f"{name}: HA {fmt(r['close'])} dentro de MA20 ({fmt(r['ma20'])}) "
                    f"y MA60 ({fmt(r['ma60'])}) - vela cerrada {r['hora']:%H:%M}"
                )
        except Exception as e:
            print(f"Error con {name}: {e}")
        time.sleep(0.2)

    if avisos:
        send_telegram(f"Velas HA {VELA_MIN}m dentro de la zona MA20-MA60 ({ahora_es:%H:%M}):\n\n"
                      + "\n".join(avisos))
    elif SEND_EMPTY:
        send_telegram(f"{ahora_es:%H:%M} - Ningún activo cumple la condición.")
    else:
        print("Ningún activo cumple la condición.")


if __name__ == "__main__":
    main()
