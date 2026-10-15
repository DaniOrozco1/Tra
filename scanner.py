"""
Escáner Heikin Ashi 1h -> aviso por Telegram.

Condiciones (resumen):
  1. Cambio de color en la vela Heikin Ashi (rojo->verde = compra, verde->rojo = venta).
  2. El cuerpo de la última vela de la tendencia anterior estaba dentro de la zona MA20-MA60.
  3. Estocástico (14,1,3) < 60 en compras, > 40 en ventas.
  4. La MA60 sube (compras) o baja (ventas) entre 0,5 y 1,1 "cuadrados" en las últimas 12 velas.
  5. Si cumple todo, avisa con entrada, stop, objetivo y unidades para arriesgar 5 EUR.
"""
import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ============================ CONFIGURACIÓN ============================
TZ = ZoneInfo("Europe/Madrid")
HORAS_AVISO = set(range(8, 24)) | {0}      # de 8:00 a 00:00 (hora de Madrid)

MA_FAST, MA_SLOW = 20, 60

STOCH_K_LEN, STOCH_K_SMOOTH, STOCH_D_SMOOTH = 14, 1, 3
STOCH_LINE = "K"             # "K" (línea rápida) o "D" (línea suavizada)
STOCH_BUY_BELOW = 60
STOCH_SELL_ABOVE = 40

# Pendiente de la MA60 medida en "cuadrados" (base = 12 velas, base = 1,6 x altura)
SLOPE_BARS = 12
SLOPE_MIN, SLOPE_MAX = 0.5, 1.1
SQUARE_RATIO = 1.6
VISIBLE_BARS = 120           # velas que se ven en tu gráfico (para el zoom vertical)
PLOT_ASPECT = 2.0            # ancho / alto de la zona del gráfico

RISK_EUR = 5.0               # lo máximo que puedes perder por operación
RR_NORMAL = 2.0              # ganar 2 veces lo arriesgado
RR_NEAR_MA60 = 2.5           # si el precio está más cerca de la MA60 que de la MA20

USE_HA_FOR_INDICATORS = True  # medias y estocástico sobre velas Heikin Ashi (como en un gráfico HA)
STOP_USES_HA = True           # stop en mínimos/máximos de las velas Heikin Ashi

# ============================== ACTIVOS ================================
FOREX_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD",
    "EURGBP", "EURJPY", "EURCHF", "EURCAD", "EURAUD", "EURNZD",
    "GBPJPY", "GBPCHF", "GBPCAD", "GBPAUD", "GBPNZD",
    "AUDJPY", "AUDCHF", "AUDCAD", "AUDNZD",
    "NZDJPY", "NZDCHF", "NZDCAD", "CADJPY", "CADCHF", "CHFJPY",
]
ASSETS = [(p, p + "=X", p[3:]) for p in FOREX_PAIRS]
ASSETS += [
    ("Petróleo WTI", "CL=F", "USD"), ("Petróleo Brent", "BZ=F", "USD"),
    ("Oro", "GC=F", "USD"), ("Plata", "SI=F", "USD"),
    ("Gas natural", "NG=F", "USD"), ("Cobre", "HG=F", "USD"), ("Platino", "PL=F", "USD"),
    ("S&P 500", "^GSPC", "USD"), ("Nasdaq 100", "^NDX", "USD"), ("Dow Jones", "^DJI", "USD"),
    ("DAX", "^GDAXI", "EUR"), ("IBEX 35", "^IBEX", "EUR"), ("FTSE 100", "^FTSE", "GBP"),
    ("CAC 40", "^FCHI", "EUR"), ("Euro Stoxx 50", "^STOXX50E", "EUR"), ("Nikkei 225", "^N225", "JPY"),
]
for _c in ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX", "TRX",
           "LINK", "DOT", "LTC", "BCH", "ATOM", "XLM", "NEAR"]:
    ASSETS.append((_c + "/USD", _c + "-USD", "USD"))


# ============================== TELEGRAM ===============================
def send_telegram(text):
    token = os.environ["TELEGRAM_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    # Telegram admite 4096 caracteres por mensaje: se parte por bloques si hace falta
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
def fetch(ticker, period="30d", retries=3):
    for _ in range(retries):
        try:
            df = yf.Ticker(ticker).history(period=period, interval="1h", auto_adjust=False)
            if df is not None and len(df) > 0:
                df = df[["Open", "High", "Low", "Close"]].dropna()
                df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
                if len(df) > 0:
                    return df
        except Exception:
            pass
        time.sleep(1.5)
    return None


def only_closed(df, now):
    """Quita la vela que todavía se está formando."""
    ends = df.index + pd.Timedelta(hours=1)
    return df[ends <= now]


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


def stochastic(h, l, c):
    ll = l.rolling(STOCH_K_LEN).min()
    hh = h.rolling(STOCH_K_LEN).max()
    raw = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    k = raw.rolling(STOCH_K_SMOOTH).mean()
    d = k.rolling(STOCH_D_SMOOTH).mean()
    return k if STOCH_LINE == "K" else d


# ============================== ANÁLISIS ===============================
def analyze(df, to_eur):
    """Devuelve un dict con la operación si la última vela cerrada cumple todo, o None."""
    if len(df) < max(150, VISIBLE_BARS + 5):
        return None

    ha = heikin_ashi(df)
    if USE_HA_FOR_INDICATORS:
        ind = ha
    else:
        ind = pd.DataFrame({"o": df.Open, "h": df.High, "l": df.Low, "c": df.Close})

    ma20 = ind.c.rolling(MA_FAST).mean()
    ma60 = ind.c.rolling(MA_SLOW).mean()
    st = stochastic(ind.h, ind.l, ind.c)

    bull = ha.c > ha.o
    bear = ha.c < ha.o
    i = len(ha) - 1          # última vela cerrada = primera vela de la nueva tendencia

    # 1) cambio de color
    if bull.iloc[i] and bear.iloc[i - 1]:
        side = "BUY"
    elif bear.iloc[i] and bull.iloc[i - 1]:
        side = "SELL"
    else:
        return None

    # 2) cuerpo de la última vela de la tendencia anterior dentro de la zona MA20-MA60
    lo = min(ma20.iloc[i - 1], ma60.iloc[i - 1])
    hi = max(ma20.iloc[i - 1], ma60.iloc[i - 1])
    body_lo = min(ha.o.iloc[i - 1], ha.c.iloc[i - 1])
    body_hi = max(ha.o.iloc[i - 1], ha.c.iloc[i - 1])
    if not (body_lo >= lo and body_hi <= hi):
        return None

    # 3) estocástico
    s = st.iloc[i]
    if np.isnan(s):
        return None
    if side == "BUY" and not s < STOCH_BUY_BELOW:
        return None
    if side == "SELL" and not s > STOCH_SELL_ABOVE:
        return None

    # 4) pendiente de la MA60 en "cuadrados"
    win = ind.iloc[-VISIBLE_BARS:]
    price_range = win.h.max() - win.l.min()
    if price_range <= 0:
        return None
    square_h = price_range * (SLOPE_BARS / VISIBLE_BARS) * PLOT_ASPECT / SQUARE_RATIO
    slope = (ma60.iloc[i] - ma60.iloc[i - SLOPE_BARS]) / square_h
    if np.isnan(slope):
        return None
    if side == "BUY" and not (SLOPE_MIN <= slope <= SLOPE_MAX):
        return None
    if side == "SELL" and not (-SLOPE_MAX <= slope <= -SLOPE_MIN):
        return None

    # 5) operación
    entry = float(df.Close.iloc[i])
    if STOP_USES_HA:
        lows, highs = ha.l, ha.h
    else:
        lows, highs = df.Low, df.High
    if side == "BUY":
        sl = float(min(lows.iloc[i], lows.iloc[i - 1]))
        risk = entry - sl
    else:
        sl = float(max(highs.iloc[i], highs.iloc[i - 1]))
        risk = sl - entry
    if risk <= 0:
        return None

    ref = ind.c.iloc[i]
    near60 = abs(ref - ma60.iloc[i]) < abs(ref - ma20.iloc[i])
    rr = RR_NEAR_MA60 if near60 else RR_NORMAL
    tp = entry + risk * rr if side == "BUY" else entry - risk * rr
    units = RISK_EUR / (risk * to_eur)

    return {"side": side, "entry": entry, "sl": sl, "tp": tp, "rr": rr,
            "units": units, "near60": near60, "stoch": float(s), "slope": float(slope)}


# ============================== FORMATO ================================
def fmt_price(x):
    s = f"{x:.5f}" if x < 10 else f"{x:.3f}" if x < 200 else f"{x:.2f}"
    return s.replace(".", ",")


def fmt_units(u):
    if u >= 100:
        return f"{round(u):d}"
    if u >= 1:
        return f"{u:.2f}".replace(".", ",")
    return f"{u:.4f}".replace(".", ",")


def fmt_eur(x):
    return f"{x:.2f}".replace(".", ",") + " €"


def format_signal(name, r):
    comprar = r["side"] == "BUY"
    lines = [
        f"{'🟢 COMPRA' if comprar else '🔴 VENTA'} {name}",
        f"• Orden: Mercado (o Limitada en {fmt_price(r['entry'])})",
        f"• Unidades: {fmt_units(r['units'])}",
        f"• Stop loss: {fmt_price(r['sl'])}",
        f"• Take profit: {fmt_price(r['tp'])}  (ratio 1:{str(r['rr']).replace('.', ',')})",
        f"• Pierdes {fmt_eur(RISK_EUR)} · Ganas {fmt_eur(RISK_EUR * r['rr'])}",
    ]
    if r["near60"]:
        lines.append("• Precio más cerca de la MA60: objetivo ampliado")
    return "\n".join(lines)


# ================================ MAIN =================================
def main():
    force = os.environ.get("FORCE", "0") == "1"
    now_local = datetime.now(TZ)
    if not force and now_local.hour not in HORAS_AVISO:
        print(f"Fuera de horario ({now_local:%H:%M} Madrid). No hago nada.")
        return

    now = pd.Timestamp.now(tz="UTC")

    # Conversión a euros: cuántos euros vale 1 unidad de cada moneda
    to_eur_rates = {"EUR": 1.0}
    for ccy in sorted({a[2] for a in ASSETS} - {"EUR"}):
        d = fetch(f"EUR{ccy}=X", period="5d")
        if d is not None:
            to_eur_rates[ccy] = 1.0 / float(d.Close.iloc[-1])

    signals, failed, closed = [], [], 0
    for name, ticker, ccy in ASSETS:
        if ccy not in to_eur_rates:
            failed.append(name)
            continue
        df = fetch(ticker)
        if df is None:
            failed.append(name)
            continue
        df = only_closed(df, now)
        if len(df) == 0:
            failed.append(name)
            continue
        last_end = df.index[-1] + pd.Timedelta(hours=1)
        if now - last_end >= pd.Timedelta(minutes=60):
            closed += 1          # mercado cerrado: la última vela es vieja
            continue
        try:
            res = analyze(df, to_eur_rates[ccy])
        except Exception as e:
            print(f"Error analizando {name}: {e}")
            failed.append(name)
            continue
        if res:
            res["close_time"] = last_end.tz_convert(TZ)
            signals.append((name, res))

    total = len(ASSETS)
    print(f"Analizados: {total - len(failed) - closed}, cerrados: {closed}, fallos: {len(failed)}, señales: {len(signals)}")

    if signals:
        hora = signals[0][1]["close_time"].strftime("%H:%M")
        msg = f"🔔 Señales 1h · cierre {hora} (Madrid)\n\n"
        msg += "\n\n".join(format_signal(n, r) for n, r in signals)
        send_telegram(msg)

    # Mensaje diario de "sigo funcionando" a las 8:00, o resumen si es una prueba
    if force or now_local.hour == 8:
        txt = (f"✅ Sistema activo · {total - len(failed) - closed} activos analizados, "
               f"{closed} con mercado cerrado, {len(failed)} sin datos.")
        if failed:
            txt += "\nSin datos: " + ", ".join(failed[:20])
        if force:
            txt = "🧪 PRUEBA\n" + txt + f"\nSeñales ahora mismo: {len(signals)}"
        send_telegram(txt)
    elif len(failed) > 0.4 * total:
        send_telegram(f"⚠️ Aviso: {len(failed)} de {total} activos sin datos esta hora. "
                      "Puede ser un fallo temporal de la fuente de datos.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        try:
            send_telegram(f"❌ El escáner ha fallado: {str(e)[:300]}")
        except Exception:
            pass
        sys.exit(1)
