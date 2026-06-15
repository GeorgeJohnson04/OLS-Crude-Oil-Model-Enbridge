"""
scrape_wcsb.py - pull WCSB production data from public sources.

Source: Stats Canada Table 25-10-0063-01 (Supply and disposition of crude
oil and equivalent products by month). Free, no auth, current to two-month lag.

This table gives us EVERY series we need:
  Mined crude bitumen production       -> AER ST39 bitumen
  Synthetic crude oil production       -> AER ST39 SCO
  In-Situ crude bitumen production     -> AER ST53
  Light & medium crude oil + Heavy
    + Condensate + Pentanes plus       -> AER ST3 conventional + condensate
  Crude oil production (Canada total)  -> canadian_production.csv
  Provincial breakdown (AB+SK+BC+MB)   -> wcsb_total.csv

Stats Canada reports monthly TOTAL in cubic metres. We convert to kbbl/day
using each month's actual days-in-month and the m³->bbl factor 6.28981.

For maximum fidelity, the user can still manually drop AER bulletins into
data/ to override what this scraper produces (the AER monthly Alberta data
is the gold standard for ST39/ST53/ST3).

Outputs (overwrites the existing seed files in data/):
  data/canadian_production.csv     Canada total            (kbbl/day)
  data/wcsb_total.csv              AB+SK+BC+MB sum         (kbbl/day)
  data/aer_st39_mining.csv         Mined bitumen + SCO     (AB-only)
  data/aer_st53_insitu.csv         In-situ bitumen         (AB-only)
  data/aer_st3_conventional.csv    Conv crude + condensate (AB-only)

Run:
  python scrape_wcsb.py
"""
from __future__ import annotations

import io
import warnings
import zipfile

import numpy as np
import pandas as pd
import requests

import config

warnings.filterwarnings("ignore")

STATCAN_WDS = "https://www150.statcan.gc.ca/t1/wds/rest/getFullTableDownloadCSV"
HEADERS = {"User-Agent": "Mozilla/5.0 (research; oil-price-model)"}

M3_TO_BBL = 6.28981

PID_NEW = 25100063   # Current table - 2016-01 onward, granular breakdown
PID_OLD = 25100014   # Discontinued table - covers 1985-2016, totals only
WCSB_PROVINCES = ["Alberta", "Saskatchewan", "British Columbia", "Manitoba"]

# Stats Canada category labels (must match the "Supply and disposition" column)
SD = {
    "total_crude":     "Crude oil production",
    "mined_bitumen":   "Mined crude bitumen production",
    "insitu_bitumen":  "In-Situ crude bitumen production",
    "sco":             "Synthetic crude oil production",
    "light_med":       "Light and medium crude oil",
    "heavy":           "Heavy crude oil",
    "condensate":      "Condensate",
    "pentanes":        "Pentanes plus",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fetch_statcan_table(pid: int) -> pd.DataFrame:
    """Pull a full Stats Canada CSV table via the WDS REST API."""
    url = f"{STATCAN_WDS}/{pid}/en"
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    info = r.json()
    if info.get("status") != "SUCCESS":
        raise RuntimeError(f"StatCan WDS failure for {pid}: {info}")
    r2 = requests.get(info["object"], headers=HEADERS, timeout=180)
    r2.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r2.content))
    csvs = [n for n in z.namelist() if n.endswith(".csv") and "MetaData" not in n]
    if not csvs:
        raise RuntimeError(f"No data CSV in zip for {pid}")
    with z.open(csvs[0]) as f:
        df = pd.read_csv(f, encoding="utf-8", low_memory=False)
    return df


def _series_kbpd(df: pd.DataFrame, geo: str, sd_label: str) -> pd.Series:
    """Filter Stats Canada table to (GEO, supply-disposition) and convert
    monthly m³ totals to a kbbl/day Series indexed by month-start."""
    sub = df[(df["GEO"] == geo)
             & (df["Supply and disposition"] == sd_label)
             & (df["UOM"] == "Cubic metres")].copy()
    sub["date"] = pd.to_datetime(sub["REF_DATE"].astype(str) + "-01", errors="coerce")
    sub = sub.dropna(subset=["date"]).set_index("date").sort_index()
    sub["VALUE"] = pd.to_numeric(sub["VALUE"], errors="coerce")
    # Convert: m³ monthly total -> bbl/day average
    days = sub.index.days_in_month
    return (sub["VALUE"] * M3_TO_BBL / days / 1000.0).rename(sd_label)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _stitch(old_series: pd.Series, new_series: pd.Series) -> pd.Series:
    """Splice old (2003-2015) and new (2016+) StatCan series into one continuous
    monthly series. Use the new series wherever it's available, fall back to
    the old series for earlier periods."""
    combined = pd.concat([old_series, new_series])
    # Drop duplicate dates by preferring the new one (it comes second)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return combined


def main():
    print("scrape_wcsb.py — Stats Canada Tables 25-10-0014 (old) + 25-10-0063 (current)\n")
    df_new = _fetch_statcan_table(PID_NEW)
    print(f"  fetched 25-10-0063: {len(df_new):,} rows, "
          f"REF_DATE {df_new['REF_DATE'].min()} -> {df_new['REF_DATE'].max()}")
    df_old = _fetch_statcan_table(PID_OLD)
    print(f"  fetched 25-10-0014: {len(df_old):,} rows, "
          f"REF_DATE {df_old['REF_DATE'].min()} -> {df_old['REF_DATE'].max()}")

    # 1) Canada total - stitch old + new for full 1985-present coverage --------
    total_old = _series_kbpd(df_old, "Canada", "Total crude production")
    total_new = _series_kbpd(df_new, "Canada", SD["total_crude"])
    total = _stitch(total_old, total_new)
    out = total.rename("Production").reset_index().rename(columns={"date": "Date"})
    out.to_csv(config.USER_CANADIAN_PROD_CSV, index=False)
    print(f"\n  + {config.USER_CANADIAN_PROD_CSV.name}  ({len(out)} months, "
          f"{out['Date'].iloc[0].date()} to {out['Date'].iloc[-1].date()})  "
          f"latest={out['Production'].iloc[-1]:.0f} kbpd")

    # 2) WCSB total (AB+SK+BC+MB) - stitch old + new ---------------------------
    def _wcsb_sum(df, label):
        parts = []
        for p in WCSB_PROVINCES:
            s = _series_kbpd(df, p, label)
            if len(s):
                parts.append(s.rename(p))
        if parts:
            return pd.concat(parts, axis=1).sum(axis=1)
        return pd.Series(dtype=float)

    wcsb_old = _wcsb_sum(df_old, "Total crude production")
    wcsb_new = _wcsb_sum(df_new, SD["total_crude"])
    wcsb = _stitch(wcsb_old, wcsb_new).rename("total_kbpd")
    out = wcsb.reset_index().rename(columns={"date": "Date"})
    out.to_csv(config.USER_WCSB_TOTAL_CSV, index=False)
    print(f"  + {config.USER_WCSB_TOTAL_CSV.name}  ({len(out)} months, "
          f"{out['Date'].iloc[0].date()} to {out['Date'].iloc[-1].date()})  "
          f"latest={out['total_kbpd'].iloc[-1]:.0f} kbpd")

    # Bind the working frame to 'df' for the rest of the routine (new table)
    df = df_new

    # 3) AER ST39 - Mined bitumen + SCO (Alberta) ------------------------------
    mining = _series_kbpd(df, "Alberta", SD["mined_bitumen"]).rename("bitumen_kbpd")
    sco    = _series_kbpd(df, "Alberta", SD["sco"]).rename("sco_kbpd")
    st39 = pd.concat([mining, sco], axis=1).reset_index().rename(columns={"date": "Date"})
    st39.to_csv(config.USER_AER_ST39_CSV, index=False)
    print(f"  + {config.USER_AER_ST39_CSV.name}  "
          f"({len(st39)} months)  last={st39['Date'].iloc[-1].date()}  "
          f"mining={st39['bitumen_kbpd'].iloc[-1]:.0f}  sco={st39['sco_kbpd'].iloc[-1]:.0f}")

    # 4) AER ST53 - In-situ bitumen (Alberta) ----------------------------------
    insitu = _series_kbpd(df, "Alberta", SD["insitu_bitumen"]).rename("bitumen_kbpd")
    st53 = insitu.reset_index().rename(columns={"date": "Date"})
    st53.to_csv(config.USER_AER_ST53_CSV, index=False)
    print(f"  + {config.USER_AER_ST53_CSV.name}  "
          f"({len(st53)} months)  last={st53['Date'].iloc[-1].date()}  "
          f"in-situ={st53['bitumen_kbpd'].iloc[-1]:.0f}")

    # 5) AER ST3 - Conventional crude + condensate (Alberta) -------------------
    light_med = _series_kbpd(df, "Alberta", SD["light_med"]).rename("lm")
    heavy     = _series_kbpd(df, "Alberta", SD["heavy"]).rename("hv")
    cond      = _series_kbpd(df, "Alberta", SD["condensate"]).rename("cond")
    pent      = _series_kbpd(df, "Alberta", SD["pentanes"]).rename("pent")
    conv_total = (light_med.add(heavy, fill_value=0)).rename("conventional_kbpd")
    cond_total = (cond.add(pent, fill_value=0)).rename("condensate_kbpd")
    st3 = pd.concat([conv_total, cond_total], axis=1).reset_index().rename(columns={"date": "Date"})
    st3.to_csv(config.USER_AER_ST3_CSV, index=False)
    print(f"  + {config.USER_AER_ST3_CSV.name}  "
          f"({len(st3)} months)  last={st3['Date'].iloc[-1].date()}  "
          f"conv={st3['conventional_kbpd'].iloc[-1]:.0f}  cond={st3['condensate_kbpd'].iloc[-1]:.0f}")

    print("\nDone. Next: python fetch_data.py && python build_features.py && python main.py")


if __name__ == "__main__":
    main()
