"""
Pulls every input series from FRED and EIA and assembles a single monthly
panel saved to data/panel_raw.csv.

FRED: WTI, Brent, CPI, DXY (free CSV, no API key)
EIA:  Production, inventory, refinery utilization, exports, imports, SPR
      (free XLS hist_xls bulk files, no API key)
GPR:  Caldara-Iacoviello Geopolitical Risk Index (XLS)
CBOE: OVX (oil volatility index, daily CSV)
CFTC: Disaggregated COT - managed money positioning in WTI futures
"""
from __future__ import annotations

import io
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import requests

import config

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
EIA_XLS  = "https://www.eia.gov/dnav/pet/hist_xls/{sid}m.xls"
OVX_CSV  = "https://cdn.cboe.com/api/global/us_indices/daily_prices/OVX_History.csv"
COT_ZIP  = "https://www.cftc.gov/sites/default/files/files/dea/history/fut_disagg_xls_{year}.zip"
HEADERS  = {"User-Agent": "Mozilla/5.0 (oil-ols-model)"}

# EIA series we need (replaces broken FRED IDs)
EIA_SERIES = {
    "production_us":  "MCRFPUS2",   # U.S. Field Production of Crude Oil, kbbl/day
    "inventory_us":   "MCESTUS1",   # U.S. Ending Stocks of Crude Oil, kbbl
    "refinery_util":  "MOPUEUS2",   # Refinery Operable Utilization, %
    "exports_us":     "MCREXUS2",   # Crude Oil Exports, kbbl/day
    "imports_us":     "MCRIMUS2",   # Crude Oil Imports, kbbl/day
    "spr_stocks":     "MCSSTUS1",   # SPR Ending Stocks, kbbl (monthly)
}


def _download_fred(series_id: str) -> pd.Series:
    url = FRED_CSV.format(sid=series_id)
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    date_col, val_col = df.columns[0], df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col])
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    return df.set_index(date_col)[val_col].rename(series_id)


def _download_eia(series_id: str) -> pd.Series:
    """Pull EIA monthly hist_xls file and return clean monthly series."""
    url = EIA_XLS.format(sid=series_id)
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    df = pd.read_excel(io.BytesIO(r.content), sheet_name="Data 1", skiprows=2, header=None)
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date"])
    # EIA dates are mid-month (e.g. 1920-01-15); snap to month-start
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    return df.set_index("date")["value"].rename(series_id)


def _to_monthly(s: pd.Series, how: str = "mean") -> pd.Series:
    if how == "mean":
        return s.resample("MS").mean()
    if how == "last":
        return s.resample("MS").last()
    raise ValueError(how)


def _load_user_wcs() -> pd.Series | None:
    if not config.USER_WCS_CSV.exists():
        return None
    df = pd.read_csv(config.USER_WCS_CSV)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns or "price" not in df.columns:
        print(f"  ! {config.USER_WCS_CSV.name} found but needs Date,Price columns - ignoring")
        return None
    df["date"] = pd.to_datetime(df["date"])
    return _to_monthly(df.set_index("date")["price"].rename("wcs_user"), "mean")


def _load_user_canadian_production() -> pd.Series | None:
    """User-supplied Canadian crude production (kbbl/day, monthly).
    Source: Statistics Canada Table 25-10-0014-01 or CER monthly export reports.
    CSV format: columns 'Date' and 'Production' (kbbl/day average for the month)."""
    if not config.USER_CANADIAN_PROD_CSV.exists():
        return None
    df = pd.read_csv(config.USER_CANADIAN_PROD_CSV)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns or "production" not in df.columns:
        print(f"  ! {config.USER_CANADIAN_PROD_CSV.name} found but needs Date,Production columns - ignoring")
        return None
    df["date"] = pd.to_datetime(df["date"])
    return _to_monthly(df.set_index("date")["production"].rename("canadian_production"), "mean")


def _load_aer_csv(path, required_cols: list[str], prefix: str) -> pd.DataFrame | None:
    """Generic loader for AER report CSVs. Returns monthly DataFrame with
    columns named '{prefix}_{col}_kbpd' or None if file missing/malformed."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns:
        print(f"  ! {path.name}: missing 'date' column — ignoring")
        return None
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"  ! {path.name}: missing columns {missing} — ignoring")
        return None
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    out = pd.DataFrame(index=pd.date_range(df.index.min(), df.index.max(), freq="MS"))
    for col in required_cols:
        s = df[col].resample("MS").mean()
        # Avoid double-suffixing: if col already ends with _kbpd, keep as-is.
        out_col = f"{prefix}_{col}" if col.endswith("_kbpd") else f"{prefix}_{col}_kbpd"
        out[out_col] = s
    return out


def _load_aer_st39() -> pd.DataFrame | None:
    """AER ST39 - Alberta Mineable Oil Sands Plant Statistics.
    Expected cols: Date, bitumen_kbpd, sco_kbpd."""
    return _load_aer_csv(config.USER_AER_ST39_CSV,
                         ["bitumen_kbpd", "sco_kbpd"], "aer_mining")


def _load_aer_st53() -> pd.DataFrame | None:
    """AER ST53 - Alberta Crude Bitumen In Situ Production.
    Expected cols: Date, bitumen_kbpd."""
    return _load_aer_csv(config.USER_AER_ST53_CSV,
                         ["bitumen_kbpd"], "aer_insitu")


def _load_aer_st3() -> pd.DataFrame | None:
    """AER ST3 - Alberta Energy Outlook (conventional + condensate).
    Expected cols: Date, conventional_kbpd, condensate_kbpd."""
    return _load_aer_csv(config.USER_AER_ST3_CSV,
                         ["conventional_kbpd", "condensate_kbpd"], "aer")


def _load_wcsb_total() -> pd.Series | None:
    """Total WCSB production (raw, kbbl/day) from S&P or WoodMac.
    Expected cols: Date, total_kbpd."""
    if not config.USER_WCSB_TOTAL_CSV.exists():
        return None
    df = pd.read_csv(config.USER_WCSB_TOTAL_CSV)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns or "total_kbpd" not in df.columns:
        print(f"  ! {config.USER_WCSB_TOTAL_CSV.name}: need Date,total_kbpd — ignoring")
        return None
    df["date"] = pd.to_datetime(df["date"])
    return _to_monthly(df.set_index("date")["total_kbpd"].rename("wcsb_total_raw"), "mean")


def _fetch_ovx() -> pd.Series | None:
    """CBOE Oil VIX - daily, resampled to monthly mean."""
    try:
        r = requests.get(OVX_CSV, headers=HEADERS, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = [c.strip() for c in df.columns]
        df["DATE"] = pd.to_datetime(df["DATE"])
        s = df.set_index("DATE")["OVX"].astype(float).rename("ovx")
        return s.resample("MS").mean()
    except Exception as e:
        print(f"  ! OVX fetch failed: {e}")
        return None


def _fetch_cot_managed_money(years: list[int]) -> pd.Series | None:
    """CFTC Disaggregated COT - managed money net WTI futures positioning.

    Returns positions as PERCENT of open interest (scale-invariant).

    NYMEX renamed the contract around 2022:
      - 2010-2021: 'CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE'
      - 2022+:     'WTI FINANCIAL CRUDE OIL - NEW YORK MERCANTILE EXCHANGE'
    We try both each year and take whichever has more rows.
    """
    market_aliases = [
        "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE",
        "WTI FINANCIAL CRUDE OIL - NEW YORK MERCANTILE EXCHANGE",
    ]
    pieces = []
    for year in years:
        url = COT_ZIP.format(year=year)
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            if not r.ok:
                continue
            zf = zipfile.ZipFile(io.BytesIO(r.content))
            df = pd.read_excel(zf.open(zf.namelist()[0]))
            best = None
            for m in market_aliases:
                sub = df[df["Market_and_Exchange_Names"] == m]
                if best is None or len(sub) > len(best):
                    best = sub
            if best is None or best.empty:
                continue
            sub = best.copy()
            sub["date"] = pd.to_datetime(sub["Report_Date_as_MM_DD_YYYY"])
            sub["mm_net_pct_oi"] = (
                sub["Pct_of_OI_M_Money_Long_All"]
                - sub["Pct_of_OI_M_Money_Short_All"]
            )
            pieces.append(sub[["date", "mm_net_pct_oi"]])
        except Exception as e:
            print(f"  ! COT {year} failed: {e}")
    if not pieces:
        return None
    out = (
        pd.concat(pieces)
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .set_index("date")["mm_net_pct_oi"]
    )
    return out.resample("MS").mean().rename("cot_mm_net_pct_oi")


def _load_user_csv(path, value_col: str) -> pd.Series | None:
    """Generic user-CSV loader for fallback data sources (rig count, OPEC,
    apportionment). Expects two columns: Date + a value column. Renames the
    value to the supplied name and snaps to month-start."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns:
        print(f"  ! {path.name}: missing 'date' column")
        return None
    val = next((c for c in df.columns if c != "date"), None)
    if val is None:
        return None
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
    return df.set_index("date")[val].astype(float).rename(value_col)


def _fetch_gpr() -> pd.Series | None:
    """Geopolitical Risk Index. Try both XLS and known CSV mirrors."""
    candidates = [
        ("https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls", "xls"),
        ("https://www.policyuncertainty.com/media/Geopolitical_Risk_Data.xlsx", "xlsx"),
    ]
    for url, kind in candidates:
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            df = pd.read_excel(io.BytesIO(r.content))
            df.columns = [str(c).strip().lower() for c in df.columns]
            date_col = next((c for c in df.columns if c in ("month", "date", "obs")), df.columns[0])
            gpr_col  = next((c for c in df.columns if c == "gpr"), None)
            if gpr_col is None:
                continue
            df[date_col] = pd.to_datetime(df[date_col])
            s = df.set_index(date_col)[gpr_col].astype(float).rename("gpr")
            return s.resample("MS").mean()
        except Exception as e:
            print(f"  GPR mirror {url} failed: {e}")
    return None


def fetch_panel() -> pd.DataFrame:
    print("Fetching FRED series...")
    raw = {}
    for key, sid in config.FRED_SERIES.items():
        # Skip the FRED IDs we now know are broken; we'll get those from EIA
        if key in ("production_us", "inventory_us", "refinery_util", "net_imports", "us_exports"):
            continue
        try:
            s = _download_fred(sid)
            print(f"  {key:14s} {sid:14s} {len(s):5d} obs  {s.index.min().date()} -> {s.index.max().date()}")
            raw[key] = s
        except Exception as e:
            print(f"  ! FRED {key} ({sid}) failed: {e}")

    print("\nFetching EIA series...")
    for key, sid in EIA_SERIES.items():
        try:
            s = _download_eia(sid)
            print(f"  {key:14s} {sid:14s} {len(s):5d} obs  {s.index.min().date()} -> {s.index.max().date()}")
            raw[key] = s
        except Exception as e:
            print(f"  ! EIA {key} ({sid}) failed: {e}")

    panel = pd.DataFrame(index=pd.date_range(config.START_DATE, datetime.today(), freq="MS"))

    for k in ("wti", "brent", "cpi"):
        if k in raw:
            panel[k] = raw[k].reindex(panel.index)

    for k in ("production_us", "inventory_us", "refinery_util", "spr_stocks"):
        if k in raw:
            panel[k] = raw[k].reindex(panel.index)

    if "exports_us" in raw and "imports_us" in raw:
        panel["net_exports"] = (raw["exports_us"] - raw["imports_us"]).reindex(panel.index)

    if "dxy_broad" in raw:
        panel["dxy"] = _to_monthly(raw["dxy_broad"], "mean").reindex(panel.index)

    # Heavy crude price: prefer user CSV, then EIA Imported RAC, then flat proxy.
    # EIA Imported RAC is DELIVERED-to-US-refinery: it includes ~$11-12 of pipeline
    # transport from Hardisty/Edmonton to PADD 3. We back out a time-varying
    # transport cost so the resulting series approximates wellhead-equivalent WCS
    # at Hardisty -- which is the price Tidal Energy / Enbridge shippers see and
    # the level at which Heavy must trade BELOW WTI (lower API, higher sulfur).
    wcs_user = _load_user_wcs()
    if wcs_user is not None:
        panel["wcs"] = wcs_user.reindex(panel.index)
        panel.attrs["wcs_source"] = "user_csv"
        print(f"\n  wcs            user_csv       {wcs_user.notna().sum()} obs from {config.USER_WCS_CSV.name}")
    else:
        try:
            heavy_delivered = _download_eia(config.EIA_IMPORTED_RAC_ID).reindex(panel.index)
            # Time-varying egress adjustment: higher pre-TMX (apportionment era),
            # lower post-TMX (more pipeline capacity).
            tmx_date = pd.Timestamp(config.TMX_INSERVICE_DATE)
            egress = pd.Series(config.HEAVY_NETBACK_TRANSPORT, index=panel.index, name="egress_cost")
            egress.loc[egress.index < tmx_date] = config.HEAVY_NETBACK_TRANSPORT_PRE_TMX
            panel["egress_cost"] = egress
            panel["wcs"] = heavy_delivered - egress
            panel.attrs["wcs_source"] = "eia_imported_rac_netback"
            n_obs = int(panel["wcs"].notna().sum())
            pre_tmx_cost = float(egress.loc[egress.index < tmx_date].iloc[-1]) if (egress.index < tmx_date).any() else 0
            post_tmx_cost = float(egress.loc[egress.index >= tmx_date].iloc[0]) if (egress.index >= tmx_date).any() else 0
            print(f"\n  wcs            EIA RAC -      {n_obs} obs  (Imported RAC minus egress: ${pre_tmx_cost:.2f} pre-TMX, ${post_tmx_cost:.2f} post-TMX)")
            print(f"                 netback        netback to wellhead Hardisty/Edmonton; "
                  f"matches what Enbridge shippers actually realize")
            print(f"                                drop wcs_prices.csv in data/ to override with true WCS")
        except Exception as e:
            print(f"\n  ! EIA RAC fetch failed ({e}) - falling back to WTI - ${config.HEAVY_DIFFERENTIAL_FALLBACK:.0f}")
            panel["wcs"] = panel["wti"] - config.HEAVY_DIFFERENTIAL_FALLBACK
            panel.attrs["wcs_source"] = f"wti_minus_{config.HEAVY_DIFFERENTIAL_FALLBACK:.0f}"

    print("\nFetching GPR index...")
    gpr = _fetch_gpr()
    if gpr is not None:
        panel["gpr"] = gpr.reindex(panel.index)
        print(f"  gpr            iacoviello/pu  {gpr.notna().sum()} obs")
    else:
        panel["gpr"] = 100.0
        print(f"  ! GPR fetch failed - using neutral baseline 100")

    print("\nFetching CBOE OVX (oil volatility)...")
    ovx = _fetch_ovx()
    if ovx is not None:
        panel["ovx"] = ovx.reindex(panel.index)
        print(f"  ovx            cboe           {panel['ovx'].notna().sum()} obs (starts ~2009)")
    else:
        panel["ovx"] = np.nan
        print(f"  ! OVX fetch failed - column will be NaN")

    print("\nFetching CFTC COT (managed money WTI positioning)...")
    cot_years = list(range(2010, datetime.today().year + 1))
    cot = _fetch_cot_managed_money(cot_years)
    if cot is not None:
        panel["cot_mm_net_pct"] = cot.reindex(panel.index)
        print(f"  cot_mm_net_pct cftc           {panel['cot_mm_net_pct'].notna().sum()} obs (managed money net % of OI)")
    else:
        panel["cot_mm_net_pct"] = np.nan
        print(f"  ! COT fetch failed - column will be NaN")

    print("\nLoading optional user-CSV fallbacks (data/ directory)...")
    rig = _load_user_csv(config.USER_RIGCOUNT_CSV, "rig_count")
    if rig is not None:
        panel["rig_count"] = rig.reindex(panel.index)
        print(f"  rig_count      user_csv       {panel['rig_count'].notna().sum()} obs")
    else:
        panel["rig_count"] = np.nan
        print(f"  - rig_count: no {config.USER_RIGCOUNT_CSV.name} (drop in to enable)")

    opec = _load_user_csv(config.USER_OPEC_PROD_CSV, "opec_production")
    if opec is not None:
        panel["opec_production"] = opec.reindex(panel.index)
        print(f"  opec_production user_csv      {panel['opec_production'].notna().sum()} obs")
    else:
        panel["opec_production"] = np.nan
        print(f"  - opec_production: no {config.USER_OPEC_PROD_CSV.name} (using event dummies instead)")

    appo = _load_user_csv(config.USER_APPORTIONMENT_CSV, "apportionment_pct")
    if appo is not None:
        panel["apportionment_pct"] = appo.reindex(panel.index)
        print(f"  apportionment  user_csv       {panel['apportionment_pct'].notna().sum()} obs")
    else:
        panel["apportionment_pct"] = np.nan
        print(f"  - apportionment_pct: no {config.USER_APPORTIONMENT_CSV.name} (used in differential model if present)")

    can_prod = _load_user_canadian_production()
    if can_prod is not None:
        panel["canadian_production"] = can_prod.reindex(panel.index)
        print(f"  canadian_prod  user_csv       {panel['canadian_production'].notna().sum()} obs (kbbl/day)")
    else:
        panel["canadian_production"] = np.nan
        print(f"  - canadian_production: no {config.USER_CANADIAN_PROD_CSV.name} (Stats Canada 25-10-0014-01 — drop in to enable)")

    # AER report breakdown (WCSB / Alberta detail)
    st39 = _load_aer_st39()
    if st39 is not None:
        for col in st39.columns:
            panel[col] = st39[col].reindex(panel.index)
        print(f"  aer ST39 mining   user_csv    {st39.notna().any(axis=1).sum()} obs (bitumen + SCO)")
    else:
        print(f"  - aer ST39 mining: no {config.USER_AER_ST39_CSV.name}")

    st53 = _load_aer_st53()
    if st53 is not None:
        for col in st53.columns:
            panel[col] = st53[col].reindex(panel.index)
        print(f"  aer ST53 in-situ  user_csv    {st53.notna().any(axis=1).sum()} obs")
    else:
        print(f"  - aer ST53 in-situ: no {config.USER_AER_ST53_CSV.name}")

    st3 = _load_aer_st3()
    if st3 is not None:
        for col in st3.columns:
            panel[col] = st3[col].reindex(panel.index)
        print(f"  aer ST3 conv      user_csv    {st3.notna().any(axis=1).sum()} obs")
    else:
        print(f"  - aer ST3 conventional: no {config.USER_AER_ST3_CSV.name}")

    wcsb_total = _load_wcsb_total()
    if wcsb_total is not None:
        panel["wcsb_total_raw"] = wcsb_total.reindex(panel.index)
        print(f"  wcsb_total       user_csv     {panel['wcsb_total_raw'].notna().sum()} obs (S&P or WoodMac)")
    else:
        print(f"  - wcsb_total: no {config.USER_WCSB_TOTAL_CSV.name} (S&P or WoodMac feed)")

    panel.index.name = "date"
    panel.to_csv(config.RAW_PANEL_CSV)
    print(f"\nSaved raw panel: {config.RAW_PANEL_CSV.name}  ({len(panel)} rows, {panel.shape[1]} cols)")
    return panel


if __name__ == "__main__":
    fetch_panel()
