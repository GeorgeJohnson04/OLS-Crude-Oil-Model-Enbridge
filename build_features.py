"""
Builds the feature matrix for the four OLS models.

Reads data/panel_raw.csv (produced by fetch_data.py) and writes
data/panel_features.csv. The four models share the same RHS regressor list;
B1 (crude type) and B2 (war length) act as segmenters, not regressors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def _war_state_per_month(index: pd.DatetimeIndex) -> pd.DataFrame:
    """For each month, classify the war regime AS OF that month.

    war_long is time-varying within a single war:
      - First LONG_WAR_MONTHS months of any war  -> war_long = 0 (initial phase)
      - Month LONG_WAR_MONTHS+1 onward            -> war_long = 1 (sustained phase)
      - Wars that end before reaching threshold   -> war_long = 0 throughout
      - Peacetime months                          -> war_long = 0
    """
    rows = []
    for d in index:
        active = None
        for start, end, label, _iran in config.WAR_WINDOWS:
            s = pd.Timestamp(start)
            e = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
            if s <= d <= e:
                months_elapsed = (d - s).days / 30.44
                active = (label, months_elapsed >= config.LONG_WAR_MONTHS)
                break
        if active is None:
            rows.append({"in_war": False, "war_long": False, "war_label": ""})
        else:
            rows.append({"in_war": True, "war_long": active[1], "war_label": active[0]})
    return pd.DataFrame(rows, index=index)


def _hormuz_threat_per_month(index: pd.DatetimeIndex) -> pd.Series:
    flag = pd.Series(0, index=index, name="hormuz_threat")
    for start, end in config.HORMUZ_THREAT_WINDOWS:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
        flag.loc[(flag.index >= s) & (flag.index <= e)] = 1
    return flag


def _structural_break_dummies(index: pd.DatetimeIndex) -> pd.DataFrame:
    """One column per (key) in config.STRUCTURAL_BREAKS - 1 in window, else 0."""
    out = pd.DataFrame(index=index)
    for key, start, end, _desc in config.STRUCTURAL_BREAKS:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
        out[f"regime_{key}"] = ((index >= s) & (index <= e)).astype(int)
    return out


def _opec_event_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Two features from OPEC_EVENTS:
       opec_shock          - the signed delta in mbd in the event month, else 0
       opec_cumulative     - running cumulative sum (memory of policy stance)
    """
    shock = pd.Series(0.0, index=index)
    for date_str, delta, _label in config.OPEC_EVENTS:
        ts = pd.Timestamp(date_str)
        if ts in shock.index:
            shock.loc[ts] += delta
    return pd.DataFrame({
        "opec_shock":      shock,
        "opec_cumulative": shock.cumsum(),
    })


def build_features() -> pd.DataFrame:
    panel = pd.read_csv(config.RAW_PANEL_CSV, index_col="date", parse_dates=True)
    df = panel.copy()

    # Inflation-adjust prices to REAL_PRICE_BASE_YEAR USD.
    # CPI publishes ~2 weeks after month-end, so the most recent month often
    # has a NaN - forward-fill so price-real series stay continuous.
    base_cpi = df.loc[df.index.year == config.REAL_PRICE_BASE_YEAR, "cpi"].mean()
    if pd.isna(base_cpi):
        base_cpi = df["cpi"].dropna().iloc[-12:].mean()  # fallback: latest year average
    cpi_filled = df["cpi"].ffill()
    df["wti_real"]  = df["wti"]  * (base_cpi / cpi_filled)
    df["wcs_real"]  = df["wcs"]  * (base_cpi / cpi_filled)

    # B2 + B4 binaries
    war = _war_state_per_month(df.index)
    df["war_long"]      = war["war_long"].astype(int)        # B2
    df["in_war"]        = war["in_war"].astype(int)
    df["war_label"]     = war["war_label"]
    df["hormuz_threat"] = _hormuz_threat_per_month(df.index) # B4

    # Net exports already computed in fetch_data; rescale to mbbl/day for readability
    df["net_exports"] = df["net_exports"] / 1000.0

    # Logs of strictly-positive series
    df["log_production"] = np.log(df["production_us"])
    df["log_inventory"]  = np.log(df["inventory_us"])
    df["log_dxy"]        = np.log(df["dxy"])
    df["log_wti_real"]   = np.log(df["wti_real"])
    df["log_wcs_real"]   = np.log(df["wcs_real"])

    # Lagged log price (B6) - per crude type
    df["log_wti_lag1"] = df["log_wti_real"].shift(1)
    df["log_wcs_lag1"] = df["log_wcs_real"].shift(1)

    # Crack spread proxy: Brent-WTI spread (inter-crude differential, $/bbl)
    df["crack_spread"] = df["brent"] - df["wti"]

    # Seasonality (annual cycle)
    m = df.index.month
    df["month_sin"] = np.sin(2 * np.pi * m / 12)
    df["month_cos"] = np.cos(2 * np.pi * m / 12)

    # ---- NEW FEATURES ----------------------------------------------------

    # Log returns (dlog price) - the right target variable for a price series
    # that's near-random-walk in levels. Removes the lagged-price-dominates-R^2
    # problem and gives an honest model of period-to-period price changes.
    df["dlog_wti_real"] = df["log_wti_real"].diff()
    df["dlog_wcs_real"] = df["log_wcs_real"].diff()

    # WCS-WTI differential ($/bbl) - the heavy-light spread, central for
    # Enbridge mainline economics. Widening (more negative) = pipeline egress
    # is constrained, heavy producers eat the discount.
    df["wcs_wti_diff"] = df["wcs_real"] - df["wti_real"]
    df["wcs_wti_diff_lag1"] = df["wcs_wti_diff"].shift(1)

    # Pipeline egress cost ($/bbl, real) - time-varying transport from Hardisty
    # to PADD 3 refinery. Falls from $12.50 -> $11.00 when TMX comes online
    # (May 2024). This is what Tidal Energy / Enbridge shippers actually pay.
    if "egress_cost" in df.columns:
        df["egress_cost_real"] = df["egress_cost"] * (base_cpi / cpi_filled)
        df["egress_cost_lag1"] = df["egress_cost_real"].shift(1)

    # Inventory rate-of-change (level differences hide the signal)
    df["d_log_inventory"] = df["log_inventory"].diff()
    if "spr_stocks" in df.columns:
        df["log_spr"]   = np.log(df["spr_stocks"].replace(0, np.nan))
        df["d_log_spr"] = df["log_spr"].diff()
        # SPR drawdowns/refills are policy events
        df["spr_drawdown"] = (-df["d_log_spr"]).clip(lower=0)  # only positive when drawing down

    # OVX (oil VIX) - log so it's symmetric and bounded in scale
    if "ovx" in df.columns:
        df["log_ovx"] = np.log(df["ovx"].replace(0, np.nan))
        df["d_log_ovx"] = df["log_ovx"].diff()

    # COT managed money positioning - already in % of OI; add 1m change
    if "cot_mm_net_pct" in df.columns:
        df["d_cot_mm"] = df["cot_mm_net_pct"].diff()

    # Rig count (if user provided)
    if "rig_count" in df.columns and df["rig_count"].notna().any():
        df["log_rig_count"] = np.log(df["rig_count"].replace(0, np.nan))
        df["d_log_rig_count"] = df["log_rig_count"].diff()

    # OPEC production (if user provided)
    if "opec_production" in df.columns and df["opec_production"].notna().any():
        df["log_opec_production"] = np.log(df["opec_production"].replace(0, np.nan))
        df["d_log_opec_production"] = df["log_opec_production"].diff()

    # Canadian production (if user provided)
    # Most relevant for the Differential model: more Canadian heavy supply
    # with fixed egress => wider WCS-WTI discount.
    if "canadian_production" in df.columns and df["canadian_production"].notna().any():
        df["log_can_production"]   = np.log(df["canadian_production"].replace(0, np.nan))
        df["d_log_can_production"] = df["log_can_production"].diff()

    # ---- WCSB BLENDING MATH --------------------------------------------------
    # If AER ST39 + ST53 reports are loaded, decompose raw bitumen into the
    # pipeline-marketable volume (dilbit + SCO + conventional). Bitumen volumes
    # from AER are RAW - to flow through a pipeline they need ~30% diluent.
    #
    # raw_bitumen = mining_bitumen + insitu_bitumen
    # bitumen_to_upgrader = SCO / SCO_YIELD_FROM_BITUMEN
    # bitumen_for_dilbit = max(0, raw_bitumen - bitumen_to_upgrader)
    # dilbit_volume = bitumen_for_dilbit * DILBIT_VOLUME_FACTOR (~ 1.43x)
    # marketable_wcsb = dilbit_volume + sco + conventional + condensate
    # --------------------------------------------------------------------------
    has_mining = "aer_mining_bitumen_kbpd" in df.columns and df["aer_mining_bitumen_kbpd"].notna().any()
    has_insitu = "aer_insitu_bitumen_kbpd" in df.columns and df["aer_insitu_bitumen_kbpd"].notna().any()
    if has_mining or has_insitu:
        # Treat one source missing as zero only on rows where the OTHER source is present.
        # That way months with no AER data at all stay NaN and don't get a spurious zero.
        mining_bit = df.get("aer_mining_bitumen_kbpd")
        insitu_bit = df.get("aer_insitu_bitumen_kbpd")
        if mining_bit is None: mining_bit = pd.Series(np.nan, index=df.index)
        if insitu_bit is None: insitu_bit = pd.Series(np.nan, index=df.index)
        any_present = mining_bit.notna() | insitu_bit.notna()
        raw_bit = mining_bit.fillna(0) + insitu_bit.fillna(0)
        raw_bit.loc[~any_present] = np.nan
        df["raw_bitumen_kbpd"] = raw_bit

        sco = df.get("aer_mining_sco_kbpd")
        if sco is None: sco = pd.Series(np.nan, index=df.index)
        df["bitumen_to_upgrader_kbpd"] = sco.fillna(0) / config.SCO_YIELD_FROM_BITUMEN
        df["bitumen_for_dilbit_kbpd"]  = (df["raw_bitumen_kbpd"] - df["bitumen_to_upgrader_kbpd"]).clip(lower=0)

        # Apply blending - this is the volume that actually flows in heavy pipelines
        df["dilbit_volume_kbpd"] = df["bitumen_for_dilbit_kbpd"] * config.DILBIT_VOLUME_FACTOR
        df["sco_kbpd"] = sco

        conv = df.get("aer_conventional_kbpd")
        cond = df.get("aer_condensate_kbpd")
        if conv is None: conv = pd.Series(0, index=df.index)
        if cond is None: cond = pd.Series(0, index=df.index)
        # Diluent demand (kbpd): condensate volume needed to make all the dilbit
        df["diluent_demand_kbpd"]  = df["bitumen_for_dilbit_kbpd"] * (1.0 - config.DILBIT_BITUMEN_RATIO) / config.DILBIT_BITUMEN_RATIO
        # Native (AER) condensate is consumed in-region as diluent; only SURPLUS
        # condensate (if any) shows up separately as pipeline volume. Don't double-
        # count: the diluent is already baked into the dilbit_volume.
        df["condensate_surplus_kbpd"] = (cond.fillna(0) - df["diluent_demand_kbpd"]).clip(lower=0)

        # Marketable outbound pipeline volume = dilbit + SCO + conventional + cond surplus
        df["marketable_wcsb_kbpd"] = (df["dilbit_volume_kbpd"] + df["sco_kbpd"].fillna(0)
                                      + conv.fillna(0) + df["condensate_surplus_kbpd"])
        df.loc[df["raw_bitumen_kbpd"].isna(), "marketable_wcsb_kbpd"] = np.nan

        # ---- WCSB egress utilization (production / capacity) -----------------
        # The structural driver of the WCS-WTI basis. When utilization climbs
        # through 85-95% the basis widens; above 95% triggers apportionment.
        # Capacity from config.WCSB_EGRESS_CAPACITIES (step-function over time).
        cap = pd.Series(0.0, index=df.index)
        for _name, kbpd, in_serv, retired in config.WCSB_EGRESS_CAPACITIES:
            in_ts  = pd.Timestamp(in_serv)
            ret_ts = pd.Timestamp(retired) if retired else pd.Timestamp("2099-12-31")
            mask = (df.index >= in_ts) & (df.index < ret_ts)
            cap.loc[mask] += kbpd
        df["wcsb_egress_capacity_kbpd"] = cap
        df["wcsb_utilization"]   = df["marketable_wcsb_kbpd"] / cap
        df["d_wcsb_utilization"] = df["wcsb_utilization"].diff()

        # ---- Use marketable WCSB as the canonical Canadian production input --
        # (overrides the Stats Canada aggregate if AER is present, since the
        # marketable volume is the pipeline-relevant quantity.)
        df["log_can_production"]   = np.log(df["marketable_wcsb_kbpd"].replace(0, np.nan))
        df["d_log_can_production"] = df["log_can_production"].diff()

        # ---- Dilbit and SCO as separate inputs -------------------------------
        # Dilbit volume drives heavy pipeline pressure (Differential model).
        # SCO competes with WTI (Levels/Returns models for light crude).
        df["log_dilbit_volume"]   = np.log(df["dilbit_volume_kbpd"].replace(0, np.nan))
        df["d_log_dilbit_volume"] = df["log_dilbit_volume"].diff()
        df["log_sco"]   = np.log(df["sco_kbpd"].replace(0, np.nan))
        df["d_log_sco"] = df["log_sco"].diff()

    # Apportionment (if user provided) - Enbridge mainline restricted volumes
    # Higher apportionment % = more constrained pipeline = wider WCS discount
    # (i.e. a positive predictor of |WCS-WTI differential|)
    if "apportionment_pct" in df.columns and df["apportionment_pct"].notna().any():
        df["apportionment_lag1"] = df["apportionment_pct"].shift(1)

    # Structural-break dummies
    breaks = _structural_break_dummies(df.index)
    for col in breaks.columns:
        df[col] = breaks[col]

    # OPEC+ event features
    opec = _opec_event_features(df.index)
    df["opec_shock"]      = opec["opec_shock"]
    df["opec_cumulative"] = opec["opec_cumulative"]

    # Save
    df.index.name = "date"
    df.to_csv(config.FEATURES_CSV)

    print(f"Built features panel: {config.FEATURES_CSV.name}  ({len(df)} rows, {df.shape[1]} cols)")
    print(f"  Real price base year: {config.REAL_PRICE_BASE_YEAR}  (CPI base = {base_cpi:.2f})")
    print(f"  Months in war:         {df['in_war'].sum()}")
    print(f"  Months in long war:    {df['war_long'].sum()}")
    print(f"  Months Hormuz threat:  {df['hormuz_threat'].sum()}")
    for key, _, _, _ in config.STRUCTURAL_BREAKS:
        print(f"  Months in {key+':':18s} {int(df[f'regime_{key}'].sum())}")
    return df


if __name__ == "__main__":
    build_features()
