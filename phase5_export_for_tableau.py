"""
Phase 5 prep: export clean CSVs for Tableau Public.

Tableau Public can't read .parquet files, so this writes everything the
dashboards need as CSV into a tableau/ folder.

Input:  data/processed/alaska_propagation.parquet  (Phase 3)
        data/processed/buffer_scenarios.csv         (Phase 4)
        data/processed/buffer_sensitivity.csv       (Phase 4)
Output: tableau/flights.csv            one row per flight (overview + chain explorer)
        tableau/buffer_scenarios.csv   one row per scenario (recommendations)
        tableau/buffer_sensitivity.csv
"""

from pathlib import Path
import pandas as pd

DATA = Path("data/processed")
OUT = Path("tableau")
OUT.mkdir(exist_ok=True)

df = pd.read_parquet(DATA / "alaska_propagation.parquet")

# --- Friendly columns for Tableau --------------------------------------------
df["late_departure"] = (df["DEP_DELAY"] >= 15).astype(int)
df["on_time_departure"] = 1 - df["late_departure"]
df["segment"] = "Other"
df.loc[df["chain_touches_sea"] & ~df["inter_island"], "segment"] = "Seattle-linked"
df.loc[df["inter_island"], "segment"] = "Hawaii inter-island"

# One label per aircraft-day, used to pick a rotation in the chain explorer
first_dep = df.groupby("chain_id")["sched_dep"].transform("min")
df["rotation_label"] = df["TAIL_NUM"] + "  |  " + first_dep.dt.strftime("%Y-%m-%d")
df["leg_label"] = (df["leg_in_chain"].astype(str) + ". " + df["ORIGIN"] + "→" + df["DEST"]
                   + "  (" + df["sched_dep"].dt.strftime("%H:%M") + ")")
df["chain_total_inherited"] = df.groupby("chain_id")["inherited_delay"].transform("sum")

cols = {
    "FL_DATE": "flight_date", "month": "month", "dep_hour": "dep_hour",
    "TAIL_NUM": "tail_number", "OP_CARRIER_FL_NUM": "flight_number",
    "ORIGIN": "origin", "DEST": "dest", "route": "route",
    "fleet": "fleet", "segment": "segment",
    "chain_id": "chain_id", "leg_in_chain": "leg_in_chain",
    "rotation_label": "rotation_label", "leg_label": "leg_label",
    "chain_length": "chain_length", "chain_total_inherited": "chain_total_inherited",
    "sched_dep": "sched_dep", "sched_arr": "sched_arr",
    "DEP_DELAY": "dep_delay_min", "ARR_DELAY": "arr_delay_min",
    "sched_turn_min": "sched_turn_min", "slack": "slack_min",
    "inherited_delay": "inherited_delay_min", "original_delay": "original_delay_min",
    "downstream_delay_caused": "downstream_delay_caused_min",
    "is_chain_starter": "is_chain_starter",
    "late_departure": "late_departure", "on_time_departure": "on_time_departure",
}
out = df[list(cols)].rename(columns=cols)
out["is_chain_starter"] = out["is_chain_starter"].astype(int)
out.to_csv(OUT / "flights.csv", index=False)
print(f"tableau/flights.csv: {len(out):,} rows, {out.shape[1]} columns")

# --- Scenario tables ----------------------------------------------------------
scen = pd.read_csv(DATA / "buffer_scenarios.csv")
scen["scenario_type"] = "Targeted route"
scen.loc[scen["scenario"].str.contains("Blanket"), "scenario_type"] = "Across the board"
scen.loc[scen["scenario"].str.contains("morning|Thin-slack|combined", case=False),
         "scenario_type"] = "Broad policy"
scen["is_recommended"] = (scen["scenario"].isin(["Turn after SEA-SFO", "Turn after PDX-SFO"])
                          & (scen["buffer_min"] == 5)).astype(int)
scen.to_csv(OUT / "buffer_scenarios.csv", index=False)
print(f"tableau/buffer_scenarios.csv: {len(scen)} rows")

pd.read_csv(DATA / "buffer_sensitivity.csv").to_csv(OUT / "buffer_sensitivity.csv", index=False)
print("tableau/buffer_sensitivity.csv copied")