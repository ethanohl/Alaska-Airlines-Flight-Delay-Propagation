"""
Phase 4: Test buffer scenarios - where would extra turn time help most?

Idea: replay every affected aircraft day with some turns made longer, and
measure how much downstream delay disappears.

For each flight we keep what's fixed and what can change:
  FIXED  original_delay - delay that starts on this flight (Phase 3)
  FIXED  enroute        - minutes gained/lost in the air (arr delay - dep delay)
  FIXED  recovery       - extra catch-up the crew achieved beyond the slack
  CHANGES inherited     - recomputed from the (new) previous arrival delay
                          and the (new) slack after adding buffer

Replaying with 0 buffer reproduces the real data exactly (checked below).

Input:  data/processed/alaska_propagation.parquet   (from Phase 3)
        data/processed/top_chain_starters_sea.csv    (from Phase 3)
Output: data/processed/buffer_scenarios.csv
        data/processed/buffer_sensitivity.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

DATA = Path("data/processed")
BUFFERS = [5, 10, 15]          # minutes of extra turn time to test
TOP_ROUTES = 6                 # how many top chain-starter routes to test
LATE = 15

# Value of one minute of delay avoided, in dollars. PLACEHOLDER - look up the
# latest Airlines for America (A4A) "cost of delays" figure, update this
# number, and cite it in your memo.
COST_PER_DELAY_MIN = 100

df = pd.read_parquet(DATA / "alaska_propagation.parquet")
df = df.sort_values(["chain_id", "leg_in_chain"]).reset_index(drop=True)
sea_chain = df["chain_touches_sea"] & ~df["inter_island"]

# Previous flight's route (only meaningful when linked)
df["prev_route"] = df.groupby("chain_id")["route"].shift(1).where(df["linked"])


# =========================== Core model =====================================
def decompose(data: pd.DataFrame, min_turn: pd.Series) -> pd.DataFrame:
    """Split each flight's delay into fixed and changeable parts."""
    out = pd.DataFrame(index=data.index)
    out["chain"] = data["chain_id"].to_numpy()
    out["linked"] = data["linked"].to_numpy()
    out["dep"] = data["DEP_DELAY"].clip(lower=0)
    out["slack_raw"] = data["sched_turn_min"] - min_turn      # can be negative
    slack = out["slack_raw"].clip(lower=0)
    spill = data["prev_arr_delay"] - slack
    out["inherited"] = np.where(out["linked"],
                                np.minimum(spill.clip(lower=0).fillna(0), out["dep"]), 0.0)
    out["original"] = out["dep"] - out["inherited"]
    out["recovery"] = np.where(out["linked"], (spill - out["dep"]).clip(lower=0).fillna(0), 0.0)
    out["enroute"] = data["ARR_DELAY"] - out["dep"]
    return out


def replay(parts: pd.DataFrame, buffer: np.ndarray) -> np.ndarray:
    """Re-run each aircraft day, adding `buffer` minutes to selected turns.
    Returns the new departure delay of every flight."""
    chain = parts["chain"].to_numpy()
    linked = parts["linked"].to_numpy()
    dep0 = parts["dep"].to_numpy()
    slack_raw = parts["slack_raw"].to_numpy()
    original = parts["original"].to_numpy()
    recovery = parts["recovery"].to_numpy()
    enroute = parts["enroute"].to_numpy()

    new_dep = np.empty(len(parts))
    prev_arr = 0.0
    for i in range(len(parts)):
        if i == 0 or chain[i] != chain[i - 1] or not linked[i]:
            d = dep0[i]                               # nothing to inherit
        else:
            slack = max(0.0, slack_raw[i] + buffer[i])
            d = max(0.0, prev_arr - slack - recovery[i]) + original[i]
        new_dep[i] = d
        prev_arr = d + enroute[i]
    return new_dep


def run_scenario(parts, name, target_mask, minutes):
    """Add `minutes` to every targeted turn; only replay chains that change."""
    target = target_mask & parts["linked"]
    chains = parts.loc[target, "chain"].unique()
    sub = parts[parts["chain"].isin(chains)]
    buffer = np.where(target.loc[sub.index], minutes, 0.0)

    new_dep = replay(sub, buffer)
    base_dep = sub["dep"].to_numpy()
    saved = base_dep.sum() - new_dep.sum()
    added = target.sum() * minutes
    return {
        "scenario": name,
        "buffer_min": minutes,
        "turns_buffered": int(target.sum()),
        "buffer_minutes_added": int(added),
        "delay_minutes_saved": round(saved),
        "saved_per_buffer_min": round(saved / added, 2) if added else np.nan,
        "late_departures_avoided": int((base_dep >= LATE).sum() - (new_dep >= LATE).sum()),
        "est_value_usd": round(saved * COST_PER_DELAY_MIN),
    }


# =========================== Baseline + check ===============================
parts = decompose(df, df["min_turn"])
check = replay(parts, np.zeros(len(parts)))
max_err = np.abs(check - parts["dep"].to_numpy()).max()
print(f"Replay check - 0 buffer reproduces real delays (max error {max_err:.4f} min, expect 0)")

# =========================== Scenarios ======================================
top_routes = pd.read_csv(DATA / "top_chain_starters_sea.csv")["route"].head(TOP_ROUTES).tolist()
print(f"Testing turns right after these chain-starter routes: {top_routes}\n")

sea_origin = (df["ORIGIN"] == "SEA") & (df["fleet"] == "Alaska")
morning = df["dep_hour"].between(7, 10)

scenarios = {}
for r in top_routes:
    scenarios[f"Turn after {r}"] = sea_chain & (df["prev_route"] == r)
scenarios["All top routes combined"] = sea_chain & df["prev_route"].isin(top_routes)
scenarios["SEA morning departures 7-10am"] = sea_origin & morning
scenarios["Thin-slack SEA-chain turns (<10 min)"] = sea_chain & (parts["slack_raw"] < 10)
scenarios["Blanket: every SEA-chain turn"] = sea_chain.copy()

results = []
for name, mask in scenarios.items():
    for b in BUFFERS:
        results.append(run_scenario(parts, name, mask, b))
        print(f"  done: {name} +{b}")

res = pd.DataFrame(results).sort_values("saved_per_buffer_min", ascending=False)
print("\n=== Scenarios ranked by delay minutes saved per buffer minute added ===")
print(res.to_string(index=False))
res.to_csv(DATA / "buffer_scenarios.csv", index=False)

# =========================== Sensitivity check ==============================
# Phase 3 estimated SEA's minimum turn at 52 min (likely inflated by aircraft
# swaps). Re-run headline numbers with SEA = 40 min to see if findings hold.
SEA_MIN_TURN_ALT = 40
alt_min_turn = df["min_turn"].where(~sea_origin, SEA_MIN_TURN_ALT)
parts_alt = decompose(df, alt_min_turn)

def inherited_share(p, mask):
    p = p[mask]
    return p["inherited"].sum() / p["dep"].sum()

sens = pd.DataFrame([
    {"assumption": "SEA min turn = Phase 3 estimate",
     "sea_inherited_share": inherited_share(parts, sea_chain)},
    {"assumption": f"SEA min turn = {SEA_MIN_TURN_ALT} min",
     "sea_inherited_share": inherited_share(parts_alt, sea_chain)},
])
key = ["All top routes combined", "SEA morning departures 7-10am",
       "Blanket: every SEA-chain turn"]
for label, p in [("SEA min turn = Phase 3 estimate", parts),
                 (f"SEA min turn = {SEA_MIN_TURN_ALT} min", parts_alt)]:
    for name in key:
        r = run_scenario(p, name, scenarios[name], 10)
        sens.loc[sens["assumption"] == label, f"{name} (+10) saved/min"] = r["saved_per_buffer_min"]

sens["sea_inherited_share"] = (sens["sea_inherited_share"] * 100).round(1).astype(str) + "%"
print("\n=== Sensitivity: does the SEA minimum-turn assumption change the story? ===")
print(sens.to_string(index=False))
sens.to_csv(DATA / "buffer_sensitivity.csv", index=False)

print("\nSaved -> data/processed/buffer_scenarios.csv, buffer_sensitivity.csv")