"""
Phase 3: Measure how delays spread through each aircraft's day.

For every linked flight:
  min_turn     - fastest realistic turn at that airport (estimated from data)
  slack        - scheduled turn - min_turn  (cushion that absorbs lateness)
  inherited    - departure delay caused by the previous flight arriving late
  original     - departure delay that started on this flight
Then:
  - % of delay that is inherited (SEA vs. inter-island vs. all)
  - validation against BTS's LATE_AIRCRAFT_DELAY
  - "chain starters": flights whose delay spread downstream, and how much

Input:  data/processed/alaska_rotations.parquet   (from Task 5, MAX_TURN_HOURS = 3)
Output: data/processed/alaska_propagation.parquet
        data/processed/top_chain_starters_sea.csv
        data/processed/starters_by_hour_sea.csv
        data/processed/starters_by_month_sea.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

IN_PATH = Path("data/processed/alaska_rotations.parquet")
OUT_DIR = Path("data/processed")

MIN_TURN_PCTL = 0.05     # min turn = 5th percentile of "rushed" turns
MIN_SAMPLE = 30          # need this many rushed turns at an airport to trust it
LATE = 15                # FAA/BTS definition of a late flight (minutes)
HAWAII = {"HNL", "OGG", "KOA", "LIH", "ITO"}

df = pd.read_parquet(IN_PATH)
print(f"Flights: {len(df):,}")

# --- Step 0: Labels ----------------------------------------------------------
# Hawaiian-heritage planes have tail numbers ending in "HA"
df["fleet"] = np.where(df["TAIL_NUM"].str.endswith("HA"), "Hawaiian", "Alaska")
df["inter_island"] = df["ORIGIN"].isin(HAWAII) & df["DEST"].isin(HAWAII)

# Exclude data errors: plane "left before it arrived"
bad = df["linked"] & (df["actual_turn_min"] < 0)
print(f"Negative actual turns excluded as data errors: {bad.sum():,}")
df.loc[bad, "linked"] = False

linked = df["linked"]

# --- Step 1: Estimate the minimum realistic turn -----------------------------
# Only "rushed" turns tell us how fast a crew CAN turn a plane: those where the
# inbound flight arrived late, so the crew was trying to catch up.
rushed = df[linked & (df["prev_arr_delay"] >= LATE)]

# Per airport + fleet, where there's enough data...
by_airport = (rushed.groupby(["ORIGIN", "fleet"])["actual_turn_min"]
              .agg(min_turn=lambda s: s.quantile(MIN_TURN_PCTL), n="size")
              .reset_index())
by_airport = by_airport[by_airport["n"] >= MIN_SAMPLE]
# ...otherwise fall back to the fleet-wide value
by_fleet = rushed.groupby("fleet")["actual_turn_min"].quantile(MIN_TURN_PCTL)

df = df.merge(by_airport[["ORIGIN", "fleet", "min_turn"]],
              on=["ORIGIN", "fleet"], how="left")
df["min_turn"] = df["min_turn"].fillna(df["fleet"].map(by_fleet))

print("\nEstimated minimum turn (fleet-wide, minutes):")
print(by_fleet.round(0).to_string())
print("Minimum turn at SEA (Alaska fleet):",
      round(by_airport.query("ORIGIN == 'SEA' and fleet == 'Alaska'")["min_turn"].iloc[0])
      if not by_airport.query("ORIGIN == 'SEA' and fleet == 'Alaska'").empty else "n/a")

# --- Step 2: Slack, inherited delay, original delay --------------------------
linked = df["linked"]
dep_delay = df["DEP_DELAY"].clip(lower=0)          # early departures = 0 delay
df["slack"] = (df["sched_turn_min"] - df["min_turn"]).clip(lower=0)

# How much of the inbound lateness the slack could NOT absorb...
spill = (df["prev_arr_delay"] - df["slack"]).clip(lower=0)
# ...but a flight can't inherit more delay than it actually departed with.
df["inherited_delay"] = np.where(linked, np.minimum(spill.fillna(0), dep_delay), 0.0)
df["original_delay"] = dep_delay - df["inherited_delay"]

# --- Step 3: Headline numbers ------------------------------------------------
def inherited_share(mask: pd.Series) -> float:
    sub = df[mask]
    total = sub["inherited_delay"].sum() + sub["original_delay"].sum()
    return sub["inherited_delay"].sum() / total if total else np.nan

everything = pd.Series(True, index=df.index)
sea = df["chain_touches_sea"] & ~df["inter_island"]
print("\n--- Share of departure-delay minutes that were INHERITED ---")
print(f"All flights:          {inherited_share(everything):.1%}")
print(f"SEA chains:           {inherited_share(sea):.1%}")
print(f"Hawaii inter-island:  {inherited_share(df['inter_island']):.1%}")

# --- Step 4: Validate against BTS's own late-aircraft attribution ------------
late = df[df["ARR_DELAY"] >= LATE]
bts_share = late["LATE_AIRCRAFT_DELAY"].sum() / late["ARR_DELAY"].sum()
print("\n--- Validation vs. BTS ---")
print(f"BTS: late-aircraft share of arrival delay (15+ min late flights): {bts_share:.1%}")
print(f"Model: inherited share of departure delay, same flights:          "
      f"{late['inherited_delay'].sum() / late['DEP_DELAY'].clip(lower=0).sum():.1%}")
corr = late[["inherited_delay", "LATE_AIRCRAFT_DELAY"]].corr().iloc[0, 1]
print(f"Flight-level correlation, model vs. BTS: {corr:.2f}  (closer to 1 = better)")

# --- Step 5: Chain starters --------------------------------------------------
# Walk each chain. When a run of legs inherits delay, credit all of it to the
# flight just before the run - the one that "started" it.
df = df.sort_values(["chain_id", "leg_in_chain"]).reset_index(drop=True)
chain = df["chain_id"].to_numpy()
inh = df["inherited_delay"].to_numpy()
downstream = np.zeros(len(df))
legs_hit = np.zeros(len(df), dtype=int)

starter = -1
for i in range(len(df)):
    new_chain = i == 0 or chain[i] != chain[i - 1]
    if new_chain or inh[i] <= 0:
        starter = i                      # this flight could start the next run
        continue
    downstream[starter] += inh[i]        # this leg's inherited delay traces back
    legs_hit[starter] += 1

df["downstream_delay_caused"] = downstream
df["downstream_legs_delayed"] = legs_hit
df["is_chain_starter"] = df["downstream_delay_caused"] >= LATE

starters = df[df["is_chain_starter"] & sea]
print(f"\nChain starters in SEA chains: {len(starters):,} flights "
      f"caused {starters['downstream_delay_caused'].sum():,.0f} downstream delay minutes")
print(f"Average downstream flights delayed per starter: "
      f"{starters['downstream_legs_delayed'].mean():.1f}")

# Which routes start the most downstream delay? (need 20+ starts to count)
df["route"] = df["ORIGIN"] + "-" + df["DEST"]
df["dep_hour"] = df["sched_dep"].dt.hour
df["month"] = df["sched_dep"].dt.to_period("M").astype(str)
sea_df = df[sea]

route_stats = (sea_df.groupby("route")
               .agg(flights=("route", "size"),
                    chain_starts=("is_chain_starter", "sum"),
                    downstream_minutes=("downstream_delay_caused", "sum"),
                    median_slack=("slack", "median"))
               .query("chain_starts >= 20")
               .assign(start_rate=lambda t: t["chain_starts"] / t["flights"])
               .sort_values("downstream_minutes", ascending=False))
print("\nTop 10 routes by downstream delay caused (SEA chains):")
print(route_stats.head(10).round(2).to_string())

hour_stats = (sea_df.groupby("dep_hour")
              .agg(flights=("dep_hour", "size"),
                   chain_starts=("is_chain_starter", "sum"),
                   downstream_minutes=("downstream_delay_caused", "sum")))
month_stats = (sea_df.groupby("month")
               .agg(flights=("month", "size"),
                    chain_starts=("is_chain_starter", "sum"),
                    downstream_minutes=("downstream_delay_caused", "sum"),
                    inherited_minutes=("inherited_delay", "sum")))
print("\nDownstream delay by scheduled departure hour (SEA chains):")
print(hour_stats.to_string())
print("\nBy month (SEA chains):")
print(month_stats.to_string())

# --- Save --------------------------------------------------------------------
df.to_parquet(OUT_DIR / "alaska_propagation.parquet", index=False)
route_stats.to_csv(OUT_DIR / "top_chain_starters_sea.csv")
hour_stats.to_csv(OUT_DIR / "starters_by_hour_sea.csv")
month_stats.to_csv(OUT_DIR / "starters_by_month_sea.csv")
print("\nSaved propagation data + 3 summary CSVs to data/processed/")