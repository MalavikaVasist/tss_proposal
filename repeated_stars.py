"""
Identify MELCHIORS stars observed more than once on the same night.

These are stars that already have multi-epoch same-night coverage in the
archive — useful as proven telluric standards since they were tracked
across changing conditions within a single night.

Output:
    repeated_same_night.csv — full observation list for qualifying stars
    repeated_same_night_summary.csv — one row per star with counts
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

CATALOGUE_FILE = "melchiors_catalogue_colons.csv"
OUTPUT_DETAIL = "repeated_same_night.csv"
OUTPUT_SUMMARY = "repeated_same_night_summary.csv"


# ============================================================
# ASSIGN OBSERVING NIGHT
# ============================================================

def assign_observing_night(datetime_series):
    """
    Assign an observing night label to each observation.

    Observations before 12:00 UTC are considered part of the
    previous calendar night (e.g. 2010-09-28T03:00 belongs to
    the night of 2010-09-27).
    """

    dt = pd.to_datetime(datetime_series)
    # Subtract 12 hours so that anything before noon
    # rolls back to the previous date
    shifted = dt - pd.Timedelta(hours=12)
    return shifted.dt.date


# ============================================================
# MAIN
# ============================================================

def main():

    # Load catalogue
    df = pd.read_csv(CATALOGUE_FILE, header=41, sep=r"\t")
    df.columns = [str(c).strip() for c in df.columns]
    print(f"Total observations: {len(df):,}")

    # Assign observing night
    df["obs_night"] = assign_observing_night(df["date-avg"])

    # Identify star by position (ra, dec)
    # Group by (ra, dec, obs_night) and count observations
    grouped = df.groupby(["ra", "dec", "obs_night"]).agg(
        n_obs=("obsID", "count"),
        obs_ids=("obsID", list),
        time_span_hours=(
            "date-avg",
            lambda x: (
                pd.to_datetime(x).max() - pd.to_datetime(x).min()
            ).total_seconds() / 3600.0
        ),
    ).reset_index()

    # Keep only groups with more than one observation on the same night
    repeated = grouped[grouped["n_obs"] > 1].copy()
    print(f"\nStar-night combinations with >1 observation: {len(repeated)}")

    # Get the unique stars that have at least one repeated night
    repeated_stars = repeated[["ra", "dec"]].drop_duplicates()
    print(f"Unique stars observed >1 time on same night: {len(repeated_stars)}")

    # Merge back to get full info for these stars
    df_repeated = df.merge(repeated_stars, on=["ra", "dec"], how="inner")
    print(f"Total observations for these stars: {len(df_repeated)}")

    # Summary: one row per star
    summary = df_repeated.groupby(["ra", "dec"]).agg(
        SpType=("SpType", "first"),
        Vmag=("Vmag", "first"),
        total_observations=("obsID", "count"),
        nights_with_repeats=(
            "obs_night",
            lambda x: (
                x.value_counts() > 1
            ).sum()
        ),
        max_obs_same_night=(
            "obs_night",
            lambda x: x.value_counts().max()
        ),
    ).reset_index()

    summary = summary.sort_values(
        "nights_with_repeats", ascending=False
    )

    # Print summary
    print(f"\n{'='*70}")
    print("Stars with most same-night repeat observations:")
    print(f"{'='*70}")
    print(
        summary.head(30).to_string(index=False)
    )

    # Save
    df_repeated.to_csv(OUTPUT_DETAIL, index=False)
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    print(f"\nSaved detail: {OUTPUT_DETAIL}")
    print(f"Saved summary: {OUTPUT_SUMMARY}")

    # Quick stats
    print(f"\n--- Statistics ---")
    print(f"Total unique repeated stars: {len(summary)}")
    oba = summary[summary["SpType"].str.match(r"^[OBA]", na=False)]
    print(f"Of which OBA type: {len(oba)}")
    print(f"Max observations on one night: {summary['max_obs_same_night'].max()}")
    print(f"Median same-night obs count: {repeated['n_obs'].median():.0f}")


if __name__ == "__main__":
    main()
