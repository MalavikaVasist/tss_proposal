"""
Select MELCHIORS telluric standard stars observable from La Palma (Jan–Jun).

Selection criteria:
    1. Spectral type O, B, or A only.
    2. At least 6 continuous hours of observability per night
       (airmass ≤ 1.2, alt ≥ 20°, astronomical darkness, Moon > 30°).
    3. Histogram of best-altitude month for surviving stars.

Requirements:
    pip install astropy pandas numpy matplotlib
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.coordinates import (
    EarthLocation,
    SkyCoord,
    AltAz,
    get_body,
)
from astropy.time import Time

# Suppress ERFA warnings about "dubious year" for 2027
import erfa
warnings.filterwarnings("ignore", category=erfa.ErfaWarning)


# ============================================================
# USER SETTINGS
# ============================================================

CATALOGUE_FILE = "melchiors_catalogue_colons.csv"
OUTPUT_FILE = "melchiors_jan_june_visibility.csv"
YEAR = 2027

MIN_DECLINATION = -30.0
MAX_AIRMASS = 1.5
MIN_ALTITUDE = 30.0
MAX_SUN_ALTITUDE = -18.0
TIME_STEP_MINUTES = 15
MIN_CONTINUOUS_HOURS = 6.0
MIN_MOON_SEPARATION_DEG = 60.0



# ============================================================
# LA PALMA OBSERVATORY
# ============================================================

LA_PALMA = EarthLocation(
    lat=28.7561 * u.deg,
    lon=-17.8917 * u.deg,
    height=2396 * u.m,
)


# ============================================================
# LOAD & PREPARE CATALOGUE
# ============================================================

def load_catalogue(filename):
    """Load MELCHIORS catalogue and return unique OBA stars."""

    filename = Path(filename)

    if filename.suffix.lower() == ".csv":
        df = pd.read_csv(filename, header=41, sep="\t", engine="python")
    else:
        df = pd.read_csv(filename, sep=None, engine="python")

    df.columns = [str(c).strip() for c in df.columns]

    print(f"Catalogue columns: {df.columns.tolist()}")
    print(f"Total rows: {len(df):,}")

    return df


def spectral_type_filter(df):
    # Keep only unique stars (by position)
    df = df.drop_duplicates(subset=["ra", "dec"]).copy()
    print(f"Unique stars: {len(df):,}")

    # Filter: spectral type O, B, or A only
    oba_mask = df["SpType"].str.match(r"^[OBA]", na=False)
    df = df[oba_mask].copy()
    print(f"OBA-type stars: {len(df):,}")

    return df

def declination_filter(df):

    # Declination filter
    coords = SkyCoord(
        ra=df["ra"].to_numpy(),
        dec=df["dec"].to_numpy(),
        unit=(u.hourangle, u.deg),
        frame="icrs",
    )

    dec_mask = coords.dec.deg > MIN_DECLINATION
    df = df[dec_mask].copy()
    print(f"After dec > {MIN_DECLINATION}°: {len(df):,}")

    return df 


# ============================================================
# GENERATE NIGHTLY TIME GRIDS
# ============================================================

def get_nightly_times(year):
    """
    Generate per-night time grids for Jan 1 to Jun 30.

    Returns a list of (date_label, times_array) tuples,
    one per night. Each night spans 18:00 UTC to 06:00 UTC next day
    (covers La Palma astronomical darkness comfortably).
    """

    from datetime import date, timedelta

    start_date = date(year, 1, 1)
    end_date = date(year, 6, 30)

    nights = []
    current = start_date

    step = TIME_STEP_MINUTES * u.min

    while current <= end_date:
        # Evening start (18:00 UTC) to next morning (08:00 UTC)
        t_start = Time(f"{current.isoformat()} 18:00:00", scale="utc")
        t_end = t_start + 14 * u.hour  # 14 hours covers full night

        n_points = int((14 * 60) / TIME_STEP_MINUTES)
        times = t_start + np.arange(n_points) * step

        nights.append((current.month, current, times))
        current += timedelta(days=1)

    return nights


# ============================================================
# FIND LONGEST CONTINUOUS OBSERVABLE STRETCH
# ============================================================

def longest_continuous_run(boolean_array):
    """
    Return the length of the longest run of True values.
    """
    if len(boolean_array) == 0:
        return 0

    max_run = 0
    current_run = 0

    for val in boolean_array:
        if val:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0

    return max_run


# ============================================================
# CALCULATE VISIBILITY WITH CONTINUITY CONSTRAINT
# ============================================================

def calculate_visibility(df):
    """
    For each star, find:
      - The night with the longest continuous observability stretch.
      - Best altitude, airmass, moon separation on that night.
      - The month of that best night.

    Also builds a 2D heatmap array:
      heatmap[night_idx, time_idx] = number of stars with >=6h continuous
      observability on that night that are visible at that time point.

    Returns (df_with_results, heatmap, nights_list).
    """

    coords = SkyCoord(
        ra=df["ra"].to_numpy(),
        dec=df["dec"].to_numpy(),
        unit=(u.hourangle, u.deg),
        frame="icrs",
    )

    nights = get_nightly_times(YEAR)
    n_stars = len(df)
    n_nights = len(nights)
    n_times = int((14 * 60) / TIME_STEP_MINUTES)

    print(f"\nCalculating visibility for {n_stars} stars "
          f"over {n_nights} nights...")

    # Results per star: best continuous stretch
    best_continuous_hours = np.zeros(n_stars)
    best_altitude = np.full(n_stars, np.nan)
    best_airmass = np.full(n_stars, np.nan)
    best_time = np.full(n_stars, None, dtype=object)
    best_moon_sep = np.full(n_stars, np.nan)
    best_month = np.zeros(n_stars, dtype=int)

    # Per-night continuous hours for each star: (n_stars, n_nights)
    continuous_hours_grid = np.zeros((n_stars, n_nights))

    # Full visibility mask per night (for heatmap): stored per night
    visibility_per_night = []

    # Process night by night
    for night_idx, (month, night_date, times) in enumerate(nights):

        if night_idx % 30 == 0:
            print(f"  Processing night {night_idx+1}/{n_nights}: "
                  f"{night_date.isoformat()}")

        # Sun altitude
        sun = get_body("sun", times, location=LA_PALMA)
        sun_altaz = sun.transform_to(
            AltAz(obstime=times, location=LA_PALMA)
        )
        dark = sun_altaz.alt.deg <= MAX_SUN_ALTITUDE

        # If no dark time this night, skip
        if not np.any(dark):
            visibility_per_night.append(
                np.zeros((n_stars, n_times), dtype=bool)
            )
            continue

        # Moon position
        moon = get_body("moon", times, location=LA_PALMA)

        # Target positions
        target_altaz = coords[:, None].transform_to(
            AltAz(obstime=times[None, :], location=LA_PALMA)
        )
        altitude = target_altaz.alt.deg
        airmass = target_altaz.secz

        # Moon separation
        moon_sep = coords[:, None].separation(moon.icrs[None, :]).deg

        # Visibility mask: (n_stars, n_times)
        visible = (
            dark[None, :]
            & (altitude >= MIN_ALTITUDE)
            & (airmass <= MAX_AIRMASS)
            & (airmass > 0)
            & np.isfinite(airmass)
            & (moon_sep >= MIN_MOON_SEPARATION_DEG)
        )

        visibility_per_night.append(visible)

        # For each star, find longest continuous run this night
        for i in range(n_stars):
            run_length = longest_continuous_run(visible[i])
            continuous_hours = run_length * TIME_STEP_MINUTES / 60.0 ## longest continuous hours per star through one night
            continuous_hours_grid[i, night_idx] = continuous_hours   ## longest continuous hours per star through one night


            if continuous_hours > best_continuous_hours[i]:
                best_continuous_hours[i] = continuous_hours  ## longest continuous hours for each star across all the nights
                best_month[i] = month

                # Find peak altitude within only the longest visible stretch in a night, not across all nights. 
                vis_idx = np.where(visible[i])[0]
                if len(vis_idx) > 0:
                    peak_idx = vis_idx[np.argmax(altitude[i, vis_idx])]
                    best_altitude[i] = altitude[i, peak_idx]
                    best_airmass[i] = airmass[i, peak_idx]
                    best_time[i] = times[peak_idx].iso
                    best_moon_sep[i] = moon_sep[i, peak_idx]

    # Add results to dataframe
    df = df.copy()
    df["best_continuous_hours"] = best_continuous_hours
    df["best_altitude_deg"] = best_altitude
    df["best_airmass"] = best_airmass
    df["best_observing_time_utc"] = best_time
    df["best_moon_separation_deg"] = best_moon_sep
    df["best_month"] = best_month
    print(f"\nVisibility calculation complete.")

    return df, nights, n_nights, n_times, visibility_per_night, continuous_hours_grid


def compute_heatmap(df, n_nights, n_times, visibility_per_night, continuous_hours_grid):
    
    # Build heatmap: for each night, count stars that have >=6h continuous
    # AND are visible at each time point
    print("\nBuilding heatmap...")
    heatmap = np.zeros((n_nights, n_times), dtype=int)

    # Ensure numpy array for 2D indexing
    continuous_hours_grid = np.asarray(continuous_hours_grid)

    # Stars that qualify (>=6h on at least one night)
    qualified = df['best_continuous_hours'].values >= MIN_CONTINUOUS_HOURS
    qualified_indices = np.where(qualified)[0]

    for night_idx in range(n_nights):
        vis = visibility_per_night[night_idx]  # (n_stars, n_times)
        # Only count stars that have >=6h continuous THIS specific night
        for i in qualified_indices:
            if continuous_hours_grid[i, night_idx] >= MIN_CONTINUOUS_HOURS:
                heatmap[night_idx, :] += vis[i, :]

    print("\nHeatmap built.")

    return heatmap


# ============================================================
# PLOT HISTOGRAM
# ============================================================

def plot_month_histogram(df):
    """
    Histogram of the month in which each star's best continuous
    observability occurs.
    """

    months = df["best_month"].values
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]

    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.arange(0.5, 7.5, 1)
    ax.hist(months, bins=bins, color="steelblue", edgecolor="black", alpha=0.8)

    ax.set_xticks(range(1, 7))
    ax.set_xticklabels(month_names)
    ax.set_xlabel("Month of best continuous observability")
    ax.set_ylabel("Number of OBA stars")
    ax.set_title(
        f"MELCHIORS telluric standards: best observing month\n"
        f"(≥{MIN_CONTINUOUS_HOURS}h continuous, airmass ≤ {MAX_AIRMASS}, "
        f"La Palma {YEAR})"
    )

    # Annotate counts
    counts, _ = np.histogram(months, bins=bins)
    for m, count in enumerate(counts, start=1):
        if count > 0:
            ax.text(m, count + 0.5, str(count), ha="center", fontsize=10)

    plt.tight_layout()
    fname = (f"telluric_standards_month_histogram"
             f"_alt{MIN_ALTITUDE}_am{MAX_AIRMASS}_dec{MIN_DECLINATION}"
             f"_sun{MAX_SUN_ALTITUDE}_moon{MIN_MOON_SEPARATION_DEG}"
             f"_step{TIME_STEP_MINUTES}min"
             f"_cont{MIN_CONTINUOUS_HOURS}h")
    plt.savefig(f"{fname}.pdf")
    print(f"\nHistogram saved: {fname}.pdf")
    plt.show()


# ============================================================
# PLOT HEATMAP (LUMINOSITY PLOT)
# ============================================================

def plot_visibility_heatmap(heatmap, nights):
    """
    2D heatmap: x-axis = time within the night (UTC hours),
    y-axis = night (date), colour = number of OBA stars with ≥6h
    continuous observability that are visible at that moment.
    """

    from matplotlib.dates import DateFormatter, MonthLocator
    from datetime import datetime

    n_nights, n_times = heatmap.shape

    # X-axis: cell edges (n_times + 1 edges for n_times cells)
    hours = 18.0 + np.arange(n_times + 1) * TIME_STEP_MINUTES / 60.0
    # Wrap around midnight for display
    hours_label = hours.copy()
    hours_label[hours_label >= 24] -= 24

    # Y-axis: dates
    dates = [nights[i][1] for i in range(n_nights)]

    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.pcolormesh(
        hours, np.arange(n_nights + 1), heatmap,
        cmap="inferno", shading="flat",
    )

    cbar = fig.colorbar(im, ax=ax, label="Number of visible OBA stars (≥6h continuous)")

    # X-axis formatting
    xtick_vals = np.arange(18, 32.1, 1)
    xtick_labels = [f"{int(h % 24):02d}:00" for h in xtick_vals]
    ax.set_xticks(xtick_vals)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("UTC time")

    # Y-axis formatting: show month labels
    month_starts = []
    month_labels = []
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    for i, d in enumerate(dates):
        if d.day == 1:
            month_starts.append(i)
            month_labels.append(month_names[d.month - 1])

    ax.set_yticks(month_starts)
    ax.set_yticklabels(month_labels)
    ax.set_ylabel("Night (Jan–Jun 2027)")

    ax.set_title(
        f"Telluric standard availability: La Palma {YEAR}\n"
        f"OBA stars with ≥{MIN_CONTINUOUS_HOURS}h continuous observability "
        f"(airmass ≤ {MAX_AIRMASS})"
    )

    plt.tight_layout()
    fname = (f"telluric_visibility_heatmap"
             f"_alt{MIN_ALTITUDE}_am{MAX_AIRMASS}_dec{MIN_DECLINATION}"
             f"_sun{MAX_SUN_ALTITUDE}_moon{MIN_MOON_SEPARATION_DEG}"
             f"_step{TIME_STEP_MINUTES}min"
             f"_cont{MIN_CONTINUOUS_HOURS}h")
    plt.savefig(f"{fname}.pdf")
    print(f"\nHeatmap saved: {fname}.pdf")
    plt.show()


# ============================================================
# PLOT PER-MONTH: STARS PER NIGHT
# ============================================================

def plot_stars_per_night_by_month(df, nights, continuous_hours_grid):
    """
    For each month (Jan to Jun), plot a bar chart showing how many stars
    have ≥ MIN_CONTINUOUS_HOURS continuous observability on each night.

    This helps identify which 2 to 3 nights per month have ≥3 stars
    simultaneously available.
    """

    continuous_hours_grid = np.asarray(continuous_hours_grid)

    # Only consider stars that qualify overall
    qualified_mask = df["best_continuous_hours"].values >= MIN_CONTINUOUS_HOURS
    qualified_idx = np.where(qualified_mask)[0]

    # Get star names for labelling
    star_labels = df["obsID"].values.astype(str)


    n_nights = len(nights)
    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun"}

    for month_num in range(1, 7):
        # Find night indices for this month
        night_indices = [
            i for i, (m, d, t) in enumerate(nights) if m == month_num
        ]
        if not night_indices:
            continue

        # Count qualified stars per night in this month
        stars_per_night = []
        night_dates = []
        star_names_per_night = []

        for ni in night_indices:
            # How many qualified stars have >=6h on this specific night
            count = 0
            names = []
            for si in qualified_idx:
                if continuous_hours_grid[si, ni] >= MIN_CONTINUOUS_HOURS:
                    count += 1
                    names.append(star_labels[si])
            stars_per_night.append(count)
            night_dates.append(nights[ni][1].day)
            star_names_per_night.append(names)

        stars_per_night = np.array(stars_per_night)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 5))

        colors = ["firebrick" if s < 3 else "steelblue" for s in stars_per_night]
        ax.bar(range(len(night_dates)), stars_per_night, color=colors,
               edgecolor="black", alpha=0.8)

        ax.axhline(3, color="green", linestyle="--", linewidth=1.5,
                   label="Target: 3 stars/night")

        ax.set_xticks(range(0, len(night_dates), 2))
        ax.set_xticklabels(
            [str(night_dates[i]) for i in range(0, len(night_dates), 2)],
            fontsize=8,
        )
        ax.set_xlabel(f"Night of {month_names[month_num]} (day of month)")
        ax.set_ylabel("Number of stars with ≥6h continuous")
        ax.set_title(
            f"{month_names[month_num]} {YEAR}: OBA stars available per night\n"
            f"(airmass ≤ {MAX_AIRMASS}, ≥{MIN_CONTINUOUS_HOURS}h continuous)"
        )
        ax.legend()

        # Annotate nights with >=3 stars
        for idx, (count, names) in enumerate(
            zip(stars_per_night, star_names_per_night)
        ):
            if count >= 3:
                ax.text(idx, count + 0.2, str(count),
                        ha="center", fontsize=7, color="darkblue")

        plt.tight_layout()
        fname = (f"stars_per_night_{month_names[month_num]}"
                 f"_alt{MIN_ALTITUDE}_am{MAX_AIRMASS}_dec{MIN_DECLINATION}"
                 f"_sun{MAX_SUN_ALTITUDE}_moon{MIN_MOON_SEPARATION_DEG}"
                 f"_step{TIME_STEP_MINUTES}min"
                 f"_cont{MIN_CONTINUOUS_HOURS}h")
        plt.savefig(f"{fname}.pdf")
        print(f"Saved: {fname}.pdf")
        plt.show()

    # Print best nights (>=3 stars) as a summary table
    print(f"\n{'='*60}")
    print("Nights with ≥3 stars available (good for 2-3 day observing runs):")
    print(f"{'='*60}")
    for month_num in range(1, 7):
        night_indices = [
            i for i, (m, d, t) in enumerate(nights) if m == month_num
        ]
        for ni in night_indices:
            names = []
            for si in qualified_idx:
                if continuous_hours_grid[si, ni] >= MIN_CONTINUOUS_HOURS:
                    names.append(star_labels[si])
            if len(names) >= 3:
                date = nights[ni][1]
                print(f"  {date.isoformat()}  ({month_names[month_num]}): "
                      f"{len(names)} stars — {', '.join(names)}") 


# ============================================================
# LOOKUP HELPERS
# ============================================================

def get_star_idx(df, obs_id):
    """
    Return the positional (iloc) index of a star given its obsID.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe (as returned by calculate_visibility).
    obs_id : int or str
        The obsID value to look up.

    Returns
    -------
    int
        The iloc index of the star in df.

    Raises
    ------
    ValueError
        If the obsID is not found in df.
    """
    obs_id = int(obs_id) if isinstance(obs_id, str) else obs_id
    matches = df.index[df["obsID"] == obs_id]
    if len(matches) == 0:
        raise ValueError(f"obsID {obs_id} not found in dataframe.")
    # Return the positional index (iloc-compatible)
    return df.index.get_loc(matches[0])


def get_best_night_idx(df, nights, obs_id):
    """
    Return the night index (into the `nights` list) corresponding to
    the best observing night for a given obsID.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe (as returned by calculate_visibility).
    nights : list
        The nights list returned by calculate_visibility.
    obs_id : int or str
        The obsID value to look up.

    Returns
    -------
    int
        Index into the `nights` list for the star's best night.

    Raises
    ------
    ValueError
        If the obsID is not found or best night cannot be determined.
    """
    obs_id = int(obs_id) if isinstance(obs_id, str) else obs_id
    matches = df[df["obsID"] == obs_id]
    if len(matches) == 0:
        raise ValueError(f"obsID {obs_id} not found in dataframe.")

    row = matches.iloc[0]
    best_time_str = row.get("best_observing_time_utc", None)

    if best_time_str is None or (isinstance(best_time_str, float) and np.isnan(best_time_str)):
        raise ValueError(f"obsID {obs_id} has no best_observing_time_utc.")

    best_date = Time(best_time_str).datetime.date()

    for ni, (month, ndate, times) in enumerate(nights):
        if ndate == best_date:
            return ni

    raise ValueError(
        f"Best night date {best_date.isoformat()} for obsID {obs_id} "
        f"not found in nights list."
    )


# ============================================================
# PLOT NIGHT TRAJECTORY
# ============================================================

def plot_night_trajectory(df, nights, star_idx, night_idx=None):
    """
    Plot the elevation trajectory of a star through a single night,
    reusing the time grids and observatory setup already computed by
    calculate_visibility / get_nightly_times.

    Produces a plot similar to the Isaac Newton Group visibility plots:
      - Object elevation curve (solid)
      - Moon elevation (dashed)
      - Moon separation annotated at each time step
      - Airmass on the right y-axis
      - Sunset, sunrise, and astronomical twilight markers
      - LST on the top x-axis
      - Moon illumination and phase info

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe returned by calculate_visibility (must contain
        'ra', 'dec', 'obsID', 'SpType', 'Vmag', and best_* columns).
    nights : list
        The nights list returned by calculate_visibility. Each element
        is (month, date, times_array).
    star_idx : int
        Row index (positional, iloc-style) of the star in df to plot.
    night_idx : int, optional
        Index into the `nights` list for which night to plot. If None,
        uses the star's best night (from best_observing_time_utc).
    """
    from datetime import datetime

    # ---- Retrieve star info from the dataframe ----
    row = df.iloc[star_idx]
    ra_str = str(row["ra"])
    dec_str = str(row["dec"])
    obs_id = str(row["obsID"])
    sp_type = str(row.get("SpType", ""))
    vmag = row.get("Vmag", np.nan)

    target = SkyCoord(ra=ra_str, dec=dec_str, unit=(u.hourangle, u.deg),
                      frame="icrs")

    object_name = obs_id

    # ---- Determine which night to plot ----
    if night_idx is None:
        # Find the night matching the star's best observing time
        best_time_str = row.get("best_observing_time_utc", None)
        if best_time_str is not None and best_time_str is not np.nan:
            best_t = Time(best_time_str)
            best_date = best_t.datetime.date()
            # Find matching night in the nights list
            for ni, (month, ndate, times) in enumerate(nights):
                if ndate == best_date:
                    night_idx = ni
                    break
        if night_idx is None:
            # Fallback: use first night
            print("Warning: could not determine best night, using first night.")
            night_idx = 0

    # ---- Extract the time grid for this night ----
    month, night_date, times = nights[night_idx]
    n_times = len(times)

    # ---- Compute target AltAz ----
    altaz_frame = AltAz(obstime=times, location=LA_PALMA)
    target_altaz = target.transform_to(altaz_frame)
    target_alt = target_altaz.alt.deg
    target_airmass = target_altaz.secz

    # ---- Sun position (for sunset/sunrise/twilight) ----
    sun = get_body("sun", times, location=LA_PALMA)
    sun_altaz = sun.transform_to(altaz_frame)
    sun_alt = sun_altaz.alt.deg

    # ---- Moon position and separation ----
    moon = get_body("moon", times, location=LA_PALMA)
    moon_altaz = moon.transform_to(altaz_frame)
    moon_alt = moon_altaz.alt.deg
    moon_sep = target.separation(moon.icrs).deg

    # Moon illumination (elongation method)
    sun_icrs = get_body("sun", times)
    moon_icrs = moon.icrs
    elongation = moon_icrs.separation(sun_icrs).deg
    illumination = (1 - np.cos(np.radians(elongation))) / 2.0 * 100

    mean_illum = np.mean(illumination)
    if mean_illum < 5:
        moon_quarter = "New Moon"
    elif mean_illum < 40:
        moon_quarter = "Crescent"
    elif mean_illum < 60:
        moon_quarter = "Quarter"
    elif mean_illum < 95:
        moon_quarter = "Gibbous"
    else:
        moon_quarter = "Full Moon"

    # ---- Find sunset, sunrise, twilight crossings ----
    def find_crossing(alt_array, threshold, direction="down"):
        for i in range(len(alt_array) - 1):
            if direction == "down":
                if alt_array[i] >= threshold and alt_array[i+1] < threshold:
                    return i
            else:
                if alt_array[i] < threshold and alt_array[i+1] >= threshold:
                    return i
        return None

    sunset_idx = find_crossing(sun_alt, 0, "down")
    sunrise_idx = find_crossing(sun_alt, 0, "up")
    twilight_eve_idx = find_crossing(sun_alt, -18, "down")
    twilight_morn_idx = find_crossing(sun_alt, -18, "up")

    # ---- X-axis: UT hours (matching the time grid starting at 18:00 UTC) ----
    hours_utc = 18.0 + np.arange(n_times) * TIME_STEP_MINUTES / 60.0

    # ---- LST at each time step ----
    lst = times.sidereal_time("mean", longitude=LA_PALMA.lon)
    lst_hours = lst.hour

    # ---- PLOTTING ----
    fig, ax = plt.subplots(figsize=(11, 7))

    # Object trajectory (solid black line)
    ax.plot(hours_utc, target_alt, 'k-', linewidth=2,
            label=f"{object_name}  "
                  f"{target.ra.to_string(u.hour, sep=':', precision=0)}  "
                  f"{target.dec.to_string(u.deg, sep=':', precision=0)}")

    # Moon trajectory (dashed line)
    ax.plot(hours_utc, moon_alt, 'k--', linewidth=1.5, label="Moon (dashed)")

    # Annotate moon separation at regular intervals (~1 hour)
    annotation_step = max(1, int(60 / TIME_STEP_MINUTES))
    for i in range(0, n_times, annotation_step):
        if target_alt[i] > 5:
            ax.annotate(f"{moon_sep[i]:.0f}",
                        xy=(hours_utc[i], target_alt[i]),
                        xytext=(0, -10),
                        textcoords="offset points",
                        fontsize=8, color="blue", ha="center",
                        fontweight="bold")

    # Vertical lines for sunset, sunrise, twilight
    vline_kw = dict(linestyle="--", linewidth=1, alpha=0.8, color="black")
    if sunset_idx is not None:
        ax.axvline(hours_utc[sunset_idx], **vline_kw)
        ax.text(hours_utc[sunset_idx], 91, "S.set", ha="center", fontsize=8)
    if sunrise_idx is not None:
        ax.axvline(hours_utc[sunrise_idx], **vline_kw)
        ax.text(hours_utc[sunrise_idx], 91, "S.rise", ha="center", fontsize=8)
    if twilight_eve_idx is not None:
        ax.axvline(hours_utc[twilight_eve_idx], color="black",
                   linestyle="-", linewidth=1.5, alpha=0.7)
        ax.text(hours_utc[twilight_eve_idx], 91, "Twil", ha="center",
                fontsize=8)
    if twilight_morn_idx is not None:
        ax.axvline(hours_utc[twilight_morn_idx], color="black",
                   linestyle="-", linewidth=1.5, alpha=0.7)
        ax.text(hours_utc[twilight_morn_idx], 91, "Twil", ha="center",
                fontsize=8)

    # Midnight line
    midnight_hour = 24.0
    if hours_utc[0] <= midnight_hour <= hours_utc[-1]:
        ax.axvline(midnight_hour, color="black", linestyle="-",
                   linewidth=1.5, alpha=0.5)

    # Axes formatting
    ax.set_xlim(hours_utc[0], hours_utc[-1])
    ax.set_ylim(0, 90)
    ax.set_xlabel(
        f"Mean Solar Zone Time, starting night "
        f"{night_date.strftime('%m %d %Y')}",
        fontsize=11)
    ax.set_ylabel("Elevation (deg)", fontsize=11)

    # X-axis tick labels (UT hours, wrapping around midnight)
    xtick_vals = np.arange(18, hours_utc[-1] + 0.1, 1)
    xtick_labels = [f"{int(h % 24)}" for h in xtick_vals]
    ax.set_xticks(xtick_vals)
    ax.set_xticklabels(xtick_labels, fontsize=9)
    ax.text(0.0, -0.08, "UT ->", transform=ax.transAxes, fontsize=9,
            ha="left")

    # Secondary x-axis: LST
    ax2_top = ax.twiny()
    ax2_top.set_xlim(ax.get_xlim())
    lst_ticks = []
    lst_tick_labels = []
    for h in xtick_vals:
        idx = int((h - 18.0) * 60 / TIME_STEP_MINUTES)
        if 0 <= idx < n_times:
            lst_ticks.append(h)
            lst_h = lst_hours[idx]
            lst_m = int((lst_h % 1) * 60)
            lst_tick_labels.append(f"{int(lst_h)}h{lst_m:02d}m")
    ax2_top.set_xticks(lst_ticks)
    ax2_top.set_xticklabels(lst_tick_labels, fontsize=7)
    ax2_top.set_xlabel("LST ->", fontsize=9)

    # Right y-axis: airmass scale
    ax2_right = ax.twinx()
    ax2_right.set_ylim(0, 90)
    airmass_elevations = [90, 60, 45, 30, 20]
    airmass_values = [1.0 / np.sin(np.radians(e)) for e in airmass_elevations]
    ax2_right.set_yticks(airmass_elevations)
    ax2_right.set_yticklabels([f"{a:.2f}" for a in airmass_values], fontsize=9)
    ax2_right.set_ylabel("Airmass", fontsize=11, rotation=-90, labelpad=15)

    # Grid
    ax.set_yticks(np.arange(0, 91, 10))
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.grid(True, axis="x", linestyle=":", alpha=0.3)

    # Moon info box (upper left)
    mid_idx = n_times // 2
    moon_ra_str = moon_icrs[mid_idx].ra.to_string(u.hour, sep=":", precision=0)
    moon_dec_str = moon_icrs[mid_idx].dec.to_string(u.deg, sep=":", precision=0)
    moon_info = (
        f"Moon (dashed):\n"
        f"  {moon_ra_str}  {moon_dec_str}\n"
        f"  Illumination: {mean_illum:.0f}%\n"
        f"  {moon_quarter}"
    )
    ax.text(0.01, 0.97, moon_info, transform=ax.transAxes,
            fontsize=8, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.8))

    # Annotation explanation (lower left)
    ax.text(0.01, 0.38,
            "Numbers below curves\nare Moon distance\nin degrees at the\n"
            "corresponding\ntimes.",
            transform=ax.transAxes, fontsize=8, verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.8))

    # Title: site info
    title = (
        f"Observing site coordinates:     "
        f"{LA_PALMA.lon.deg:.4f}E  {LA_PALMA.lat.deg:.4f}N, "
        f"{LA_PALMA.height.value:.0f} m above sea level"
    )
    ax.set_title(title, fontsize=10, fontfamily="monospace", pad=30)

    # Object info (upper right)
    obj_info = f"{object_name}   {target.ra.to_string(u.hour, sep=':', precision=0)}  {target.dec.to_string(u.deg, sep=':', precision=0)}"
    if not np.isnan(vmag):
        obj_info += f"\nV={vmag:.2f}  {sp_type}"
    ax.text(0.99, 0.97, obj_info, transform=ax.transAxes,
            fontsize=9, ha="right", va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.8))

    # Footer with processing info
    now_str = datetime.utcnow().strftime("%Y/%m/%d at %H:%M:%S")
    ax.text(0.5, -0.12,
            f"Processed: {now_str} UT.  La Palma Observatory.",
            transform=ax.transAxes, fontsize=8, ha="center",
            fontfamily="monospace")

    plt.tight_layout()
    safe_name = object_name.replace(" ", "_")
    fname = f"trajectory_{safe_name}_{night_date.isoformat()}.pdf"
    plt.savefig(fname, bbox_inches="tight")
    print(f"\nTrajectory plot saved: {fname}")
    plt.show()

    return fig, ax


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("MELCHIORS Telluric Standard Star Selection")
    print("=" * 60)

    # Load and filter catalogue
    df = load_catalogue(CATALOGUE_FILE)
    df = spectral_type_filter(df)
    df = declination_filter(df)

    # Calculate visibility with continuity constraint
    df, nights, n_nights, n_times, visibility_per_night, continuous_hours_grid = calculate_visibility(df)

    heatmap = compute_heatmap(df, n_nights, n_times, visibility_per_night, continuous_hours_grid)

    # Filter: require >= MIN_CONTINUOUS_HOURS continuous
    print(f"\nFiltering for ≥{MIN_CONTINUOUS_HOURS} continuous hours...")
    df_good = df[df["best_continuous_hours"] >= MIN_CONTINUOUS_HOURS].copy()
    print(f"Stars with ≥{MIN_CONTINUOUS_HOURS}h continuous: {len(df_good)}")

    # Sort by best airmass (lowest = best)
    df_good = df_good.sort_values("best_airmass", ascending=True)

    # Print summary
    columns = []
    for col in [
        "obsID", "Vmag", "SpType", "ra", "dec",
        "best_altitude_deg", "best_airmass",
        "best_continuous_hours", "best_observing_time_utc",
        "best_moon_separation_deg", "best_month",
    ]:
        if col in df_good.columns:
            columns.append(col)

    print(f"\nTop candidates ({min(30, len(df_good))} shown):")
    print(df_good[columns].head(30).to_string(index=False))

    # Save full results
    df_good.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(df_good)} stars to: {OUTPUT_FILE}")

    # Plot histogram
    if len(df_good) > 0:
        plot_month_histogram(df_good)

    # Plot heatmap
    plot_visibility_heatmap(heatmap, nights)

    # Plot per-month stars-per-night
    plot_stars_per_night_by_month(df, nights, continuous_hours_grid)

    print("\nDone.")


if __name__ == "__main__":
    main()
