
import pandas as pd
import re
from pathlib import Path

# ==========================================
# 1. FILE PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = ROOT / "data" / "raw" / "train.csv"
LATER_FILE = ROOT / "data" / "raw" / "later.csv"
HOLDOUT_FILE = ROOT / "data" / "raw" / "holdout_features.csv"

OUTPUT_DIR = ROOT / "data" / "processed"
CLEAN_FILE = OUTPUT_DIR / "train_clean.csv"

FINDINGS_FILE = ROOT / "findings.yaml"


# ==========================================
# 2. LOAD DATA
# ==========================================

def load_data():
    train = pd.read_csv(TRAIN_FILE)
    later = pd.read_csv(LATER_FILE)
    holdout = pd.read_csv(HOLDOUT_FILE)

    print("DATA LOADED SUCCESSFULLY")
    print("Training shape:", train.shape)
    print("Later shape:", later.shape)
    print("Holdout shape:", holdout.shape)

    return train, later, holdout


# ==========================================
# 3. CHECK ALL SEVEN DATA ISSUES
# ==========================================

def check_data(train, later, holdout):

    findings = {}

    # CHECK 1: LEAKING COLUMN
    target = "litres_per_day"

    leaking_columns = (
        set(train.columns) - set(holdout.columns) - {target}
    )

    leaking = len(leaking_columns) > 0

    leaking_column = (
        sorted(leaking_columns)[0] if leaking else None
    )

    findings["leaking_column"] = {
        "present": leaking,
        "column": leaking_column
    }

    print("\n1. LEAKING COLUMN")
    print("Present:", leaking)
    print("Column:", leaking_column)

    # CHECK 2: UNIT MIX-UP
    tower_means = train.groupby("tower")["garden_area_m2"].mean()
    later_means = later.groupby("tower")["garden_area_m2"].mean()

    affected_tower = None
    conversion_factor = None

    for tower in tower_means.index:
        if tower in later_means.index and later_means[tower] != 0:
            ratio = tower_means[tower] / later_means[tower]

            for factor in [10, 12, 60, 100, 1000]:
                if abs(ratio - factor) / factor < 0.20:
                    affected_tower = tower
                    conversion_factor = factor
                    break

        if affected_tower:
            break

    findings["unit_mixup"] = {
        "present": affected_tower is not None,
        "tower": affected_tower,
        "conversion_factor": conversion_factor
    }

    print("\n2. UNIT MIX-UP")
    print("Present:", affected_tower is not None)
    print("Affected tower:", affected_tower)
    print("Conversion factor:", conversion_factor)

    # CHECK 3: SENTINEL VALUE
    sentinel_values = [999, -1, -999, 9999]
    sentinel_value = None
    sentinel_count = 0

    for value in sentinel_values:
        count = (train["supply_hours"] == value).sum()

        if count > sentinel_count:
            sentinel_value = value
            sentinel_count = count

    findings["sentinel_value"] = {
        "present": sentinel_count > 0,
        "value": sentinel_value,
        "count": int(sentinel_count)
    }

    print("\n3. SENTINEL VALUE")
    print("Present:", sentinel_count > 0)
    print("Value:", sentinel_value)
    print("Count:", sentinel_count)

    # CHECK 4: DUPLICATE ROWS
    duplicate_count = int(train.duplicated().sum())

    findings["duplicate_rows"] = {
        "present": duplicate_count > 0,
        "count": duplicate_count
    }

    print("\n4. DUPLICATE ROWS")
    print("Present:", duplicate_count > 0)
    print("Duplicate count:", duplicate_count)

    # CHECK 5: STUCK SENSOR
    sorted_train = train.sort_values("date").copy()

    sensor_column = "tank_level_morning_pct"

    run_ids = (
        sorted_train[sensor_column]
        .ne(sorted_train[sensor_column].shift())
        .cumsum()
    )

    run_lengths = sorted_train.groupby(run_ids).size()

    longest_run_id = run_lengths.idxmax()
    stuck_count = int(run_lengths.max())

    longest_run = sorted_train[run_ids == longest_run_id]

    stuck_detected = stuck_count >= 14

    stuck_indices = longest_run.index.tolist() if stuck_detected else []

    start_date = (
        str(longest_run["date"].iloc[0]) if stuck_detected else None
    )

    end_date = (
        str(longest_run["date"].iloc[-1]) if stuck_detected else None
    )

    findings["stuck_sensor"] = {
        "present": stuck_detected,
        "repeated_rows": stuck_count if stuck_detected else 0,
        "start_date": start_date,
        "end_date": end_date
    }

    print("\n5. STUCK SENSOR")
    print("Present:", stuck_detected)
    print("Repeated rows:", stuck_count if stuck_detected else 0)
    print("Start date:", start_date)
    print("End date:", end_date)

    # CHECK 6: TARGET TYPOS
    target_values = train["litres_per_day"].dropna().sort_values()

    gaps = target_values.diff()

    largest_gap_index = gaps.idxmax()
    lower_value = target_values.loc[target_values.index[
        target_values.index.get_loc(largest_gap_index) - 1
    ]]

    upper_value = target_values.loc[largest_gap_index]

    cutoff = (lower_value + upper_value) / 2

    typo_mask = train["litres_per_day"] > cutoff
    typo_count = int(typo_mask.sum())

    findings["target_typos"] = {
        "present": typo_count > 0,
        "rows_affected": typo_count
    }

    print("\n6. TARGET TYPOS")
    print("Present:", typo_count > 0)
    print("Rows affected:", typo_count)

    # CHECK 7: SPELLING VARIANTS
    raw_spellings = train["supply_source"].nunique()

    true_categories = later["supply_source"].nunique()

    spelling_variants = raw_spellings > true_categories

    findings["spelling_variants"] = {
        "present": spelling_variants,
        "raw_spellings": int(raw_spellings),
        "true_categories": int(true_categories)
    }

    print("\n7. SPELLING VARIANTS")
    print("Present:", spelling_variants)
    print("Raw spellings:", raw_spellings)
    print("True categories:", true_categories)

    findings["_details"] = {
        "stuck_indices": stuck_indices,
        "sentinel_value": sentinel_value,
        "affected_tower": affected_tower,
        "conversion_factor": conversion_factor,
        "target_cutoff": cutoff
    }

    return findings


# ==========================================
# 4. CLEAN TRAINING DATA
# ==========================================

def clean_data(train, findings):

    cleaned = train.copy()
    details = findings["_details"]

    # Remove duplicate rows
    cleaned = cleaned.drop_duplicates()

    # Correct Tower B unit mix-up
    if details["affected_tower"] is not None:
        mask = cleaned["tower"] == details["affected_tower"]

        cleaned.loc[mask, "garden_area_m2"] = (
            cleaned.loc[mask, "garden_area_m2"]
            / details["conversion_factor"]
        )

    # Replace sentinel values with missing values
    if details["sentinel_value"] is not None:
        cleaned["supply_hours"] = cleaned["supply_hours"].replace(
            details["sentinel_value"], pd.NA
        )

    # Replace stuck sensor readings with missing values
    if details["stuck_indices"]:
        cleaned.loc[
            details["stuck_indices"], "tank_level_morning_pct"
        ] = pd.NA

    # Correct target values affected by factor-of-10 typo
    cutoff = details["target_cutoff"]

    typo_mask = cleaned["litres_per_day"] > cutoff

    cleaned.loc[typo_mask, "litres_per_day"] = (
        cleaned.loc[typo_mask, "litres_per_day"] / 10
    )

    # Standardize supply source spellings
    def normalize_source(value):
        value = str(value).strip().lower()
        value = re.sub(r"[^a-z]", "", value)

        aliases = {
            "tan": "tanker",
            "bor": "borewell",
            "mun": "municipal"
        }

        return aliases.get(value, value)

    cleaned["supply_source"] = cleaned["supply_source"].apply(
        normalize_source
    )

    # Save cleaned dataset
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(CLEAN_FILE, index=False)

    print("\nCLEANING COMPLETED")
    print("Original rows:", len(train))
    print("Cleaned rows:", len(cleaned))
    print("Saved file:", CLEAN_FILE)

    print("\nRemaining missing values:")
    print(cleaned.isnull().sum())

    return cleaned


# ==========================================
# 5. UPDATE FINDINGS YAML
# ==========================================

def update_findings(findings):

    if not FINDINGS_FILE.exists():
        print("\nWARNING: findings.yaml not found")
        return

    details = findings["_details"]

    yaml_text = """data_issues:
  leaking_column:
    present: {leaking}
    column: {leaking_column}

  unit_mixup:
    present: {unit_present}
    column: {unit_column}
    group_value: {tower}
    factor: {factor}

  sentinel_value:
    present: {sentinel_present}
    column: {sentinel_column}
    value: {sentinel}
    count: {sentinel_count}

  duplicate_rows:
    present: {duplicate_present}
    count: {duplicate_count}

  stuck_sensor:
    present: {stuck_present}
    column: tank_level_morning_pct
    start_date: "{start_date}"
    end_date: "{end_date}"

  target_typos:
    present: {typo_present}
    count: {typo_count}

  spelling_variants:
    present: {spelling_present}
    column: supply_source
    true_levels: {true_categories}
""".format(
        leaking=str(findings["leaking_column"]["present"]).lower(),
        leaking_column=findings["leaking_column"]["column"] or "null",
        unit_present=str(findings["unit_mixup"]["present"]).lower(),
        unit_column="garden_area_m2" if findings["unit_mixup"]["present"] else "null",
        tower=details["affected_tower"] or "null",
        factor=details["conversion_factor"] or "null",
        sentinel_present=str(findings["sentinel_value"]["present"]).lower(),
        sentinel_column="supply_hours" if findings["sentinel_value"]["present"] else "null",
        sentinel=details["sentinel_value"] if details["sentinel_value"] is not None else "null",
        sentinel_count=findings["sentinel_value"]["count"],
        duplicate_present=str(findings["duplicate_rows"]["present"]).lower(),
        duplicate_count=findings["duplicate_rows"]["count"],
        stuck_present=str(findings["stuck_sensor"]["present"]).lower(),
        start_date=findings["stuck_sensor"]["start_date"] or "null",
        end_date=findings["stuck_sensor"]["end_date"] or "null",
        typo_present=str(findings["target_typos"]["present"]).lower(),
        typo_count=findings["target_typos"]["rows_affected"],
        spelling_present=str(findings["spelling_variants"]["present"]).lower(),
        true_categories=findings["spelling_variants"]["true_categories"]
    )

    existing = FINDINGS_FILE.read_text(encoding="utf-8")

    if "data_issues:" in existing:
        start = existing.index("data_issues:")

        if "\ntracking:" in existing[start:]:
            end = existing.index("\ntracking:", start)
            updated = existing[:start] + yaml_text + "\n" + existing[end:]
        else:
            updated = existing[:start] + yaml_text
    else:
        updated = existing.rstrip() + "\n\n" + yaml_text

    FINDINGS_FILE.write_text(updated, encoding="utf-8")

    print("\nFINDINGS.YAML UPDATED SUCCESSFULLY")


# ==========================================
# 6. MAIN PROGRAM
# ==========================================

def main():

    train, later, holdout = load_data()

    findings = check_data(train, later, holdout)

    clean_data(train, findings)

    update_findings(findings)

    print("\n================================")
    print("ALL DATA CHECKS COMPLETED")
    print("ALL CLEANING STEPS COMPLETED")
    print("FINDINGS YAML UPDATED")
    print("================================")


if __name__ == "__main__":
    main()

