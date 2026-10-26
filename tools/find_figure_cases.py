"""Find informative test cases for the attention map figure."""
from pathlib import Path
import pandas as pd

RUNS = Path("./runs")
SEEDS = [0, 1, 2, 3, 42]
FG = [1, 2, 3]
CLASS_NAMES = {1: "LV_endo", 2: "LV_myo", 3: "LA"}


def load_seed_avg(model):
    frames = [pd.read_csv(RUNS / f"{model}_seed{s}" / "test_per_sample.csv")
              for s in SEEDS]
    return pd.concat(frames, ignore_index=True).groupby(
        ["patient", "view", "phase", "class_id"], as_index=False
    )[["dice", "hd95"]].mean()


def per_sample_summary(df):
    fg = df[df["class_id"].isin(FG)]
    keys = ["patient", "view", "phase"]
    overall = fg.groupby(keys)["hd95"].mean().rename("overall_hd95")
    pivot = fg.pivot_table(index=keys, columns="class_id",
                           values="hd95", aggfunc="mean")
    pivot.columns = [f"hd95_{CLASS_NAMES[c]}" for c in pivot.columns]
    return pd.concat([overall, pivot], axis=1).reset_index()


attn = per_sample_summary(load_seed_avg("kan_unet_attn"))
unet = per_sample_summary(load_seed_avg("kan_unet"))
merged = attn.merge(unet, on=["patient", "view", "phase"],
                    suffixes=("_attn", "_unet"))
merged["overall_delta"] = (merged["overall_hd95_attn"]
                            - merged["overall_hd95_unet"])
merged["la_delta"] = merged["hd95_LA_attn"] - merged["hd95_LA_unet"]


def show(title, rows, cols):
    print(f"\n=== {title} ===")
    print(rows[cols].to_string(index=False,
                                float_format=lambda x: f"{x:7.2f}"))


attn_cols = ["patient", "view", "phase", "overall_hd95",
             "hd95_LV_endo", "hd95_LV_myo", "hd95_LA"]
merged_cols = ["patient", "view", "phase",
               "overall_hd95_attn", "hd95_LV_endo_attn",
               "hd95_LV_myo_attn", "hd95_LA_attn",
               "overall_delta", "la_delta"]

show("EASY cases (low overall HD95 — balanced attention demo)",
     attn.nsmallest(5, "overall_hd95"), attn_cols)

show("HARD-LA cases (high LA HD95 with attn — where gates matter)",
     attn.nlargest(8, "hd95_LA"), attn_cols)

show("WHERE-ATTN-HELPS-MOST (overall_delta = attn - unet; large negative = attn much better)",
     merged.nsmallest(8, "overall_delta"), merged_cols)

show("WHERE-ATTN-HELPS-MOST-ON-LA (la_delta large negative)",
     merged.nsmallest(8, "la_delta"), merged_cols)
