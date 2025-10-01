import pandas as pd
import requests
from io import StringIO
import csv
from datetime import datetime

BASE_URL = "https://www.football-data.co.uk/mmz4281/{}/E0.csv"

def read_csv_fix(text):
    f = StringIO(text)
    reader = csv.reader(f, quotechar='"', delimiter=',')
    rows = list(reader)
    if not rows:
        return pd.DataFrame()
    header = rows[0]
    data = rows[1:]
    return pd.DataFrame(data, columns=header)

def read_csv_flexible(text):
    f = StringIO(text)
    reader = csv.reader(f, quotechar='"', delimiter=',')
    rows = list(reader)
    if not rows:
        return pd.DataFrame()
    max_cols = max(len(r) for r in rows)
    header = rows[0] + [f"Extra_{i}" for i in range(len(rows[0]), max_cols)]
    data = [r + [None]*(max_cols-len(r)) for r in rows[1:]]
    return pd.DataFrame(data, columns=header)

def make_unique_columns(df):
    cols = pd.Series(df.columns)
    for dup in df.columns[df.columns.duplicated()]:
        dup_indices = [i for i, x in enumerate(df.columns) if x == dup]
        for i, idx in enumerate(dup_indices):
            cols[idx] = f"{dup}_{i}"
    df.columns = cols
    return df

# Build season codes
start_year = 1993
current_year = datetime.now().year + 1
seasons = [f"{str(y)[-2:]}{str(y+1)[-2:]}" for y in range(start_year, current_year)]
print("Seasons to fetch:", seasons[:5], "...", seasons[-5:])

all_data = []
all_columns = set()

# Fetch and parse all seasons
for season in seasons:
    url = BASE_URL.format(season)
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        text = response.text.strip()
        if text:
            if season in ["0304", "0405"]:
                df = read_csv_flexible(text)
            else:
                df = read_csv_fix(text)
            if not df.empty:
                df = make_unique_columns(df)
                all_data.append(df)
                all_columns.update(df.columns)
                print(f"✅ {season}: {len(df)} rows, {len(df.columns)} cols")
            else:
                print(f"⚠️ {season}: empty DataFrame")
        else:
            print(f"⚠️ {season}: empty response")
    except Exception as e:
        print(f"❌ {season} failed: {e}")

# Standardize columns efficiently
all_columns = list(all_columns)
for i, df in enumerate(all_data):
    missing_cols = set(all_columns) - set(df.columns)
    if missing_cols:
        df = pd.concat(
            [df, pd.DataFrame({col: pd.NA for col in missing_cols}, index=df.index)],
            axis=1
        )
    all_data[i] = df[all_columns]

# Merge all DataFrames
merged_df = pd.concat(all_data, ignore_index=True, sort=False)

# Remove completely empty rows
merged_df = merged_df.dropna(how="all")

# Fix Attendance column
if "Attendance" in merged_df.columns:
    merged_df["Attendance"] = merged_df["Attendance"].replace("", 0)
    merged_df["Attendance"] = merged_df["Attendance"].fillna(0)
    merged_df["Attendance"] = merged_df["Attendance"].astype(int)

# Convert numeric columns safely
for col in merged_df.columns:
    merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce").combine_first(merged_df[col])

# Convert Date column to datetime
date_cols = [col for col in merged_df.columns if col.lower() == "date"]
for col in date_cols:
    merged_df[col] = pd.to_datetime(
        merged_df[col],
        errors="coerce",
        dayfirst=True
    )

# Remove unwanted columns
columns_to_remove = [
    "BWCD","B365>2.5","Extra_63","_18","HTHG","WHCH","_6","WHCD","B365CAHH","PCAHA",
    "BSH","LBD","HS","IWA","Bb1X2","PSA","Extra_61","BFCD","MaxAHA","BbAvH","BFA","SJH",
    "VCCD","_10","_8","BbAHh","SBH","Extra_64","Extra_70","BSA","MaxD","PAHH","LBAH",
    "MaxC>2.5","B365AHH","AvgC<2.5","BFECAHH","_0","MaxCAHH","GBAHA","VCH","1XBD","HC",
    "Extra_67","B365H","AY","1XBCA","BbMx<2.5","AvgCA","HO","VCD","BFE<2.5","AC","SYH",
    "VCA","VCCA","BFEAHA","AvgC>2.5","SOA","BbMxAHA","HBP","_5","PC<2.5","Max>2.5","HHW",
    "SJA","_16","PSD","AvgH","PSCD","_2","MaxCA","BFEC>2.5","IWCD","WHD","B365A","AR",
    "_11","BbAv<2.5","Extra_58","AvgAHH","SBD","SYD","MaxH","IWCA","LBAHA","_3","HTAG",
    "MaxCH","GBD","BbAvAHA","MaxCAHA","Extra_59","PSH","SYA","B365AH","BFECA","BbAvAHH",
    "B365CH","B365AHA","AvgCH","1XBH","Extra_60","ABP","BbAv>2.5","WHH","_9","BbOU",
    "B365CD","IWCH","VCCH","SOH","BWCH","_17","P<2.5","BFCH","Extra_69","BbMxAHH","GBH",
    "GBA","P>2.5","Extra_57","B365C<2.5","HY","SBA","BFCA","Extra_71","Div","GB>2.5",
    "AHCh","BbAvA","BWH","Avg>2.5","BFEA","BWCA","AST","_12","Extra_68","AO",
    "GB<2.5","1XBCD","_7","AS","_19","BbMxH","BFECH","SJD","B365CA","BFECD","B365D",
    "WHCA","HTR","Extra_65","_20","BFEC<2.5","GBAH","PC>2.5","Extra_66","BWD","BSD",
    "1XBA","BFEAHH","BbMxA","BFECAHA","HF","PCAHH","Avg<2.5","B365C>2.5","BFD","LBH",
    "AvgCD","MaxAHH","BbMxD","_1","AvgAHA","1XBCH","BWA","AF","PSCA","Max<2.5","ï»¿Div",
    "WHA","MaxC<2.5","B365CAHA","LBAHH","_14","IWH","BFEH","MaxA","Extra_62","IWD","AHh",
    "AHW","BFH","BFED","_13","B365<2.5","AvgCAHA","PAHA","BFE>2.5","_4","BbAvD","PSCH",
    "AvgD","HR","_15","BbAH","MaxCD","SOD","BbMx>2.5","AvgCAHH","GBAHH","LBA","HST","AvgA",
    "BFDA","CLA","BVD","LBCH","CLCA","BMGMD","BVCH","LBCA","CLH","BMGMCA","BFDD","BMGMA",
    "BVA","LBCD","BVCA","BFDCA","BVH","BMGMCD","BFDH","BFDCD","BMGMCH","BFDCH","BMGMH",
    "CLCD","BVCD","CLCH","CLD"
]

merged_df = merged_df.drop(columns=[col for col in columns_to_remove if col in merged_df.columns], errors='ignore')

# Reorder columns as requested
desired_order = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "Referee", "Attendance", "Time"]
existing_cols = [col for col in desired_order if col in merged_df.columns]
other_cols = [col for col in merged_df.columns if col not in existing_cols]
merged_df = merged_df[existing_cols + other_cols]

# Sort by Date
if "Date" in merged_df.columns:
    merged_df = merged_df.sort_values("Date").reset_index(drop=True)

# Save cleaned CSV
merged_df.to_csv("premier_league_all_seasons_cleaned_testfile.csv", index=False)
print(f"\n🎉 Done! Cleaned, ordered, and sorted data saved with {len(merged_df)} rows")
