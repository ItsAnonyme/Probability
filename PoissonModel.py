import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson
from collections import Counter
import math

# Choose your teams
HomeTeam = "Crystal Palace"
AwayTeam = "Wolves"
time_of_match = "Unknown"
max_goals = 10
decay_rate = 0.004
lower_bound = 1000

# Where we get our data
Data = pd.read_csv("premier_league_all_seasons_cleaned_testfile.csv")
List = pd.read_csv("premier_league_all_seasons_cleaned_testfile.csv", skiprows=11944, skipfooter=50, engine="python")

# Sort by date first
Data["Date"] = pd.to_datetime(Data["Date"], errors="coerce")
Data = Data.sort_values("Date")

# Create a long-format table of all matches (team, date)
home = Data[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].rename(
    columns={"HomeTeam": "team", "AwayTeam": "opponent", "FTHG": "goals_home", "FTAG": "goals_away"}
)
away = Data[["Date", "AwayTeam", "HomeTeam", "FTAG", "FTHG"]].rename(
    columns={"AwayTeam": "team", "HomeTeam": "opponent", "FTAG": "goals_home", "FTHG": "goals_away"}
)
all_matches = pd.concat([home, away]).sort_values(["team", "Date"])

# Compute rolling averages for each team (last 5 matches)
all_matches["goals_last5"] = all_matches.groupby("team")["goals_home"].rolling(5, min_periods=1).mean().shift(
    1).reset_index(0, drop=True)
all_matches["conceded_last5"] = all_matches.groupby("team")["goals_away"].rolling(5, min_periods=1).mean().shift(
    1).reset_index(0, drop=True)

all_matches["last_result"] = np.where(
    all_matches["goals_home"] > all_matches["goals_away"], 1,
    np.where(all_matches["goals_home"] < all_matches["goals_away"], -1, 0)
)
all_matches["last_result_shifted"] = all_matches.groupby("team")["last_result"].shift(1)

Data = Data.merge(
    all_matches[["team", "Date", "goals_last5", "conceded_last5", "last_result_shifted"]],
    left_on=["HomeTeam", "Date"], right_on=["team", "Date"], how="left"
).rename(columns={
    "goals_last5": "team_goals_last5",
    "conceded_last5": "team_conceded_last5",
    "last_result_shifted": "team_last_result"
}).drop(columns=["team"])

Data = Data.merge(
    all_matches[["team", "Date", "goals_last5", "conceded_last5", "last_result_shifted"]],
    left_on=["AwayTeam", "Date"], right_on=["team", "Date"], how="left"
).rename(columns={
    "goals_last5": "opp_goals_last5",
    "conceded_last5": "opp_conceded_last5",
    "last_result_shifted": "opp_last_result"
}).drop(columns=["team"])

# previous result vs same opponent
Data["h2h_last"] = Data.groupby(["HomeTeam", "AwayTeam"])["FTR"].shift(1)
Data["h2h_last"] = Data["h2h_last"].map({'H': 1, 'A': -1, 'D': 0})

# Compute each team's previous match date
all_matches["last_match_date"] = all_matches.groupby("team")["Date"].shift(1)

# Merge last match info back to main Data for both home and away teams
Data = Data.merge(all_matches[["team", "Date", "last_match_date"]],
                  left_on=["HomeTeam", "Date"], right_on=["team", "Date"], how="left")
Data = Data.rename(columns={"last_match_date": "home_last_match"})

Data = Data.merge(all_matches[["team", "Date", "last_match_date"]],
                  left_on=["AwayTeam", "Date"], right_on=["team", "Date"], how="left")
Data = Data.rename(columns={"last_match_date": "away_last_match"})

# Calculate rest days
Data["rest_days_home"] = (Data["Date"] - Data["home_last_match"]).dt.days
Data["rest_days_away"] = (Data["Date"] - Data["away_last_match"]).dt.days

# Replace NaN with median or a default (like average rest days)
Data["rest_days_home"] = Data["rest_days_home"].fillna(Data["rest_days_home"].median())
Data["rest_days_away"] = Data["rest_days_away"].fillna(Data["rest_days_away"].median())

# Rewriting the data
home_df = pd.DataFrame({
    "team": Data["HomeTeam"],
    "opponent": Data["AwayTeam"],
    "goals": Data["FTHG"],
    "home": 1,
    "Date": Data["Date"],
    "Time": Data["Time"],
    "team_goals_last5": Data["team_goals_last5"],
    "team_conceded_last5": Data["team_conceded_last5"],
    "opp_goals_last5": Data["opp_goals_last5"],
    "opp_conceded_last5": Data["opp_conceded_last5"],
    "team_last_result": Data["team_last_result"],
    "opp_last_result": Data["opp_last_result"],
    "h2h_last": Data["h2h_last"],
    "team_rest_days": Data["rest_days_home"],
    "opp_rest_days": Data["rest_days_away"],
    "result": Data["FTR"]
})

away_df = pd.DataFrame({
    "team": Data["AwayTeam"],
    "opponent": Data["HomeTeam"],
    "goals": Data["FTAG"],
    "home": 0,
    "Date": Data["Date"],
    "Time": Data["Time"],
    "team_goals_last5": Data["opp_goals_last5"],
    "team_conceded_last5": Data["opp_conceded_last5"],
    "opp_goals_last5": Data["team_goals_last5"],
    "opp_conceded_last5": Data["team_conceded_last5"],
    "team_last_result": Data["opp_last_result"],
    "opp_last_result": Data["team_last_result"],
    "h2h_last": Data["h2h_last"],
    "team_rest_days": Data["rest_days_away"],
    "opp_rest_days": Data["rest_days_home"],
    "result": Data["FTR"]
})
full_df = pd.concat([home_df, away_df]).reset_index(drop=True)
full_df = full_df.sort_values(["Date"])
full_df['Time'] = full_df['Time'].fillna('Unknown')
full_df['team_goals_last5'] = full_df['team_goals_last5'].fillna(full_df['team_goals_last5'].mean())
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(full_df.loc[11935:11955, :])
if full_df.isna().any().any():
    print("❌ new_data contains NaN → cannot predict")
    print(full_df.isna().sum())

def get_latest_team_stats(match_date, team_name, is_home):
    team_rows = full_df[(full_df["team"] == team_name) & (full_df["Date"] < match_date)].sort_values("Date")
    if team_rows.empty:
        # fallback if new team
        return {
            "team_goals_last5": full_df["team_goals_last5"].mean(),
            "team_conceded_last5": full_df["team_conceded_last5"].mean(),
            "opp_goals_last5": full_df["opp_goals_last5"].mean(),
            "opp_conceded_last5": full_df["opp_conceded_last5"].mean(),
            "team_last_result": 0,
            "opp_last_result": 0,
            "h2h_last": 0,
            "team_rest_days": full_df["team_rest_days"].median(),
            "opp_rest_days": full_df["opp_rest_days"].median()
        }
    else:
        latest = team_rows.iloc[-1]
        return {
            "team_goals_last5": latest["team_goals_last5"],
            "team_conceded_last5": latest["team_conceded_last5"],
            "opp_goals_last5": latest["opp_goals_last5"],
            "opp_conceded_last5": latest["opp_conceded_last5"],
            "team_last_result": latest["team_last_result"],
            "opp_last_result": latest["opp_last_result"],
            "h2h_last": latest["h2h_last"],
            "team_rest_days": latest["team_rest_days"],
            "opp_rest_days": latest["opp_rest_days"]
        }

# Function to calculate weights
def calculate_weights(dates, x):
    dates = pd.to_datetime(dates, errors='coerce')
    latest_date = dates.max()
    diff_half_weeks = ((latest_date - dates).dt.days)
    weights = np.exp(-x * diff_half_weeks)
    return weights

def prediction_poisson(match_date, HomeTeam, AwayTeam, team_goals_last5, team_conceded_last5,
                       opp_goals_last5, opp_conceded_last5, team_last_result, opp_last_result,
                       h2h_last, team_rest_days, opp_rest_days, x, i):
    rows = np.r_[0:i, 12374:2 * i]
    df_train = full_df.iloc[rows, :].copy()
    print("---- DEBUG FOR MATCH ----")
    print(HomeTeam, "vs", AwayTeam, "on", match_date)
    print("df_train size:", df_train.shape)

    if df_train.empty:
        print("❌ df_train is EMPTY → No past matches available")
        return None

    # Print count of NaNs in training set
    print("NaNs per column in df_train:")
    print(df_train.isna().sum())

    # If any feature has huge NaN count, highlight it
    bad_cols = df_train.columns[df_train.isna().sum() > 0].tolist()
    if bad_cols:
        print("⚠ Columns with NaNs:", bad_cols)
    fill_values = {
        "team_goals_last5": 0,
        "team_conceded_last5": 0,
        "opp_goals_last5": 0,
        "opp_conceded_last5": 0,
        "team_last_result": 0,
        "opp_last_result": 0,
        "h2h_last": 0,
        "team_rest_days": 7,
        "opp_rest_days": 7,
    }
    df_train = df_train.fillna(fill_values)

    df_train["weights"] = calculate_weights(df_train['Date'], x)
    model = smf.glm(data=df_train, family=sm.families.Poisson(),
                            formula="goals ~ home + team + opponent"
                                    "+ team_goals_last5 + team_conceded_last5 "
                                    "+ opp_goals_last5 + opp_conceded_last5 "
                                    "+ team_last_result + opp_last_result "
                                    "+ h2h_last + team_rest_days + opp_rest_days",
                            freq_weights=df_train['weights']).fit()
    #print(model.summary())
    home_features = get_latest_team_stats(match_date, HomeTeam, True)
    away_features = get_latest_team_stats(match_date, AwayTeam, False)

    input_row_home = {"team": HomeTeam, "opponent": AwayTeam, "home": 1, **home_features}
    input_row_away = {"team": AwayTeam, "opponent": HomeTeam, "home": 0, **away_features}
    new_data_home = pd.DataFrame([input_row_home])
    new_data_away = pd.DataFrame([input_row_away])

    new_data_home = new_data_home.fillna(fill_values)
    new_data_away = new_data_away.fillna(fill_values)

    #home_goals = model.predict(new_data_home).values[0]
    #away_goals = model.predict(new_data_away).values[0]

    home_goals = (model.predict(pd.DataFrame(data={"team": HomeTeam, "opponent": AwayTeam, "home": 1,
                                                   "team_goals_last5": team_goals_last5, "team_conceded_last5": team_conceded_last5,
                                                   "opp_goals_last5": opp_goals_last5, "opp_conceded_last5": opp_conceded_last5,
                                                   "team_last_result": team_last_result, "opp_last_result": opp_last_result,
                                                   "h2h_last": h2h_last, "team_rest_days": team_rest_days,
                                                   "opp_rest_days": opp_rest_days}, index=[1])).values[0])

    away_goals = (model.predict(pd.DataFrame(data={"team": AwayTeam, "opponent": HomeTeam, "home": 0,
                                                   "team_goals_last5": opp_goals_last5, "team_conceded_last5": opp_conceded_last5,
                                                   "opp_goals_last5": team_goals_last5, "opp_conceded_last5": team_conceded_last5,
                                                   "team_last_result": opp_last_result, "opp_last_result": team_last_result,
                                                   "h2h_last": h2h_last, "team_rest_days": opp_rest_days,
                                                   "opp_rest_days": team_rest_days}, index=[1])).values[0])
    if np.isnan(home_goals) or np.isnan(away_goals):
        print("❌ GLM returned NaN λ → inspect model summary next")
        print(model.summary())
        return None

    probs = np.outer(poisson.pmf(range(max_goals), home_goals),
                     poisson.pmf(range(max_goals), away_goals))

    # Find the location of the maximum value
    max_index = np.argmax(probs)
    i, j = np.unravel_index(max_index, probs.shape)

    # Print coordinates and the value
    print(f"Largest value at {i}-{j} with probability {probs[i, j]:.5f}")

    # Summing the probability of each outcome
    Home_Win = np.sum(np.tril(probs, -1))
    Draw = np.sum(np.diag(probs))
    Away_Win = np.sum(np.triu(probs, 1))

    if Home_Win > Away_Win and Home_Win > Draw:
        return "H", i, j, Home_Win, home_goals, away_goals
    elif Away_Win > Draw and Away_Win > Home_Win:
        return "A", i, j, Away_Win, home_goals, away_goals
    else:
        return "D", i, j, Draw, home_goals, away_goals

def compare_prediction_poisson_once(decay_rate):
        poisson_result_lower, poisson_result_upper, poisson_score_lower, poisson_score_upper, matches_lower, matches_upper = 0, 0, 0, 0, 0, 0
        error1_home, error1_away, error2_home, error2_away, deviance_home, deviance_away = 0, 0, 0, 0, 0, 0
        result, score_result, actual_result = Counter(), Counter(), Counter()
        for i in range(11944, 12324):
            print("Is this the right value of i?", i)
            #if List.iloc[i, 8] in Data.Time:
            #    prediction, home_goals, away_goals, winner, home_goals_predict, away_goals_predict = prediction_poisson(List.iloc[i, 0],
            #        List.iloc[i, 1], List.iloc[i, 2], List.iloc[i, 8], decay_rate)
            #else:
            #    prediction, home_goals, away_goals, winner, home_goals_predict, away_goals_predict = prediction_poisson(
            #        List.iloc[i, 0], List.iloc[i, 1], List.iloc[i, 2], "Unknown", decay_rate)
            prediction, home_goals, away_goals, winner, home_goals_predict, away_goals_predict = prediction_poisson(
                full_df.iloc[i, 4], full_df.iloc[i, 0], full_df.iloc[i, 1], full_df.iloc[i, 6],
                full_df.iloc[i, 7], full_df.iloc[i, 8], full_df.iloc[i, 9], full_df.iloc[i, 10],
                full_df.iloc[i, 11], full_df.iloc[i, 12], full_df.iloc[i, 13], full_df.iloc[i, 14], decay_rate, i)
            result[prediction] += 1
            actual_result[full_df.iloc[i, 5]] += 1
            # print("Actual match result", List.iloc[i, 3], "-", List.iloc[i, 4])
            error1_home += (home_goals_predict - full_df.iloc[i, 2]) ** 2
            error1_away += (home_goals - full_df.iloc[i, 2]) ** 2
            error2_home += (away_goals_predict - full_df.iloc[2*i, 2]) ** 2
            error2_away += (away_goals - full_df.iloc[2*i, 2]) ** 2
            if (Data.HomeTeam == full_df.iloc[i, 0]).sum() + (Data.AwayTeam == full_df.iloc[2*i, 1]).sum() <= lower_bound:
                matches_lower += 1
                if prediction == full_df.iloc[i, 15]:
                    poisson_result_lower += 1
                if home_goals == full_df.iloc[i, 2] and away_goals == full_df.iloc[2*i, 2]:
                    poisson_score_lower += 1
            else:
                matches_upper += 1
                if prediction == full_df.iloc[i, 15]:
                    poisson_result_upper += 1
                if home_goals == full_df.iloc[i, 2] and away_goals == full_df.iloc[2*i, 2]:
                    poisson_score_upper += 1
            if full_df.iloc[i, 2] > 0:
                deviance_home += full_df.iloc[i, 2] * math.log(full_df.iloc[i, 2]/home_goals_predict) - (full_df.iloc[i, 2] - home_goals_predict)
            else:
                deviance_home += - (full_df.iloc[i, 2] - home_goals_predict)
            if full_df.iloc[2*i, 2] > 0:
                deviance_away += full_df.iloc[2*i, 2] * math.log(full_df.iloc[2*i, 2]/away_goals_predict) - (full_df.iloc[2*i, 2] - away_goals_predict)
            else:
                deviance_away += - (full_df.iloc[2*i, 2] - away_goals_predict)
            if home_goals > away_goals:
                score_result['H'] += 1
            elif away_goals > home_goals:
                score_result['A'] += 1
            else:
                score_result['D'] += 1
        print(f"The Poisson Distribution got {((poisson_result_lower + poisson_result_upper)/len(List)) * 100} % of the results correct")
        print(f"The Poisson Distribution got {((poisson_score_lower + poisson_score_upper)/len(List)) * 100} % of the scores correct")
        if matches_lower > 0:
            print(
                f"The Poisson Distribution got {(poisson_result_lower / matches_lower) * 100} % correct in the lower bound\n"
                f"The Poisson Distribution got {(poisson_score_lower / matches_lower) * 100} % of the scores correct in the lower bound")
            print(matches_lower)
        if matches_upper > 0:
            print(
                f"The Poisson Distribution got {(poisson_result_upper / matches_upper) * 100} % correct in the upper bound\n"
                f"The Poisson Distribution got {(poisson_score_upper / matches_upper) * 100} % of the scores correct in the upper bound")
            print(matches_upper)
        print(dict(result))
        print(dict(score_result))
        print(dict(actual_result))
        print(f"Home team goal difference: {math.sqrt(error1_home / len(List))} and {math.sqrt(error2_home / len(List))},\n"
              f"away team goal difference: {math.sqrt(error1_away / len(List))} and {math.sqrt(error2_away / len(List))}")
        print(f"Home team deviance: {2 * deviance_home}\n"
        f"Away team deviance: {2 * deviance_away}")

if __name__ == '__main__':
    compare_prediction_poisson_once(decay_rate)
    #prediction_poisson("2025-11-20", HomeTeam, AwayTeam, decay_rate)