import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson
from decimal import Decimal

# Choose your teams
HomeTeam = "Sunderland"
AwayTeam = "Arsenal"
Time = "Unknown"
max_goals = 10
decay_rate = 0.004

# Where we get our data
Data = pd.read_csv("premier_league_all_seasons_cleaned.csv", skiprows=0, skipfooter=430, engine='python')
List = pd.read_csv("premier_league_all_seasons_cleaned.csv", skiprows=range(0, 11944), skipfooter=50, engine="python")

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
all_matches["goals_last5"] = all_matches.groupby("team")["goals_home"].rolling(5, min_periods=1).mean().shift(1).reset_index(0, drop=True)
all_matches["conceded_last5"] = all_matches.groupby("team")["goals_away"].rolling(5, min_periods=1).mean().shift(1).reset_index(0, drop=True)

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
Data["rest_days_home"].fillna(Data["rest_days_home"].median(), inplace=True)
Data["rest_days_away"].fillna(Data["rest_days_away"].median(), inplace=True)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(all_matches.head(10))
print(all_matches.tail(10))

# Show the first 5 rows
print("FIRST 5 ROWS:")
print(Data.head(50))

# Show the last 5 rows
print("\nLAST 5 ROWS:")
print(Data.tail(5))

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
    "opp_rest_days": Data["rest_days_away"]
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
    "opp_rest_days": Data["rest_days_home"]
})
full_df = pd.concat([home_df, away_df]).reset_index(drop=True)
full_df['Time'] = full_df['Time'].fillna('Unknown')

def get_latest_team_stats(team_name, is_home):
    team_rows = full_df[full_df["team"] == team_name].sort_values("Date")
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
    latest_date = dates.max()
    diff_half_weeks = ((latest_date - dates).dt.days)
    return np.exp(-x * diff_half_weeks)

def prediction_poisson(HomeTeam, AwayTeam, x, Time_of_Match):
    # Apply time-based weights
    full_df['weights'] = calculate_weights(full_df['Date'], x)

    # Applying Poisson Distribution
    model_Poisson = smf.glm(data=full_df, family=sm.families.Poisson(),
                            formula="goals ~ home + team + opponent + Time "
                                    "+ team_goals_last5 + team_conceded_last5 "
                                    "+ opp_goals_last5 + opp_conceded_last5 "
                                    "+ team_last_result + opp_last_result "
                                    "+ h2h_last + team_rest_days + opp_rest_days", freq_weights=full_df['weights']).fit()

    print(model_Poisson.summary())

    home_features = get_latest_team_stats(HomeTeam, True)
    away_features = get_latest_team_stats(AwayTeam, False)

    input_row_home = {"team": HomeTeam, "opponent": AwayTeam, "home": 1, "Time": Time_of_Match, **home_features}
    input_row_away = {"team": AwayTeam, "opponent": HomeTeam, "home": 0, "Time": Time_of_Match, **away_features}

    home_goals = model_Poisson.predict(pd.DataFrame([input_row_home])).values[0]
    away_goals = model_Poisson.predict(pd.DataFrame([input_row_away])).values[0]

    # The matrix containing the probability of each outcome up to max_goals
    Probability_matrix = [[poisson.pmf(i, team_avg) for i in range(0, max_goals)]
                          for team_avg in [home_goals, away_goals]]

    matrix = np.outer(np.array(Probability_matrix[0]), np.array(Probability_matrix[1]))

    # Summing the probability of each outcome
    Home_Win = np.sum(np.tril(matrix, -1))
    Draw = np.sum(np.diag(matrix))
    Away_Win = np.sum(np.triu(matrix, 1))

    if Home_Win == max(Away_Win, Home_Win, Draw):
        return "H"
    elif Away_Win == max(Draw, Away_Win, Home_Win):
        return "A"
    else:
        return "D"

def compare_prediction_poisson_once(decay_rate):
    temp_poisson = 0
    for i in range(0, len(List)):
        if prediction_poisson(List.iloc[i, 1], List.iloc[i, 2], decay_rate, List.iloc[i, 8]) == List.iloc[i, 5]:
            temp_poisson += 1
    print(f"The Poisson Distribution got {round(Decimal((temp_poisson/len(List)) * 100), 10)} % correct")

if __name__ == "__main__":
    print(f"Expected winner: {prediction_poisson(HomeTeam, AwayTeam, decay_rate, Time)}")
    #compare_prediction_poisson(decay_rate)
