import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson
from decimal import Decimal

#from AvgGoalMethods import relative_homefield_advantage

# Choose your teams
HomeTeam = "Sunderland"
AwayTeam = "Arsenal"
Time = "16:00"
max_goals = 10

# Where we get our data
Data = pd.read_csv("premier_league_all_seasons_cleaned.csv", skiprows=0, skipfooter=242, engine='python')
List = pd.read_csv("premier_league_all_seasons_cleaned.csv", skiprows=range(0, 12132), skipfooter=0, engine="python")

# Rewriting the data
home_df = pd.DataFrame(data={"team": Data.HomeTeam, "opponent": Data.AwayTeam, "goals": Data.FTHG, "home": 1, "Date": pd.to_datetime(Data.Date, errors="coerce"), "Time": Data.Time})
away_df = pd.DataFrame(data={"team": Data.AwayTeam, "opponent": Data.HomeTeam, "goals": Data.FTAG, "home": 0, "Date": pd.to_datetime(Data.Date, errors="coerce"), "Time": Data.Time})
full_df = pd.concat([home_df, away_df]).reset_index(drop=True)

# Function to calculate weights
def calculate_weights(dates, x):
    dates = pd.to_datetime(dates, errors="coerce")
    latest_date = dates.max()
    diff_half_weeks = ((latest_date - dates).dt.days) / 3.5
    return np.exp(-x * diff_half_weeks)

def prediction_poisson(HomeTeam, AwayTeam, x, Time_of_Match):
    #print("Predicting for:", HomeTeam, "vs", AwayTeam)
    # Apply time-based weights
    full_df['weights'] = calculate_weights(full_df['Date'], x)

    # Applying Poisson Distribution
    model_Poisson = smf.glm(data=full_df, family=sm.families.Poisson(),
                            formula="goals ~ home + team + opponent + Time", freq_weights=full_df['weights']).fit()
    #print(model_Poisson.summary())

    # Goals prediction
    home_goals = \
    (model_Poisson.predict(pd.DataFrame(data={"team": HomeTeam, "opponent": AwayTeam, "home": 1, "Time": Time_of_Match}, index=[1])).values[0])
    away_goals = \
    model_Poisson.predict(pd.DataFrame(data={"team": AwayTeam, "opponent": HomeTeam, "home": 0, "Time": Time_of_Match}, index=[1])).values[0]
    #print(f"{HomeTeam} expected goals: {home_goals}")
    #print(f"{AwayTeam} expected goals: {away_goals}")

    # The matrix containing the probability of each outcome up to max_goals
    Probability_matrix = [[poisson.pmf(i, team_avg) for i in range(0, max_goals)]
                          for team_avg in [home_goals, away_goals]]

    matrix = np.outer(np.array(Probability_matrix[0]), np.array(Probability_matrix[1]))

    # Summing the probability of each outcome
    Home_Win = np.sum(np.tril(matrix, -1))
    Draw = np.sum(np.diag(matrix))
    Away_Win = np.sum(np.triu(matrix, 1))
    #print(f"{HomeTeam} win: {Home_Win}\n"
          #f"{AwayTeam} win: {Away_Win}\n"
          #f"Draw: {Draw}\n")

    if Home_Win == max(Away_Win, Home_Win, Draw):
        #print("Home Win!")
        return "H"
    elif Away_Win == max(Draw, Away_Win, Home_Win):
        #print("Away Win!")
        return "A"
    else:
        #print("Draw!")
        return "D"

def compare_prediction_poisson():
    results = {}
    for j in np.arange(0.0055, 0.05, 0.0005):
        temp_poisson = 0
        for i in range(0, len(List)):
            if prediction_poisson(List.iloc[i, 1], List.iloc[i, 2], j, List.iloc[i, 8]) == List.iloc[i, 5]:
                temp_poisson += 1
        results[j] = temp_poisson/len(List)
        print(f"{j}: {temp_poisson/len(List)}")
        print(f"The Poisson Distribution got {round(Decimal((temp_poisson/len(List)) * 100), 10)} % correct")
    print(results)
    print(max(results, key=results.get))

def compare_prediction_poisson_once():
    temp_poisson = 0
    for i in range(0, len(List)):
        if prediction_poisson(List.iloc[i, 1], List.iloc[i, 2], 0.012, List.iloc[i, 8]) == List.iloc[i, 5]:
            temp_poisson += 1
    print(f"The Poisson Distribution got {round(Decimal((temp_poisson/len(List)) * 100), 10)} % correct")

if __name__ == "__main__":
    print(f"Expected winner: {prediction_poisson(HomeTeam, AwayTeam, 0.027, Time)}")
    compare_prediction_poisson_once()
