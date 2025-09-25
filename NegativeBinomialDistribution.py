import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import nbinom

# Choose your teams
#HomeTeam = "Aalborg"
#AwayTeam = "Brondby"

# Where we get our data
List = pd.read_csv("https://www.football-data.co.uk/new/DNK.csv", skiprows=range(2764, 2809))

def prediction_nbinomial(HomeTeam, AwayTeam):
    max_goals = 10
    # Rewriting the data
    home_df = pd.DataFrame(data={"team": List.Home, "opponent": List.Away, "goals": List.HG, "home": 1})
    away_df = pd.DataFrame(data={"team": List.Away, "opponent": List.Home, "goals": List.AG, "home": 0})

    # Applying Negative Binomial Distribution
    model_NegBin = smf.glm(data=pd.concat([home_df, away_df]), family=sm.families.NegativeBinomial(alpha=0.1),
                           formula="goals ~ home + team + opponent").fit()

    # Goals prediction
    home_goals_NegBin = model_NegBin.predict(pd.DataFrame(data={"team": HomeTeam, "opponent": AwayTeam, "home": 1}, index=[1])).values[0]
    away_goals_NegBin = model_NegBin.predict(pd.DataFrame(data={"team": AwayTeam, "opponent": HomeTeam, "home": 0}, index=[1])).values[0]

    # The matrix containing the probability of each outcome up to max_goals
    Probability_matrix_NegBin = [
        [nbinom.pmf(k=i, n=max_goals - i, p=(max_goals - i) / (max_goals - i + team_avg)) for i in range(0, max_goals)]
        for team_avg in [home_goals_NegBin, away_goals_NegBin]]

    matrix_NegBin = np.outer(np.array(Probability_matrix_NegBin[0]), np.array(Probability_matrix_NegBin[1]))

    # Summing the probability of each outcome
    Home_Win_NegBin = np.sum(np.tril(matrix_NegBin, -1))
    Draw_NegBin = np.sum(np.diag(matrix_NegBin))
    Away_Win_NegBin = np.sum(np.triu(matrix_NegBin, 1))
    if Home_Win_NegBin > Away_Win_NegBin and Home_Win_NegBin > Draw_NegBin:
        return "H"
    elif Away_Win_NegBin > Draw_NegBin and Away_Win_NegBin > Home_Win_NegBin:
        return "A"
    else:
        return "D"

if __name__ == "__main__":
    print(f"The probability of each team winning the match, according to the Negative Binomial Distribution, is:"
          f"\n {HomeTeam}: {Home_Win_NegBin}\n {AwayTeam}: {Away_Win_NegBin}\n Draw: {Draw_NegBin}"
          f"\nThis is calculated with the maximum number of goals set to {max_goals}"
          f"\nand an expected number of goals for the home team of {home_goals_NegBin}, and for the away team {away_goals_NegBin}\n")