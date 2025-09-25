import pandas as pd
from PoissonDistribution import prediction_poisson
from NegativeBinomialDistribution import prediction_nbinomial
from decimal import Decimal
from nbinomTest import prediction_nbinomial2

List = pd.read_csv("Dataset-small.csv", skiprows=range(0, 3205), engine="python")

def compare_prediction_poisson():
    temp_poisson = 0
    for i in range(1, len(List)):
        #print("Input row:", List.iloc[i])
        if prediction_poisson(List.iloc[i, 2], List.iloc[i, 3]) == List.iloc[i, 6]:
            temp_poisson += 1
    print(f"The Poisson Distribution got {round(Decimal((temp_poisson/len(List)) * 100), 10)} % correct")

def compare_prediction_nbinomial():
    temp_nbinom = 0
    for i in range(0, 35):
        if prediction_nbinomial2(List.iloc[i, 5], List.iloc[i, 6]) == List.iloc[i, 9]:
            temp_nbinom += 1
    print(f"The Negative Binomial Distribution got {round(Decimal((temp_nbinom/35) * 100), 10)} % correct")

if __name__ == "__main__":
    compare_prediction_poisson()
    #compare_prediction_nbinomial()