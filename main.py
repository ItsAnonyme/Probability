import requests
from bs4 import BeautifulSoup
import pandas as pd


# To get predictions for any upcoming matches, please fill the variables below
HomeTeam = None
AwayTeam = None

Date_storage = {"Date": None}

def get_data():
    data_url = "https://www.football-data.co.uk/new/DNK.csv"
    respose = requests.get(data_url)
    soup = BeautifulSoup(respose.content, "html.parser")
    List = pd.read_csv(soup)
    #print(List)

def get_date():
    date_url = "https://www.football-data.co.uk/denmark.php"
    response = requests.get(date_url)
    soup = BeautifulSoup(response.content, "html.parser")
    Date = soup.find_all('i')
    if 'Last updated' in Date[0].text:
        Last_update = Date[0].text
        if Last_update == Date_storage['Date']:
            print("Date is already up-to-date")
            return Date_storage['Date']
        else:
            Date_storage['Date'] = Last_update
            get_data()
            return Date_storage['Date']



if __name__ == '__main__':
    get_date()
    print(Date_storage['Date'])