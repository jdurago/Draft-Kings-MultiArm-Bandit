from bs4 import BeautifulSoup
import urllib2
import pandas as pd
import csv

url = "https://rotogrinders.com/lineups/mlb?site=draftkings"

def scrape():
    content = urllib2.urlopen(url).read()
    soup = BeautifulSoup(content, 'html.parser')

    lineup = []
    for player in soup.find_all("a", class_="player-popup"):
        lineup.append(player.text)

    with open('Output/current-projections.csv', 'w') as fp:
        w = csv.writer(fp, delimiter=',')
        for player in lineup:
            w.writerow([player])

    return pd.DataFrame(lineup, columns=['Name'])


if __name__ == '__main__':
    lineup = scrape()
    print lineup