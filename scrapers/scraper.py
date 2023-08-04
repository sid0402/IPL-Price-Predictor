import pandas as pd
from bs4 import BeautifulSoup 
import requests

df = pd.read_csv('ipl_2023_dataset.csv')
df['Nationality']=''
df['Image']=''

teams = ['mumbai-indians','chennai-super-kings','kolkata-knight-riders','punjab-kings','rajasthan-royals',
        'gujarat-titans', 'royal-challengers-bangalore','lucknow-super-giants','delhi-capitals','sunrisers-hyderabad']

def player_link():
    for team in teams:
        req = requests.get("https://www.iplt20.com/teams/"+team)
        soup = BeautifulSoup(req.content, 'html.parser')


        batsmen = soup.find_all('div', class_="ih-pcard-sec")[0].find('div', class_='ih-pcard-wrap').find('ul', id='identifiercls0')
        indiv_batsmen = batsmen.find_all('li','dys-box-color ih-pcard1')
        for bat in indiv_batsmen:
            link = bat.find('a')['href']
            indiv_player(link)
            #print(bat.find('a')['href'])

        allrounders = soup.find_all('div', class_="ih-pcard-sec")[1].find('div', class_='ih-pcard-wrap').find('ul', id='identifiercls1')
        indiv_all = allrounders.find_all('li','dys-box-color ih-pcard1')
        for allr in indiv_all:
            link = allr.find('a')['href']
            indiv_player(link)
        bowlers = soup.find_all('div', class_="ih-pcard-sec")[2].find('div', class_='ih-pcard-wrap').find('ul', id='identifiercls2')
        indiv_bowler = batsmen.find_all('li','dys-box-color ih-pcard1')
        for bowl in indiv_bowler:
            link = bowl.find('a')['href']
            indiv_player(link)

def indiv_player(link):
    req = requests.get(link)
    soup = BeautifulSoup(req.content, 'html.parser')

    section = soup.find('div','vn-teamOverviewWrap col-100 floatLft m-0').find('section', class_='squad-member-detail-main position-relative')
    #member_details = soup.find('div',class_='container').find('div', class_='row').find().find('div',class_='membr-details').find('div',class_='membr-details-img position-relative')
    member_details = section.find().find().find().find().find()
    img = member_details.find('img')['src']
    name_nationality = member_details.find('div',class_='plyr-name-nationality')
    name = name_nationality.find('h1').text.strip()
    nationality = name_nationality.find('span').text.strip()
    #print(name, nationality, img)
    add(img,name,nationality)

def add(img, name, nationality):
    #df[df['Name']==name]['Image'] = img
    df.loc[df['Name']==name,'Image'] = img
    nation = "Indian" if nationality=="Indian" else "Overseas"
    df.loc[df['Name']==name,'Nationality'] = nation
    #df[df['Name']==name]['Nationality'] = nation
    #df.to_csv('ipl_2023_dataset.csv')

#df.to_csv('ipl_2023_dataset.csv')

#player_link()

#print(df.head())

a= lambda x: x.strip()
df['Name'] = df['Name'].apply(a)

player_link()

df.to_csv('ipl_2023_dataset.csv')

#df['Name'] = df['Name'].values.astype('str')
'''
print(df[df['Name'] == 'Ravindra Jadeja'])
print(type(df.iloc[0]['Name']))
print((df.iloc[0]['Name'] == "Ben Stokes"))
#print(df['Name'])
'''