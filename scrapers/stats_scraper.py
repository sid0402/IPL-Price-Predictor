import pandas as pd
from bs4 import BeautifulSoup 
import requests
import pickle

df = pd.read_csv('AuctionId.csv')

linkstr = "http://www.howstat.com/cricket/Statistics/IPL/PlayerOverview.asp?PlayerID="


req = requests.get(linkstr+str(3845))
soup = BeautifulSoup(req.content, 'html.parser')

'''
matches = soup.find('td',string='Matches:').find_next_sibling('td')
print(matches.text.strip().split()[0])

year_tag = soup.find('td',string="Born:").find_next_sibling('td')
print(int(year_tag.text.strip().split()[-1]))
'''
basic_stats_collection = []

def basic_stats():
    for id in df['id'].unique():
        req = requests.get(linkstr+str(id))
        soup = BeautifulSoup(req.content, 'html.parser')

        fields = soup.find_all('span','FieldName2')
        values = soup.find_all('td',['FieldValue','FieldValueAsterisk'])

        text_values=[]
        for value in values:
            text_values.append(value.text.replace('\n','').replace('\t','').replace('\r','').replace('*','').strip())

        text_fields=[]
        for field in fields:
            text_fields.append(field.text.replace(':',''))

        try:
            matches_tag = soup.find('td',string='Matches:').find_next_sibling('td')
            matches = int(matches_tag.text.strip().split()[0])

            year_tag = soup.find('td',string="Born:").find_next_sibling('td')
            birthyear = int(year_tag.text.strip().split()[-1])

        except:
            matches = 0
            birthyear = 0

        print(id)

        basic_stats_dict = {'id':id, 'matches':matches,'birthyear':birthyear}
        for i in range(len(text_fields)):
            basic_stats_dict[text_fields[i]] = text_values[i]
        basic_stats_collection.append(basic_stats_dict)

    return basic_stats_collection


basic_stats_collection = basic_stats()

filehandler = open('scrapers/ScrapedData/basic_stats_dict.txt', 'wb')
pickle.dump(basic_stats_collection, filehandler)
filehandler.close()



'''
linkstr = "http://www.howstat.com/cricket/statistics/IPL/PlayerProgressBat.asp?PlayerID="

#print(rows[1])
#print(int(rows[0].find('a').text.split('/')[2]))



progressive_collection = []

def batting_advanced():
    for i in range(len(df)):
        id = df.loc[i,'id']
        auction_year = int(df.loc[i,'Year'])

        print(id)
        print(auction_year)

        req = requests.get(linkstr+str(id))
        soup = BeautifulSoup(req.content, 'html.parser')

        rows = soup.find('table','TableLined').find().find_all('tr')
        progressive_dict={}
        for row in rows:
            try:
                date = int(row.find('a').text.split('/')[2])
                if (date<auction_year):
                    aggr = int(row.find_all('td')[11].text.strip())
                    avg = float(row.find_all('td')[12].text.strip())
                    sr = float(row.find_all('td')[13].text.strip())
                    progressive_dict = {'id':id,'auction_year':auction_year}
                    progressive_dict['prog_agg'] = aggr
                    progressive_dict['prog_avg'] = avg                    
                    progressive_dict['prog_sr'] = sr
            except Exception:
                continue

        if (len(progressive_dict)!=0):
            progressive_collection.append(progressive_dict)
    return progressive_collection

progressive_collection = batting_advanced()

filehandler = open('scrapers/ScrapedData/progressive_battingcollection.txt', 'wb')
pickle.dump(progressive_collection, filehandler)
filehandler.close()
'''

'''
linkstr = "http://www.howstat.com/cricket/statistics/IPL/PlayerProgressBowl.asp?PlayerID="

progressive_collection = []

def bowling_advanced():
    for i in range(len(df)):
        id = df.loc[i,'id']
        auction_year = int(df.loc[i,'Year'])

        print(id)
        print(auction_year)

        req = requests.get(linkstr+str(id))
        soup = BeautifulSoup(req.content, 'html.parser')

        rows = soup.find('table','TableLined').find().find_all('tr')
        progressive_dict={}
        for row in rows:
            try:
                date = int(row.find('a').text.split('/')[2])
                if (date<auction_year):
                    prog_bowlsr = (row.find_all('td')[9].text.strip())
                    prog_er = (row.find_all('td')[10].text.strip())
                    prog_wkts = (row.find_all('td')[11].text.strip())
                    prog_bowlavg = (row.find_all('td')[12].text.strip())
                    prog_matches = int(row.find().text.strip())
                    progressive_dict = {'id':id,'auction_year':auction_year}
                    progressive_dict['prog_bowlsr'] = prog_bowlsr
                    progressive_dict['prog_er'] = prog_er                   
                    progressive_dict['prog_wkts'] = prog_wkts
                    progressive_dict['prog_bowlavg'] = prog_bowlavg
                    progressive_dict['prog_matches'] = prog_matches
            except Exception:
                continue

        if (len(progressive_dict)!=0):
            progressive_collection.append(progressive_dict)
    return progressive_collection


progressive_collection = bowling_advanced()

filehandler = open('scrapers/ScrapedData/progressive_bowlingollection.txt', 'wb')
pickle.dump(progressive_collection, filehandler)
filehandler.close()
'''
'''
linkstr = "http://www.howstat.com/cricket/statistics/IPL/PlayerSeries.asp?PlayerID="

last_collection = []
def last_bat():
    for i in range(len(df)):
        id = df.loc[i,'id']
        auction_year = int(df.loc[i,'Year'])

        print(id)
        print(auction_year)

        req = requests.get(linkstr+str(id)+"#bat")
        soup = BeautifulSoup(req.content, 'html.parser')

        rows = soup.find('table','TableLined').find_all('tr')

        last_dict = {}
        for row in rows:
            try:
                date = int(row.find('a').text.strip())
                if (date<auction_year):
                    last_dict = {'id':id,'auction_year':auction_year}
                    last_dict['last_matches'] = (row.find_all('td')[1].text.strip())
                    last_dict['last_inn'] = (row.find_all('td')[2].text.strip())
                    last_dict['last_hs'] = (row.find_all('td')[7].text.strip())
                    last_dict['last_runs'] = (row.find_all('td')[8].text.strip())
                    last_dict['last_avg'] = (row.find_all('td')[9].text.strip())
                    last_dict['last_sr'] = (row.find_all('td')[10].text.strip())
                    last_dict['last_batpercent'] = (row.find_all('td')[11].text.strip())
            except Exception as e:
                continue

        if (len(last_dict)!=0):
            last_collection.append(last_dict)            
    return last_collection

last_collection = last_bat()

filehandler = open('scrapers/ScrapedData/last_bat.txt', 'wb')
pickle.dump(last_collection, filehandler)
filehandler.close()
'''

'''
linkstr = "http://www.howstat.com/cricket/statistics/IPL/PlayerSeries.asp?PlayerID="

last_collection = []
check_ids = []
def last_bowl():
    for i in range(len(df)):
        id = df.loc[i,'id']
        auction_year = int(df.loc[i,'Year'])

        print(id)
        print(auction_year)

        req = requests.get(linkstr+str(id))
        soup = BeautifulSoup(req.content, 'html.parser')

        try:
            rows = soup.find_all('table','TableLined')[1].find_all('tr')
        except:
            continue

        last_dict = {}
        for row in rows:
            try:
                date = int(row.find('a').text.strip())
                if (date<auction_year):
                    last_dict = {'id':id,'auction_year':auction_year}
                    last_dict['last_matches'] = (row.find_all('td')[1].text.strip())
                    last_dict['overs'] = (row.find_all('td')[2].text.strip())
                    last_dict['last_wkts'] = (row.find_all('td')[5].text.strip())
                    last_dict['last_bowlavg'] = (row.find_all('td')[8].text.strip())
                    last_dict['last_bowlsr'] = (row.find_all('td')[9].text.strip())
                    last_dict['last_bowler'] = (row.find_all('td')[10].text.strip())
                    last_dict['last_bowlpercent'] = (row.find_all('td')[11].text.strip())
            except Exception as e:
                continue

        if (len(last_dict)!=0):
            last_collection.append(last_dict)          
    return last_collection

last_collection = last_bowl()

filehandler = open('scrapers/ScrapedData/last_bowl.txt', 'wb')
pickle.dump(last_collection, filehandler)
filehandler.close()
'''
