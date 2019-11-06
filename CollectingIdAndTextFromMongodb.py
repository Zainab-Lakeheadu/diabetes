import pymongo
import pandas as pd
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
db = myclient.Twitter
mycol = db["Tweets"]
data = []
for x in mycol.find({},{ "_id": 0, "id_str": 1, "text": 1 }):
  data.append(x)
  print(x)
df = pd.DataFrame(data)
df.to_csv(r'C:\Users\Kazi Zainab Khanam\Documents\T\IdWithTweets.csv',index =False)

