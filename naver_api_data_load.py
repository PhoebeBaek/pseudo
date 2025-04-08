import urllib.request
import urllib.parse
import json
import config
from pymongo import MongoClient

try:
    client = MongoClient(config.URI)
    db = client["dining_ai"]
    collection = db["items"]

    client_id = config.client_id
    client_secret = config.client_secret

    for n in range(1,1000,100):
        url = "https://openapi.naver.com/v1/search/shop.json?query=" + urllib.parse.quote("홈플러스장보기") + f"&start={n}&display=100"

        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id", client_id)
        request.add_header("X-Naver-Client-Secret", client_secret)

        response = urllib.request.urlopen(request)
        rescode = response.getcode()

        if rescode == 200:
            response_body = response.read()
            response = response_body.decode('utf-8')
            data = json.loads(response)["items"]
            # print(data)

            if data:
                collection.insert_many(data)
                print("{n} documents inserted!")
            else:
                print("No data to insert.")

        else:
            print("Error Code:", rescode)



except Exception as e:
    print("An error occurred:", str(e))


