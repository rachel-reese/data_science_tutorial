from dotenv import load_dotenv
import os
import botometer
from pprint import pprint

load_dotenv()

rapidapi_key =
twitter_app_auth = {
    'consumer_key': ,
    'consumer_secret': ,
}
bom = botometer.Botometer(wait_on_ratelimit=True,
                          rapidapi_key=rapidapi_key,
                          **twitter_app_auth)
accounts = ['@clayadavis', '@onurvarol', '@jabawack']

for id, result in bom.check_accounts_in(accounts):
    pprint(result)
