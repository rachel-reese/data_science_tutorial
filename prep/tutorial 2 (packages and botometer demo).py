from dotenv import load_dotenv
import os
import botometer
from pprint import pprint

load_dotenv()

rapidapi_key = os.getenv('RAPIDAPI_KEY')
twitter_app_auth = {
    'consumer_key': os.getenv('TWITTER_API_KEY'),
    'consumer_secret': os.getenv('TWITTER_API_SECRET'),
}
bom = botometer.Botometer(wait_on_ratelimit=True,
                          rapidapi_key=rapidapi_key,
                          **twitter_app_auth)
accounts = ["@roun_sa_ville"]

for id, result in bom.check_accounts_in(accounts):
    pprint(result)
