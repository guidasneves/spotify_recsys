import os
from requests import post, get
import json
from base64 import b64encode
from dotenv import load_dotenv
load_dotenv()

import pandas as pd

client_id = os.environ['CLIENT_ID_SPOTIFY']
client_secret = os.environ['CLIENT_SECRET_SPOTIFY']
base_url = 'https://accounts.spotify.com'
endpoint = '/api/token'
redirect_uri = 'http://localhost:3000'

