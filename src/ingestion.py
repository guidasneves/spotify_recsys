# Packages used in the system
# Pacotes utilizados no sistema
import os
import sys
from requests import post, get
from base64 import b64encode
from dotenv import load_dotenv
load_dotenv() # access environment variables (acessa as variáveis de ambiente)

import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath( # Getting the absolute normalized path version (Obtendo a versão absoluta normalizada do path)
    os.path.join( # Concatenating the paths (Concatenando os paths)
        os.path.dirname(__file__), # Getting the path of the scripts directory (Obtendo o path do diretório dos scripts do projeto)
        os.pardir # Gettin the constant string used by the OS to refer to the parent directory (Obtendo a string constante usada pelo OS para fazer referência ao diretório pai)
    )
)
# Adding path to the list of strings that specify the search path for modules
# Adicionando o path à lista de strings que especifica o path de pesquisa para os módulos
sys.path.append(PROJECT_ROOT)
from utils.ingestion_utils import *

# Setting the environment variables
# Definindo as variáveis de ambiente
CLIENT_ID = os.environ['CLIENT_ID_SPOTIFY']
CLIENT_SECRET = os.environ['CLIENT_SECRET_SPOTIFY']

# Requesting Spotify Authorization (Requisitando a autorização do Spotify)
access_token, token_type, token_expires = request_auth(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)

# Setting the playlists to be extracted (Definindo as playlists que serão extraidas)
good_songs = 'https://open.spotify.com/playlist/6DI0NiX9bE3fIF6cEoI2zL?si=ef8a610d53f64627'
good_songs_2 = 'https://open.spotify.com/playlist/08bsg8CsImua5vzGMoiGLT?si=7ea323f1c8404b0b'
bad_songs = 'https://open.spotify.com/playlist/6IBody2iNg5TgmAeYiHYpW?si=xsEvNjbbQYqt0rs9wP3yOg'

# Extracting the data from playlists (Extraindo os dados das plalists).
df_good = playlist_to_dataframe(good_songs, token_type, access_token, label=1)
df_good_2 = playlist_to_dataframe(good_songs_2, token_type, access_token, label=1)
df_bad = playlist_to_dataframe(bad_songs, token_type, access_token, label=0).drop(columns=['name'])

# Creating the `duration_min` feature (Criando a feature `duration_min`).
df_good['duration_min'] = df_good['duration_ms'] / 60000
df_good_2['duration_min'] = df_good_2['duration_ms'] / 60000
df_bad['duration_min'] = df_bad['duration_ms'] / 60000
display(df_good.head())

# Loading each dataset into the `./data/raw/` directory (Carregando cada dataset no diretório `./data/raw/`)
path = os.path.join(PROJECT_ROOT, 'data\\raw')
df_good.to_csv(os.path.join(PROJECT_ROOT, 'df_good.csv'), index=False)
df_good_2.to_csv(os.path.join(PROJECT_ROOT, 'df_good_2.csv'), index=False)
df_bad.to_csv(os.path.join(PROJECT_ROOT, 'df_bad.csv'), index=False)
