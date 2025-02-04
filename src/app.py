# Packages used in the system
# Pacotes utilizados no sistema
import os
import sys

PROJECT_ROOT = os.path.abspath( # Getting Obtaining the absolute normalized version of the project root path (Obtendo a versão absoluta normalizada do path raíz do projeto)
    os.path.join( # Concatenating the paths (Concatenando os paths)
        os.path.dirname(__file__), # Getting the path of the scripts directory (Obtendo o path do diretório dos scripts do projeto)
        os.pardir # Gettin the constant string used by the OS to refer to the parent directory (Obtendo a string constante usada pelo OS para fazer referência ao diretório pai)
    )
)
# Adding path to the list of strings that specify the search path for modules
# Adicionando o path à lista de strings que especifica o path de pesquisa para os módulos
sys.path.append(PROJECT_ROOT)
from utils.ingestion_utils import request_auth, playlist_to_dataframe
from utils.preprocessing_utils import get_user_dataset
from utils.model_utils import model_compile
from utils.load_utils import *

import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import StandardScaler

from warnings import simplefilter
simplefilter('ignore')


# Setting the environment variables (Definindo as variáveis de ambiente)
CLIENT_ID = os.environ['CLIENT_ID_SPOTIFY']
CLIENT_SECRET = os.environ['CLIENT_SECRET_SPOTIFY']

# Setting global variables with the path of each directory with data (Definindo as variáveis globais com o path de cada diretório com os dados).
PATH_T = os.path.join(PROJECT_ROOT, 'data\\transformed')
PATH_P = os.path.join(PROJECT_ROOT, 'data\\preprocessed')
PATH_M = os.path.join(PROJECT_ROOT, 'model')

# Reading pre-processed datasets (Lendo os datasets pré-processados)
item_train_norm = pd.read_csv(os.path.join(PATH_P, 'item_train_norm.csv'))
user_train_norm = pd.read_csv(os.path.join(PATH_P, 'user_train_norm.csv'))

# Requesting Spotify Authorization (Requisitando a autorização do Spotify)
access_token, token_type, token_expires = request_auth(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
print(f'The token expires in: {token_expires}s')

# Setting the playlist to be extracted for inference in the model (Definindo a playlist que será extraida para inferência no modelo)
playlist = 'https://open.spotify.com/playlist/37i9dQZEVXbMDoHDwVN2tF?si=e850a816edf44117'

# Extracting playlist data for recommendation (Extraindo os dados da playlist para recomendação)
df_rec = playlist_to_dataframe(playlist, token_type, access_token)

# Creating the `duration_min` feature (Criando a feature `duration_min`).
df_rec['duration_min'] = df_rec['duration_ms'] / 60000
# Setting the dataset with only the track ID, song and artist names
# Definindo o dataset apenas com o ID das tracks, os nomes das músicas e dos artistas
items_rec = df_rec.iloc[:, :3].copy()

# Creating a dataset with only numerical features
# Criando um dataset apenas com as features numéricas
X_rec_num = df_rec.drop(columns=['id', 'name', 'artists', 'duration_ms']).copy()

# Creating the one-hot encoding (Criando o one-hot encoding).
key_oh = pd.get_dummies(X_rec_num['key'], prefix='key', drop_first=True, dtype=np.int64)

# Setting the numeric dataset without the feature key
# Definindo o dataset numérico sem a feature key
X_rec_num_wkey = X_rec_num.drop(columns=['key']).copy()
# Looping through the dataset columns X_num_key and key_oh to define them as feature names
# Percorrendo as colunas do dataset X_num_key e do key_oh para definí-las como nomes das features
columns_oh = {i: j for i, j in enumerate(X_rec_num_wkey.columns.tolist() + key_oh.columns.tolist())}

# Concatenating the datasets
# Concatenando os datasets
X_rec_num_oh = pd.concat(
    [X_rec_num_wkey, key_oh],
    axis=1,
    ignore_index=True,
).rename(columns=columns_oh)

# Setting the user dataset (Definindo o dataset do usuário
user_set = pd.read_csv('../data/preprocessed/user_train_norm.csv').iloc[:1, :]
user_rec = get_user_dataset(
    user_set,
    X_rec_num
)

# Applying z-score normalization to the training dataset and recommendation dataset (Aplicando a normalização z-score no dataset de treino e no dataset de recomendação)
# We calculate the mean and standard deviation of the training set, and then apply the z-score to the recommender dataset with the mean and standard deviation of the training set
# Calculamos a média e desvio-padrão do training set, e então, aplicamos o z-score para o dataset de recomendação com a média e o desvio-padrão do training set
item_train = pd.read_csv('../data/preprocessed/item_train_norm.csv')

item_scaler = StandardScaler()
item_train_norm = item_scaler.fit_transform(item_train)
item_rec_norm = item_scaler.transform(X_rec_num_oh)

# Defining the number of features for the user and item neural network (Definindo o número de features para a rede neural do usuário e do item)
num_user_features, num_item_features = user_rec.shape[1], item_rec_norm.shape[1]
print(f'num_user_features: {num_user_features}\nnum_item_features: {num_item_features}')

# Loading the pre-trained model (Carregando o modelo pré-treinado)
OPT = Adam()
LOSS = MeanSquaredError()

model = model_compile(OPT, LOSS, num_user_features, num_item_features)
model.load_weights(os.path.join(PATH_M, 'pretrained.weights.h5'))
print(f'model ummary: {model.summary()}')

# Running inference on the model (Executando a inferência no modelo)
y_hat_rec = model.predict(
    [user_rec, item_rec_norm],
    verbose=0
)

# Plotting the top $m$ recommendations on the recommender data (Plotando as $m$ primeiras recomendações sobre os dados de recomendações)
m = 10

# Sorting the vector with the predictions in descending order and selecting only the indexes
# Ordenando o vetor com as previsões em ordem decrescente e selecionando apenas os índices
sorted_idx_rec = np.argsort(-y_hat_rec, axis=0).squeeze().tolist()
# Slicing the recommended indexes in the recommender items dataset, to sort it in the same order
# Selecionando os índices recomendados no dataset de items de recomendações, para ordená-lo na mesma ordem
sorted_items_rec = items_rec.iloc[sorted_idx_rec].reset_index().drop(columns=['index']).copy()
# Plotting the top m recommended items
# Plotando os top m items recomendados
print(sorted_items_rec.head(m))

# Requesting User ID (Requisitando o ID do usuário)
user_id = get_user(token_type, access_token)

# Creating the playlist where the tracks will be added (Criando a playlist onde as tracks serão adicionadas)
playlist_id = create_playlist(user_id, token_type, access_token)

# Adding tracks to the playlist (Adicionando as tracks na playlist)
# Turning the uris of recommended tracks into a list (Transformando os uris das tracks recomendadas em uma lista)
uris = items_rec.loc[sorted_idx_rec, 'uri'].tolist()
add_tracks(playlist_id, uris, token_type, access_token)
