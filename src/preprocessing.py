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
from utils.preprocessing_utils import *

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Setting global variables with the path of each directory with data (Definindo as variáveis globais com o path de cada diretório com os dados).
PATH_R = os.path.join(PROJECT_ROOT, 'data\\raw')
PATH_T = os.path.join(PROJECT_ROOT, 'data\\transformed')
PATH_P = os.path.join(PROJECT_ROOT, 'data\\preprocessed')

# Reading the dataset from the `../data/transformed/` directory (Lendo o dataset do diretório `../data/transformed/`)
data = pd.read_csv(os.path.join(PATH_T, 'data.csv'))

# Defining the dataset with only the track ID, song and artist names (Definindo o dataset apenas com o ID das tracks, os nomes das músicas e dos artistas)
items = data.iloc[:, :3].copy()
# Loading the items set in the `../data/tranfomed/` directory (Carregando o set de itens no diretório `../data/preprocessed/`)
items.to_csv(os.path.join(PATH_T, 'items.csv'), index=False)

# Creating a dataset with only numerical features (Criando um dataset apenas com as features numéricas)
X_num = data.drop(columns=['id', 'name', 'artists', 'y', 'duration_ms']).copy()

# Creating the one-hot encoding of the `key` feature (Criando o one-hot encoding da feature `key`)
key_oh = pd.get_dummies(X_num['key'], prefix='key', drop_first=True, dtype=np.int64)

# Setting the numeric dataset without the feature key
# Definindo o dataset numérico sem a feature key
X_num_wkey = X_num.drop(columns=['key']).copy()
# Looping through the dataset columns X_num_key and key_oh to define them as feature names
# Percorrendo as colunas do dataset X_num_key e do key_oh para definí-las como nomes das features
columns_oh = {i: j for i, j in enumerate(X_num_wkey.columns.tolist() + key_oh.columns.tolist())}

# Concatenating the datasets
# Concatenando os datasets
X_num_oh = pd.concat(
    [X_num_wkey, key_oh],
    axis=1,
    ignore_index=True,
).rename(columns=columns_oh)

# Creating the user dataset given the average of each feature from `good_df` (Criando o dataset do usuário dada a média de cada feature do `good_df`)
good_df = pd.read_csv(os.path.join(PATH_R, 'df_good.csv'))

# Creating the good dataset with only the numerical features
# Criando o dataset good apenas com as features numéricas
good_df_num = good_df.drop(columns=['id', 'name', 'artists', 'y', 'duration_ms']).copy()
# Calculating the average of each feature to create the user dataset
# Calculando a média de cada feature para criar o dataset do usuário
user_vec = [[i for i in good_df_num.mean(axis=0)]]
user_df = get_user_dataset(
    user_vec,
    X_num
)

# Exporting the pre-processed numeric dataset into the `../data/preprocessed/` directory, however, without scaling, to adjust the final playlist dataset for the recommendations
# Exportando o dataset numérico pré-processado no diretório `../data/preprocessed/`, porém, sem o escalonamento, para ajustar o dataset da playlist final para as recomendações
X_num_oh.to_csv(os.path.join(PATH_P, 'X_pre.csv'), index=False)

# Creating the column vector of the target label y to be divided along
# Criando o vetor de coluna do target label y para ser divido junto
y = data.iloc[:, -2].copy().to_numpy().reshape((-1, 1))

# Splitting the dataset between the training, validation and test sets (Dividindo o dataset entre o set de treino, validação e teste)
item_train, item_, y_train, y_ = train_test_split(X_num_oh, y, test_size=.4, random_state=42)
user_train, user_ = train_test_split(user_df, test_size=.4, random_state=42)
print(f'item_train.shape: {item_train.shape}\ny_train.shape: {y_train.shape}\n')
print(f'user_train.shape: {user_train.shape}\n')

item_cv, item_test, y_cv, y_test = train_test_split(item_, y_, test_size=.5, random_state=42)
user_cv, user_test = train_test_split(user_, test_size=.5, random_state=42)
print(f'item_cv.shape: {item_cv.shape}, item_test.shape: {item_test.shape}\ny_cv.shape: {y_cv.shape}, y_test.shape: {y_test.shape}\n')
print(f'user_cv.shape: {user_cv.shape}, user_test.shape: {user_test.shape}')

# Applying z-score normalization to each dataset, so that they have a mean of 0 and a standard deviation of 1
# Aplicando a normalização z-score em cada dataset, para eles terem média 0 e desvio padrão 1
# We calculate the mean and standard deviation of the training set, and then apply the z-score to all datasets with the mean and standard deviation of the training set
# Calculamos a média e desvio-padrão do training set, e então, aplicamos o z-score para todos os datasets com a média e o desvio-padrão do training set
item_scaler = StandardScaler()
user_scaler = StandardScaler()

item_train_norm = item_scaler.fit_transform(item_train)
user_train_norm = user_scaler.fit_transform(user_train)

item_cv_norm = item_scaler.transform(item_cv)
user_cv_norm = user_scaler.transform(user_cv)

item_test_norm = item_scaler.transform(item_test)
user_test_norm = user_scaler.transform(user_test)

# Defining the columns of each dataset (Definindo as colunas de cada dataset)
item_columns, user_columns = item_train.columns, user_train.columns

# Loading preprocessed sets into the `../data/preprocessed/` directory (Carregando os sets pré-processados no diretório `../data/preprocessed/`)
# Item train dataset
pd.DataFrame(
    item_train_norm,
    columns=item_columns
).to_csv(os.path.join(PATH_P, 'item_train_norm.csv'), index=False)
# Train target y
pd.DataFrame(
    y_train,
    columns=['y']
).to_csv(os.path.join(PATH_P, 'y_train.csv'), index=False)
# User train dataset
pd.DataFrame(
    user_train_norm,
    columns=user_columns
).to_csv(os.path.join(PATH_P, 'user_train_norm.csv'), index=False)

# Item cv dataset
pd.DataFrame(
    item_cv_norm,
    columns=item_columns
).to_csv(os.path.join(PATH_P, 'item_cv_norm.csv'), index=False)
# Cv target y
pd.DataFrame(
    y_cv,
    columns=['y']
).to_csv(os.path.join(PATH_P, 'y_cv.csv'), index=False)
# User cv dataset
pd.DataFrame(
    user_cv_norm,
    columns=user_columns
).to_csv(os.path.join(PATH_P, 'user_cv_norm.csv'), index=False)

# Item test dataset
pd.DataFrame(
    item_test_norm,
    columns=item_columns
).to_csv(os.path.join(PATH_P, 'item_test_norm.csv'), index=False)
# Test target y
pd.DataFrame(
    y_test,
    columns=['y']
).to_csv(os.path.join(PATH_P, 'y_test.csv'), index=False)
# User test dataset
pd.DataFrame(
    user_test_norm,
    columns=user_columns
).to_csv(os.path.join(PATH_P, 'user_test_norm.csv'), index=False)
