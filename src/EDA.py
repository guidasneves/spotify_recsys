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
from utils.plot_utils import *

import pandas as pd
import numpy as np
import numpy.ma as ma
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from random import randint

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15, 15) # set default size of plots (definindo o tamanho padrão dos plots)
import seaborn as sns
import plotly.express as px

# Reading datasets from the `../data/raw/` directory (Lendo os datasets do diretório `../data/raw/`)
path = os.path.join(PROJECT_ROOT, 'data\\raw')
good_df = pd.read_csv(os.path.join(path, 'df_good.csv'))
bad_df = pd.read_csv(os.path.join(path, 'df_bad.csv')).drop(columns=['name'])

# Dropping duplicate examples (Dropando os exemplos duplicados)
good_df = good_df.drop_duplicates()
bad_df = bad_df.drop_duplicates()

# Analyzing the distribution of the `artists` feature for each dataset, the good and the bad (Analisando a distribuição da feature `artists` para cada dataset, o good e o bad)
print(
    f'{"=" * 30}\ngood_df:\n{good_df['artists'].value_counts().sort_values(ascending=False).head(10)}\n\n \
    bad_df:\n{bad_df['artists'].value_counts().sort_values(ascending=False).head(10)}'
)

print(f'{"=" * 30}\nTotal tracks in the good playlist: {good_df.shape[0]}\nTotal tracks in the bad playlist: {bad_df.shape[0]}')

# Counting null values (Contando os valores nulos)
print('=' * 30, '\ngood_df:\n', good_df.isnull().sum(), '\n\nbad_df:\n', bad_df.isnull().sum())

# Performing descriptive analysis (Executando a análise descritiva).
print('good_df:\n', good_df.describe().T, 'bad_df:\n', bad_df.describe().T)

# Creating the item's dataset `X` with the sets of each class (Criando o dataset `X` dos itens com os sets de cada classe)
X = pd.concat([good_df, bad_df], axis=0, ignore_index=True).fillna('explicit')
print(f'Total tracks in the items dataset: {X.shape[0]}')

# Loading the concatenated dataset in the `../data/transformed/dataset.csv` directory (Carregando o dataset concatenado no diretório `../data/transformed/dataset.csv`)
path_t = os.path.join(PROJECT_ROOT, 'data\\transformed')
X.to_csv(os.path.join(path_t, 'dataset.csv'), index=False)

# Plotting the comparison of the distribution between `label 1` (good track) and `label 0` (bad track) of the dataset of each numerical feature
# Plotando a comparação da distribuição entre o `label 1` (good track) e o `label 0` (bad track) do dataset de cada feature numérica)
plot_hist_vs(X.drop(columns=['id']))

# Creating a dataset with only numerical features (Criando um dataset apenas com as features numéricas)
X_num = X.drop(columns=['id', 'name', 'artists', 'y', 'duration_ms']).copy()

# Plotting the distribution between each 25 different random pairs of features (Plotando a distribuição entre cada 25 pares aleatórios diferentes das features)
pairs = get_pairs(X_num)

fig, axs = plt.subplots(5, 5)
i = 0
for rows in axs:
    for ax in rows:
        ax.scatter(X_num[pairs[i][0]], X_num[pairs[i][1]], c='g')
        ax.set_xlabel(pairs[i][0])
        ax.set_ylabel(pairs[i][1])
        i+=1
plt.show()

# Computing the correlation between each feature (Calculando a correlação entre cada feature)
X_corr = X_num.corr()
print(X_corr)

# Applying a condition to view only correlations above a threshold and different from 1
# Aplicando uma condição para visualizar apenas as correlações acima de um threshold e diferente de 1
mask = (abs(X_corr) > .4) & (abs(X_corr) != 1)
print('=' * 50, X_corr.where(mask).stack().sort_values(ascending=False))

# Plotting the heatmap with the correlation between all features (Plotando o heatmap com a correlação entre todas as features)
sns.heatmap(X_corr, annot=True, fmt='.2f', cmap='PuBuGn')
plt.show()

# Applying normalization to facilitate PCA computation. Detailed explanation of normalization in notebooks `03_preprocessing.ipynb`
# Aplicando a normalização para facilitar a computação do PCA. Explicação detalhada sobre a normalização no notebooks `03_preprocessing.ipynb`
X_num_norm = StandardScaler().fit_transform(X_num)

# Instantiating PCA with 3 principal components
# Instânciando o PCA com 3 principal components
pca = PCA(n_components=3)
# Fitting PCA to our original scaled dataset
# Ajustando o PCA ao nosso dataset original escalado
X_num_trans = pca.fit_transform(X_num_norm)
# Creating a dataframe with the matrix transformed into only 3 features
# Criando um dataframe com a matriz transformada em apenas 3 features
df_pca = pd.DataFrame(
    X_num_trans,
    columns=['principal_component_1', 'principal_component_2', 'principal_component_3']
)
print(f'Explained variance: {pca.explained_variance_ratio_}')

# Plotting the dataset in 2D after PCA transformation (Plotando o dataset em 2D após a transformação do PCA)
plt.scatter(df_pca['principal_component_1'],
            df_pca['principal_component_2'],
            c='g')
plt.xlabel('principal_component_1')
plt.ylabel('principal_component_2')
plt.title('PCA Decomposition')
plt.show()

# Plotting the dataset in 3D after PCA transformation (Plotando o dataset em 3D após a transformação do PCA)
fig = px.scatter_3d(df_pca,
                    x='principal_component_1',
                    y='principal_component_2',
                    z='principal_component_3',
                    width=800,
                    height=600).update_traces(marker=dict(color='green'))
fig.show()
