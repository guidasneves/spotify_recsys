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
from utils.model_utils import model_compile

import pandas as pd
import numpy as np
import numpy.ma as ma
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15, 15) # set default size of plots (definindo o tamanho padrão dos plots)
import plotly.express as px

from warnings import simplefilter
simplefilter('ignore')


# Checking tensorflow version (Verificando a versão do tensorflow)
print(f'TensorFlow version: {tf.__version__}')

# Setting global variables with the path of each directory with data (Definindo as variáveis globais com o path de cada diretório com os dados).
PATH_T = os.path.join(PROJECT_ROOT, 'data\\transformed')
PATH_P = os.path.join(PROJECT_ROOT, 'data\\preprocessed')
PATH_M = os.path.join(PROJECT_ROOT, 'model')

# Reading pre-processed datasets (Lendo os datasets pré-processados)
item_train_norm = pd.read_csv(os.path.join(PATH_P, 'item_train_norm.csv'))
user_train_norm = pd.read_csv(os.path.join(PATH_P, 'user_train_norm.csv'))

item_cv_norm = pd.read_csv(os.path.join(PATH_P, 'item_cv_norm.csv'))
user_cv_norm = pd.read_csv(os.path.join(PATH_P, 'user_cv_norm.csv'))

item_test_norm = pd.read_csv(os.path.join(PATH_P, 'item_test_norm.csv'))
y_test = pd.read_csv(os.path.join(PATH_P, 'y_cv.csv'))
user_test_norm = pd.read_csv(os.path.join(PATH_P, 'user_test_norm.csv'))

items = pd.read_csv(os.path_join(PATH_T, 'items.csv'))

# Defining the number of features for the user and item neural network (Definindo o número de features para a rede neural do usuário e do item)
num_user_features, num_item_features = user_train_norm.shape[1], item_train_norm.shape[1]
print(f'num_user_features: {num_user_features}\nnum_item_features: {num_item_features}')

# Loading the pre-trained model (Carregando o modelo pré-treinado)
OPT = Adam()
LOSS = MeanSquaredError()

model = model_compile(OPT, LOSS, num_user_features, num_item_features)
model.load_weights(os.path.join(PATH_M, 'pretrained.weights.h5'))
print(f'model ummary: {model.summary()}')

# Prediction with model only (Previsão apenas com o modelo)
# Performing prediction on training data (Realizando a previsão sobre os dados de treino).
y_hat_train = model.predict([user_train_norm, item_train_norm], verbose=0)
# Plotting the top $m$ recommendations on the training data (Plotando as $m$ primeiras recomendações sobre os dados de treino)
m = 10

# Sorting the vector with the predictions in descending order and selecting only the indexes
# Ordenando o vetor com as previsões em ordem decrescente e selecionando apenas os índices
sorted_idx_train = np.argsort(-y_hat_train, axis=0).squeeze().tolist()
# Slicing the recommended training indices from the items dataset and sorting them in the same order
# Selecionando os índices de treino recomendados no dataset de items e os ordenando na mesma ordem
sorted_items_train = items.iloc[sorted_idx_train].copy()
# Plotting the top m recommended items
# Plotando os top m items recomendados
print(sorted_items_train.head(m))

# Performing prediction on the validation data (Realizando a previsão sobre os dados de validação)
y_hat_cv = model.predict([user_cv_norm, item_cv_norm], verbose=0)
# Plotting the top $m$ recommendations on the validation data (Plotando as $m$ primeiras recomendações sobre os dados de validação)
m = 10

# Sorting the vector with the predictions in descending order and selecting only the indexes
# Ordenando o vetor com as previsões em ordem decrescente e selecionando apenas os índices
sorted_idx_cv = np.argsort(-y_hat_cv, axis=0).squeeze().tolist()
# Slicing the recommended validation indices from the items dataset and sorting them in the same order
# Selecionando os índices de validação recomendados no dataset de items e os ordenando na mesma ordem
sorted_items_cv = items.iloc[sorted_idx_cv].copy()
# Plotting the top m recommended items
# Plotando os top m items recomendados
print(sorted_items_cv.head(m))

# Performing prediction on test data (Realizando a previsão sobre os dados de teste)
y_test = pd.read_csv('../data/preprocessed/y_test.csv')
test_eval = model.evaluate([user_test_norm, item_test_norm], y_test, verbose=0)
print(f'Test set evaluation: {test_eval[0]:.2f}')

y_hat_test = model.predict([user_test_norm, item_test_norm], verbose=0)
# Plotting the top $m$ recommendations on the test data (Plotando as $m$ primeiras recomendações sobre os dados de teste)
m = 10

# Sorting the vector with the predictions in descending order and selecting only the indexes
# Ordenando o vetor com as previsões em ordem decrescente e selecionando apenas os índices
sorted_idx_test = np.argsort(-y_hat_test, axis=0).squeeze().tolist()
# Slicing the recommended test indices from the items dataset and sorting them in the same order
# Selecionando os índices de teste recomendados no dataset de items e os ordenando na mesma ordem
sorted_items_test = items.iloc[sorted_idx_test].copy()
# Plotting the top m recommended items
# Plotando os top m items recomendados
print(sorted_items_test.head(m))

# Cosine similarity prediction (Previsão com a similaridade de cosseno)
# Setting the model using only the item neural network (Definindo o modelo usando apenas a rede neural do item)
LOSS_ITEM = MeanSquaredError()

model_item = model_compile(OPT, LOSS_ITEM, num_user_features, num_item_features, item_similarity=True)
print(f'model item ummary: {model_item.summary()}')

# Collecting the item network weights from the pre-trained complete neural network and setting them in the new network
# Coletando os pesos da rede do item, da rede neural completa pré-treinada e definindo na nova rede
item_nn_weights = model.get_layer('item_NN').get_weights()
model_item.get_layer('item_NN').set_weights(item_nn_weights)

# Scaling the complete item matrix (Escalando a matriz de itens completa)
# Applying z-score normalization to each dataset, so that they have a mean of 0 and a standard deviation of 1
# Aplicando a normalização z-score em cada dataset, para eles terem média 0 e desvio padrão 1
# We calculate the mean and standard deviation of the training set, and then apply the z-score to all dataset with the mean and standard deviation of the training set
# Calculamos a média e desvio-padrão do training set, e então, aplicamos o z-score para todos os dataset com a média e o desvio-padrão do training set
X_pre = pd.read_csv(os.path.join(PATH_P, 'X_pre.csv'))
X_scaler = StandardScaler()
X_norm = X_scaler.fit_transform(X_pre)

# Calculating the feature vector of all items (Calculando o vetor de features de todos os itens)
item_feature_vectors = model_item.predict(X_norm, verbose=0)
print(f'Size of all predicted item feature vectors: {item_feature_vectors.shape}')

# Computing cosine similarity for all items (Calculando a similaridade de cosseno para todos os itens)
dim = item_feature_vectors.shape[0]
cos_sim = np.zeros((dim, dim))

# Looping through all items and calculating the cosine similarity between each different item
# Percorrendo todos os itens e calculando a similaridade de cosseno entre cada item diferente
for i in range(dim):
    for j in range(dim):
        cos_sim[i, j] = cosine_similarity(
            item_feature_vectors[i, :].reshape(1, -1),
            item_feature_vectors[j, :].reshape(1, -1)
        )
# Avoiding selecting the same item. The masked values along the diagonal won't be included in the computation
# Evitando selecionar o mesmo item. Os valores mascarados ao longo da diagonal não serão incluídos no cálculo
m_cos_sim = ma.masked_array(cos_sim, mask=np.identity(dim))

# Number of recommended tracks to be added to the playlist
# Quantidade de tracks recomendadas para ser adicionada à playlist 
m = 20
# List to store the indexes, to add to the playlist later
# Lista para armazenar os índices, para adicionar à playlist posteriormente
idx = []
# List to store each example
# Lista para armazenar cada exemplo
data = []
# Looping through each example and selecting the index of the most similar vector
# Percorrendo cada exemplo e selecionando o índice do vetor mais similar
for i in range(dim):
    max_idx = np.argmax(m_cos_sim[i])
    example = [
        items.loc[i, 'name'],
        items.loc[i, 'artists'],
        items.loc[max_idx, 'name'],
        items.loc[max_idx, 'artists']
    ]
    data.append(example)
    if i < m:
        idx.append(max_idx)

# Creating the dataframe with the items and their most similar items
# Criando o dataframe com os itens e seus itens mais similares
features = ['name_root', 'artists_root', 'name_recommended', 'artists_recommended']
cos_sim_rec = pd.DataFrame(data, columns=features)
# Number of items to display (Número de itens a serem exibidos)
m = 20
print(cos_sim_rec.head(m))

# Applying PCA with only 3 principal components ($Z$ axes) to the dataset (Aplicando o PCA com apenas 3 principal components (eixos $Z$) ao dataset)
pca_cos = PCA(n_components=3)
# Fitting PCA to our original scaled dataset
# Ajustando o PCA ao nosso dataset original escalado
item_features_trans = pca_cos.fit_transform(item_feature_vectors)
# Creating a dataframe with the matrix transformed into only 3 features
# Criando um dataframe com a matriz transformada em apenas 3 features
df_item_pca = pd.DataFrame(
    item_features_trans,
    columns=['principal_component_1', 'principal_component_2', 'principal_component_3']
)
print(f'Explained variance: {pca_cos.explained_variance_ratio_}')

# Plotting the dataset in 2D after PCA transformation (Plotando o dataset em 2D após a transformação do PCA)
fig = plt.figure(figsize=(5, 5))
plt.scatter(df_item_pca['principal_component_1'],
            df_item_pca['principal_component_2'],
            color='g')
plt.xlabel('principal_component_1')
plt.ylabel('principal_component_2')
plt.title('PCA Decomposition')
plt.show()

# Plotting the dataset in 3D after PCA transformation (Plotando o dataset em 3D após a transformação do PCA)
fig = px.scatter_3d(df_item_pca,
                    x='principal_component_1',
                    y='principal_component_2',
                    z='principal_component_3',
                    width=800,
                    height=700).update_traces(marker=dict(color='green'))
fig.show()
