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
from utils.model_utils import *
from utils.plot_utils import plot_history

import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from skopt import gp_minimize
import shap

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15, 15) # set default size of plots (definindo o tamanho padrão dos plots)

from warnings import simplefilter
simplefilter('ignore')


# Checking tensorflow version (Verificando a versão do tensorflow)
print(f'TensorFlow version: {tf.__version__}')

# Setting global variables with the path of each directory with data (Definindo as variáveis globais com o path de cada diretório com os dados).
PATH_P = os.path.join(PROJECT_ROOT, 'data\\preprocessed')
PATH_M = os.path.join(PROJECT_ROOT, 'model')

# Reading pre-processed datasets (Lendo os datasets pré-processados)
item_train_norm = pd.read_csv(os.path.join(PATH_P, 'item_train_norm.csv'))
y_train = pd.read_csv(os.path.join(PATH_P, 'y_train.csv'))
user_train_norm = pd.read_csv(os.path.join(PATH_P, 'user_train_norm.csv'))

item_cv_norm = pd.read_csv(os.path.join(PATH_P, 'item_cv_norm.csv'))
y_cv = pd.read_csv(os.path.join(PATH_P, 'y_cv.csv'))
user_cv_norm = pd.read_csv(os.path.join(PATH_P, 'user_cv_norm.csv'))

# Defining the number of features for the user and item neural network (Definindo o número de features para a rede neural do usuário e do item)
num_user_features, num_item_features = user_train_norm.shape[1], item_train_norm.shape[1]
print(f'num_user_features: {num_user_features}\nnum_item_features: {num_item_features}')

# Computing Bayesian Optimization (Calculando a otimização bayesiana)
# Defining the range for testing each hyperparameter (Definindo a faixa para teste de cada hiperparâmetro)
space = [
    (1e-6, 1e-1, 'log-uniform'), # learning_rate
    (1e-6, 1e-1), # lambda_r
]

# Copying the scaled training items and user matrix to optimize the hyperparameters
# Copiando a matriz de itens e de usuário de treino escaladas para otimizar os hiperparâmetros
X_opt = item_train_norm.copy()
user_opt = user_train_norm.copy()

# Performing Bayesian optimization (Performando a bayesian optimization)
opt = gp_minimize(hiperparams_tune, space, random_state=42, verbose=0, n_calls=25, n_random_starts=10)
# Output with the best combinations of hyperparameters
# Output com as melhores combinações dos hiperparâmetros
print(f'Learning rate: {opt.x[0]}\nLambda: {opt.x[1]}')

# Setting the model (Definindo o modelo)
LR = opt.x[0]
LAMBDA = opt.x[1]
OPT = Adam(learning_rate=LR)
LOSS = MeanSquaredError()

model = model_compile(OPT, LOSS, num_user_features, num_item_features, lambda_r=LAMBDA)
print(f'model.summary: {model.summary()}')

# Computing SHAP values (Calculando os SHAP values)
item_model = model_compile(OPT, LOSS, num_user_features, num_item_features, lambda_r=LAMBDA, item_similarity=True)

explainer = shap.Explainer(item_model, item_train_norm)
shap_values = explainer(item_train_norm)

# Ploting SHAP values with beeswarm summary plot (Plotando os SHAP values com o beeswarm summary plot)
shap.plots.beeswarm(
    shap_values[:, :, 0],
    max_display=20,
    order=shap.Explanation.abs.max(0),
    show=False,
    #color=plt.get_cmap('PuBuGn')
)
# Setting the show parameter to False allows the plot to be customized further after it has been created,
# returning the current axis via matplotlib.pyplot.gca()
# Definir o parâmetro show como False permite que o gráfico seja personalizado ainda mais depois de ter sido criado,
# retornando o eixo atual via matplotlib.pyplot.gca()
ax = plt.gca()
# Selecting the index of each column of the training set of items
# Selecionando o índice de cada coluna do training set dos itens
labels = item_train_norm.columns.tolist()
# Selecting the current labels of each feature on the y-axis
# Selecionando os labels atuais de cada feature no eixo y
tick_labels = ax.get_yticklabels()

for i in range(len(tick_labels)):
    # The first label is the 'Sum of 4 other features
    # So we don't want to update it
    # O primeiro label é a 'Sum of 4 other features
    # Portanto, não queremos atualizá-la
    if i == 0:
        continue
    # Updating the rest of the labels with the corresponding feature label
    # Atualizando o restante dos labels com o label da feature correspondente
    idx = tick_labels[i].get_text().replace('Feature ', '')
    tick_labels[i] = labels[int(idx)]

# Defining new labels for each y-axis feature
# Definindo os novos labels de cada feature do eixo y
ax.set_yticklabels(tick_labels)
plt.show()

# Fitting the model (Treinando o modelo)
# Verbose 0 due to the number of epochs (Verbose 0 por conta da quantidade de epochs)
history = model.fit([user_train_norm, item_train_norm], y_train, epochs=300, verbose=0)

# Plotting the error and metric during model training (Plotando o erro e a métrica durante o treinamento do modelo)
plot_history(history)

# Evaluating the model on training and validation data (Avaliando o modelo nos dados de treino e de validação)
train_eval = model.evaluate([user_train_norm, item_train_norm], y_train, verbose=0)
cv_eval = model.evaluate([user_cv_norm, item_cv_norm], y_cv, verbose=0)

print(f'Train set evaluation: {train_eval[0]:.2f}\nValidation set evaluation: {cv_eval[0]:.2f}')

# Saving the pre-trained weights of the model (Salvando os pesos pré-treinados do modelo)
model.save_weights(os.path.join(PATH_M, 'pretrained.weights.h5'))
