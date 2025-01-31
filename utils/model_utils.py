from tensorflow.keras.layers import Dense, BatchNormalization, Input, Dot, Layer
from tensorflow.keras.models import Sequential
from tensorflow.keras import Layer, Model
from tensorflow.linalg import l2_normalize
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.backend import clear_session
import numpy as np

class L2_Norm(Layer):
    """
    [EN-US]
    Transforms the L2 norm computation into a layer.

    [PT-BR]
    Transforma o cálculo da norma L2 em uma layer.
    """
    def call(self, X):
        """
        [EN-US]
        Computes the L2 norm in the matrix.
    
        [PT-BR]
        Calcula a norma L2 na matriz X.
        
        Argument:
            X -- X matrix (Matriz X).
        
        Return:
            tf.linalg.l2_normalize -- The L2 norm that will be calculated in the layer.
                                      (A nroma L2 que será calculada na layer.)
        """
        # Returning the calculation that will be performed (Retornando o cálculo que será realizado).
        return l2_normalize(X, axis=1)

def user_nn(units=[256, 128, 64], num_outputs=32, lambda_r=1e-3):
    """
    [EN-US]
    Creates the user's neural network.
    
    [PT-BR]
    Cria a rede neural do usuário.
    
    Arguments:
        units -- Vector with the number of neurons in each layer, there are 3 hidden dense layers in total
                 (Vetor com a quantidade de neurônios em cada layers, são 3 camadas densas ocultas no total).
        num_outputs -- Number of units in the output layer (Número de neurônios da output layer).
        lambda_r -- Value of the layer regularization hyperparameter, lambda
                    (Valor do hiperparâmetro de regularização da layer, lambda).

    Return:
        user_NN -- User neural network (Rede neural do usuário).
    """
    # Defining the user's neural network, with 3 dense hidden layers, and 1 output dense layer. Between them we are calculating the batch norm, with the BatchNormalization() layer
    # Definindo a rede neural do usuário, com 3 dense hidden layers, e 1 output dense layer. Entre elas estamos calculando a batch norm, com a layer BatchNormalization()
    user_NN = Sequential([
        Dense(units=units[0], activation='relu', kernel_regularizer=l2(lambda_r), name='user_l1'),
        BatchNormalization(),
        Dense(units=units[1], activation='relu', kernel_regularizer=l2(lambda_r), name='user_l2'),
        BatchNormalization(),
        Dense(units=units[2], activation='relu', kernel_regularizer=l2(lambda_r), name='user_l3'),
        BatchNormalization(),
        Dense(units=num_outputs, activation='linear', name='user_output')
    ], name='user_NN')
    return user_NN

def item_nn(units=[256, 128, 64], num_outputs=32, lambda_r=1e-3):
    """
    [EN-US]
    Creates the item's neural network.
    
    [PT-BR]
    Cria a rede neural do item.
    
    Arguments:
        units -- Vector with the number of neurons in each layer, there are 3 hidden dense layers in total
                 (Vetor com a quantidade de neurônios em cada layers, são 3 camadas densas ocultas no total).
        num_outputs -- Number of units in the output layer (Número de neurônios da output layer).
        lambda_r -- Value of the layer regularization hyperparameter, lambda
                    (Valor do hiperparâmetro de regularização da layer, lambda).

    Return:
        item_NN -- Item neural network (Rede neural do item).
    """
    # Defining the item's neural network, with 3 dense hidden layers, and 1 output dense layer. Between them we are calculating the batch norm, with the BatchNormalization() layer
    # Definindo a rede neural do item, com 3 dense hidden layers, e 1 output dense layer. Entre elas estamos calculando a batch norm, com a layer BatchNormalization()
    item_NN = Sequential([
        Dense(units=units[0], activation='relu', kernel_regularizer=l2(lambda_r), name='item_l1'),
        BatchNormalization(),
        Dense(units=units[1], activation='relu', kernel_regularizer=l2(lambda_r), name='item_l2'),
        BatchNormalization(),
        Dense(units=units[2], activation='relu', kernel_regularizer=l2(lambda_r), name='item_l3'),
        BatchNormalization(),
        Dense(units=num_outputs, activation='linear', name='item_output')
    ], name='item_NN')
    return item_NN

def model_compile(
    optimizer,
    loss, 
    num_user_features, 
    num_item_features,
    item_similarity=False,
    user_units=[256, 126, 64],
    item_units=[256, 126, 64],
    num_outputs=32,
    lambda_r=1e-3
):
    """
    [EN-US]
    Compiles the model.
    
    [PT-BR]
    Compila o modelo.

    Arguments:
        optimizer -- optimizers metrics (Métrica de otimização).
        loss -- Loss function (Função de perca).
        num_user_features -- Number of features in the user dataset (Número de features do dataset do usuário).
        num_item_features -- Number of features in the item dataset (Número de features do dataset do item).
        item_similarity -- If set to True, it will compile only the item's neural network for similarity calculation, otherwise it will compile the complete network.
                           (Se for definida como True, compilará apenas a rede neural do item para o calculo da similaridade, caso contrário, compilará a rede completa).
        user_units -- Vector with the number of neurons in each layer for the user net, there are 3 hidden dense layers in total
                      (Vetor com a quantidade de neurônios em cada layers para a rede do usuário, são 3 camadas densas ocultas no total).
        item_units -- Vector with the number of neurons in each layer for the item net, there are 3 hidden dense layers in total
                      (Vetor com a quantidade de neurônios em cada layers para a rede do item, são 3 camadas densas ocultas no total).
        num_outputs -- Number of units in the output layer (Número de neurônios da output layer).
        lambda_r -- Value of the layer regularization hyperparameter, lambda
                    (Valor do hiperparâmetro de regularização da layer, lambda).

    Return:
        model -- Compiled model (Modelo compilado).
    """
    # Clearing all internal variables (Limpando todas as variáveis internas)
    clear_session()
    
    # Setting the user and item neural network (Definindo a rede neural do usuário e do item)
    user_NN = user_nn(
        user_units,
        num_outputs,
        lambda_r
    )
    item_NN = item_nn(
        item_units,
        num_outputs,
        lambda_r
    )
    
    # Setting the user network input shape
    # Definindo o input shape da rede do usuário
    input_user = Input(shape=(num_user_features,))
    # The user network will receive the output from the Input layer
    # A rede do usuário receberá o output da Input layer
    vu = user_NN(input_user)
    # The l2 norm layer will receive the output from the user network
    # A layer da norma l2 receberá o output da rede do usuário
    vu = L2_Norm()(vu)

    # Setting the item network input shape
    # Definindo o input shape da rede do item
    input_item = Input(shape=(num_item_features,))
    # The item network will receive the output from the Input layer
    # A rede do item receberá o output da Input layer
    vr = item_NN(input_item)
    # The l2 norm layer will receive the output from the item network
    # A layer da norma l2 receberá o output da rede do item
    vr = L2_Norm()(vr)

    # Computing the dot product between the user vector vu and the item vector vi
    # Calculando o dot product entre o vetor do usuário vu e o vetor de itens vi
    output = Dot(axes=1)([vu, vr])

    # Defining the model depending on the parameter value
    # Definindo o modelo dependendo do valor do parâmetro
    if item_similarity:
        # Model with only the item network (Modelo apenas com a rede do item)
        model = Model(input_item, vr)
    else:
        # Complete model, with user and item network (Modelo completo, com a rede do usuário e do item)
        model = Model([input_user, input_item], output)

    # Setting the optimizer and the loss function for the model
    # Definindo o otimizador e a função de perda para o modelo
    opt = optimizer
    cost = loss
    # Compiling the model (Compilando o modelo)
    model.compile(loss=loss, optimizer=opt, metrics=['auc'])

    return model

def hiperparams_tune(hiperparams):
    """
    [EN-US]
    Setting the model for hyperparameter optimization.

    [PT-BR]
    Define o modelo para a otimização dos hiperparâmetros.

    Argument:
        hyperparams -- List of hyperparameter value ranges to be optimized
                       (Lista com as faixas de valores dos hiperparâmetros para serem otimizados).

    Return:
        -acc -- Performance metric times negative (Métrica de avaliação vezes negativo). 
    """
    # Setting the threshold to define the probabilities for the classes
    # Definindo o threshold para definir as probabilidades para as classes
    threshold = .5
    # Setting the hyperparameters
    # Definindo os hiperparâmetros
    LR = hiperparams[0]
    lambda_r = hiperparams[1]
    # Setting the loss and the optimizer
    # Definindo a loss e o otimizador
    OPT = Adam(learning_rate=LR)
    LOSS = MeanSquaredError()

    # Defining the model to perform the optimization
    # Definindo o modelo para performar a otimização
    model = model_compile(OPT, LOSS, num_user_features, num_item_features, lambda_r=lambda_r)
    model.fit([user_opt, X_opt], y, epochs=200, verbose=0)

    # Computing prediction and performance
    # Calculando a previsão e o desempenho
    acc = model.predict([user_opt, X_opt], verbose=0)
    acc = np.where(acc >= threshold, 1, 0)
    acc = np.mean(acc == y)

    # -acc, because we want the pair of hyperparameters that minimizes the metric
    # -acc, porque queremos o par de hiperparâmetros que minimize a métrica
    return -acc
