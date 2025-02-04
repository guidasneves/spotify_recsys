import pandas as pd
import matplotlib.pyplot as plt
from random import randint


def get_pairs(df):
    """
    [EN-US]
    The input is a DataFrame and returns a list of unique pairs of features from that DataFrame.
    
    [PT-BR]
    A entrada é um DataFrame e retorna uma lista com pares únicos de features desse DataFrame.
    
    Argument:
        df -- DataFrame to randomly get 25 unique pairs of your features
              (DataFrame para pegar aleatóriamente 25 pares únicos de suas features).
    
    Returns:
        pairs -- List with unique pairs of DataFrame features
                 (Lista com pares únicos de features do DataFrame).
    """
    # Empty list to store tuples with pairs (Lista vazia para armazenar os as tuplas com os pares).
    pairs = []

    for i in range(25):
        # Selecting the first feature (Selecionando a primeira feature).
        x = df.columns[randint(0,12)]
        # Selecting the second feature (Selecionando a segunda feature).
        y = df.columns[randint(0,12)]
        # Loop to not select the 2nd feature repeated or equal to the first feature
        # (Loop para não selecionar a 2 feature repetida ou igual a primeira feature).
        while x == y or (x, y) in pairs or (y, x) in pairs:
            y = df.columns[randint(0,12)]
        pairs.append((x, y))
    
    return pairs


def plot_hist_vs(X):
    """
    [EN-US]
    Returns a histogram comparing the distribution of numeric features between the positive and negative class.
    
    [PT-BR]
    Retorna um histograma comparando a distribuição das features numéricas entre a classe positiva e negativa.
    
    Argument:
        X -- X matrix (Matriz X).
    """
    # Excluding categorical features (Excluindo as features categóricas).
    X = X.drop(columns=['name', 'artists', 'duration_ms', 'mode']).copy()
    # Selecting the positive class (Selecionando a classe positiva).
    label_1 = X[X['y'] == 1].drop(columns=['y']).copy()
    # Selecting the negative class (Selecionando a classe negativa).
    label_0 = X[X['y'] == 0].drop(columns=['y']).copy()

    # Creating the figure for the plot (Criando a figura para o plot).
    fig, axs = plt.subplots(4, 3, figsize=(12, 6))
    i = 0

    # Going through each line of the axis (Percorrendo cada linha do eixo).
    for rows in axs:
        # Going through each element of the line (Percorrendo cada elemento da linha),
        for ax in rows:
            # Creating the plot for each feature (Criando o plot para cada feature).
            title = label_1.columns[i].capitalize()
            ax.hist(label_1[label_1.columns[i]].to_list(), bins=100, color='b', label='Good')
            ax.hist(label_0[label_0.columns[i]].to_list(), bins=100, color='r', alpha=0.7, label='Bad')
            ax.set_title(title)
            i += 1
            if i == 12:
                break
    plt.legend(loc='best')
    plt.show()


def plot_history(history):
    """
    [EN-US]
    Plots the history of the model's training loss and metrics by epoch.
    
    [PT-BR]
    Plota o histórico da loss e da métrica do treinamento do model por epoch.

    Argument:
        history -- history returned by model training (histórico retornado pelo treino do modelo).
    """
    # Accessing the vector with the loss history (Acessando o vetor com o histórico da loss)
    loss = history.history['loss']
    # Accessing the vector with the metric history (Acessando o vetor com o histórico da métrica)
    metric = history.history['auc']
    # Selecting the number of epochs (Selecionando a quantidade de epochs)
    epochs = range(len(loss))
    # Defining the list with the values to plot (Definindo a lista com os valores para plotar)
    utils = [loss, 'loss'], [metric, 'auc']
    
    # Defining the figure and creating the plots (Definindo a figura e criando os plots)
    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    for i in range(2):
        fig.suptitle('Performance per Epoch')
        # Plotting with all epochs (Plotando com todas as epochs)
        ax[i, 0].plot(epochs, utils[i][0], color='g')
        ax[i, 0].set_ylabel(utils[i][1])
        ax[i, 0].set_xlabel('epochs')
    
        # Plotting only the final 25% of the epoch (PLotando apenas os 25% final da epoch) 
        ax[i, 1].plot(epochs, utils[i][0], color='g')
        ax[i, 1].set_xlim(int((len(utils[i][0]) * .75)), len(utils[i][0]))
        ax[i, 1].set_xlabel('epochs')
    plt.show()
