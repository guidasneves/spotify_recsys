import numpy as np
import pandas as pd
import tensorflow as tf


def get_user_dataset(user_vec, item_df):
    """
    [EN-US]
    Creates the User Dataset with the same number of examples as the dataset that was given as reference.
    
    [PT-BR]
    Cria o Dataset do usuário com a mesma quantidade de exemplos do que o dataset que foi dado como referência.
    
    Arguments:
        user_vec -- Row vector with user features
                    (Vetor de linha com as features do usuário).
        item_df -- Reference Dataset to create User Dataset
                   (Dataset de referência para criar o Dataset do usuário).
    
    Return:
        user_df -- User dataset with the same dimensions as the reference dataset
                   (Dataset do usuário com as mesmas dimensões do dataset de referência).
    """
    # Creating the user matrix (Criando a matriz do usuário).
    user_df = np.tile(user_vec, (len(item_df), 1))
    features = list(item_df.columns)
    
    # Transforming the numpy array into a pandas DataFrame (Transformando o array numpy em um DataFrame pandas).
    user_df = pd.DataFrame(user_df, columns=features)
    return user_df


def df_to_tfdataset(X, y, shuffle_buffer=1000, batch_size=32, shuffle=True):
    """
    [EN-US]
    Transforms a dataframe or numpy array into a tf.data.Dataset object, applies preprocessing and optimizes performance.
    
    [PT-BR]
    Transforma um dataframe ou matriz numpy em um objeto tf.data.Dataset, aplica os pré-processamentos e otimizan o desempenho.
    
    Arguments:
        X -- DataFrame or numpy array with features (DataFrame ou matriz numpy com as features).
        y -- DataFrame or numpy array with labels (DataFrame ou matriz numpy com os labels).
        shuffle_buffer -- Elements that will initially be left out and one of them is randomly chosen as part of the random dataset
                          (Elementos que serão inicialmente deixados de lado e um deles é escolhido aleatoriamente como parte do dataset aleatorio).
        batch_size -- Size of dataset mini-batches (Tamanho dos mini-batches do dataset).
        shuffle -- If True, the dataset will be shuffled, otherwise not (Caso seja True, o dataset será embaralhado, caso contrário, não).
    
    Return:
        dataset -- Preprocessed tf.data.Dataset (tf.data.Dataset pré-processado).
    """
    # Concatenating the dataset X with the labels y
    # Concatenando o dataset X com os labels y
    dataset = pd.concat([X, y], axis=1, ignore_index=True)
    # Transforming the concatenated dataset into a tf.data.Dataset object
    # Transformando o dataset concatenado em um objeto tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    # Storing elements in memory (Armazenando elementos na memória)
    dataset = dataset.cache()
    if shuffle:
        # Shuffling the dataset (Embaralhando o dataset)
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    # Applying the final preprocessing
    # Aplicando os pré-processamentos finais
    dataset = (
        dataset
        .map(lambda x: (x[:-1], x[-1])) # Separating features from labels into tuples (Separando as features dos labels em tuplas)
        .batch(batch_size) # Creating batches of this dataset (Criando batches desse dataset)
        .prefetch(buffer_size=tf.data.AUTOTUNE) # Allowing parallel execution of this dataset (Permitindo a execução paralela dessa dataset)
    )
    return dataset
