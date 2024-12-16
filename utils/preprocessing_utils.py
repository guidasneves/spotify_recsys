import numpy as np
import pandas as pd
import tensorflow as tf

def get_user_dataset(user_vec, item_df):
    """
    [EN-US]
    Creates the User Dataset with the same number of examples as the dataset that was given as reference.
    
    [PT-BR]
    Cria o Dataset do usuário com a mesma quantidade de exemplos do que o dataset que foi dado como referência.
    
    Argument:
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

class L2_Norm(tf.keras.Layer):
    """
    [EN-US]
    Transforms the L2 norm computation into a layer.

    [PT-BR]
    Transforma o cálculo da nomra L2 em uma layer.
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
        return tf.linalg.l2_normalize(X, axis=1)
