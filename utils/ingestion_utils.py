from requests import post, get
from base64 import b64encode
import pandas as pd


def request_auth(client_id, client_secret):
    """
    [EN-US]
    The first step is to send a POST request to the /api/token endpoint of the Spotify OAuth 2.0 Service.
    If everything goes well, you'll receive a response with a 200 OK status and the JSON data.
    
    [PT-BR]
    A primeira etapa é enviar uma requisição POST para o endpoint /api/token do serviço Spotify OAuth 2.0.
    Se tudo correr bem, você receberá uma resposta com o status 200 OK e os dados JSON.
    
    Arguments:
        client_id -- The client ID generated after registering your application
                     (O cliente ID gerado após registrar seu aplicativo).
        client_secret -- the client secret generated after registering your application
                         (O client secret gerado após o registro do seu aplicativo).
        
    Returns:
        access_token -- An access token that can be provided in subsequent calls, for example to Spotify Web API services
                        (Um token de acesso que pode ser fornecido em chamadas subsequentes, por exemplo, para serviços Spotify Web API).
        token_type -- How the access token may be used: always "Bearer"
                      (Como o token de acesso pode ser utilizado: sempre “Bearer”).
        token_expires -- The time period (in seconds) for which the access token is valid
                         (O período de tempo (em segundos) durante o qual o token de acesso é válido).
    """
    # Base 64 encoded string that contains the client ID and client secret key (String codificada em base 64 que contém o clinet ID e a client secret key)
    b64_encoded = b64encode(f'{client_id}:{client_secret}'.encode())
    auth_b64 = str(b64_encoded, 'utf-8')
    # Setting options for authorization (Definindo as opções para a autorização)
    base_url = 'https://accounts.spotify.com/api/token'    
    
    auth_options = {
        'url': base_url,
        'headers':{
            'Authorization': 'Basic ' + auth_b64,
            'content-type': 'application/x-www-form-urlencoded'
        },
        'form':{
            'grant_type': 'client_credentials'
        },
        'json': True
    }

    # Send a post request (Enviando a requisição post)
    request = post(url=auth_options['url'], headers=auth_options['headers'], data=auth_options['form'])
    # If the request status code is 200 (Caso o status code da requisição for 200)
    if request.status_code == 200:
        json_request = request.json()
        access_token = json_request['access_token']
        token_type = json_request['token_type']
        token_expires = json_request['expires_in']
        print('Acesso autorizado e dados extraídos!')
    else:
        print('Acesso não autorizado!')
        
    return access_token, token_type, token_expires


def get_playlist(playlist, token_type, access_token, offset=0, limit=100):
    """
    [EN-US]
    Get full details of the items of a playlist owned by a Spotify user.
    
    [PT-BR]
    Obtenha detalhes completos dos itens de uma playlist de propriedade de um usuário do Spotify.
    
    Arguments:
        playlist -- The Spotify URL of the playlist
                    (O URL do Spotify da playlist).
        token_type -- How the access token may be used: always "Bearer"
                      (Como o token de acesso pode ser utilizado: sempre “Bearer”).
        access_token -- An access token that can be provided in subsequent calls, for example to Spotify Web API services
                        (Um token de acesso que pode ser fornecido em chamadas subsequentes, por exemplo, para serviços Spotify Web API).
        offset -- The index of the first item to return. Default: 0 (the first item). Use with limit to get the next set of items
                  (O índice do primeiro item a ser retornado. Padrão: 0 (o primeiro item). Use com limit para obter o próximo conjunto de itens).
        limit -- The maximum number of items to return. Default: 100. Minimum: 1. Maximum: 100
                 (O número máximo de itens a serem retornados. Padrão: 100. Mínimo: 1. Máximo: 100).
    
    Return:
        response.json() -- Pages of tracks in JSON (Páginas de tracks em JSON).
    """
    # Turning the Spotify playlist URL into just the Playlist ID to send the get request (Transformando o Spotify URL da playlist apenas no Playlist ID para enviar a requisição get)
    playlist_id = playlist.split('/')[-1].split('?')[0]
    # Setting options for the request (Definindo as opções para a requisição)
    endpoint = f'https://api.spotify.com/v1/playlists/{playlist_id}/tracks?offset={str(offset)}&limit={str(limit)}'
    headers={
        'Authorization': token_type + ' ' + access_token
    }

    # Send a get request (Enviando a requisição get)
    response = get(url=endpoint, headers=headers)
    # If the request status code is not 200 (Caso o status code da requisição não seja 200)
    if response.status_code != 200:
        print('Error! Playlist data not extracted.')
    
    return response.json()


def track_features(track_id, token_type, access_token):
    """
    Deprecated
    [EN-US]
    Get audio feature information for a single track identified by its unique Spotify ID.
    
    [PT-BR]
    Obtenha informações sobre features de áudio de uma única faixa identificada por seu ID exclusivo do Spotify.
    
    Arguments:
        track_id -- The Spotify ID for the track (O ID do Spotify para a track).
        token_type -- How the access token may be used: always "Bearer"
                      (Como o token de acesso pode ser utilizado: sempre “Bearer”).
        access_token -- An access token that can be provided in subsequent calls, for example to Spotify Web API services
                        (Um token de acesso que pode ser fornecido em chamadas subsequentes, por exemplo, para serviços Spotify Web API).
    
    Return:
        response.json() -- Audio features for one track in JSON
                           (Features de áudio para uma track em JSON).
    """
    # Setting options for the request (Definindo as opções para a requisição)
    endpoint = f'https://api.spotify.com/v1/audio-features/{track_id}'
    headers={
        'Authorization': token_type + ' ' + access_token
    }
    
    # Send a get request (Enviando a requisição get)
    response = get(url=endpoint, headers=headers)
    # If the request status code is not 200 (Caso o status code da requisição não seja 200)
    if response.status_code != 200:
        print('Error! Track data not extracted.')
    
    return response.json()    


def playlist_to_dataframe(playlist_endpoint, token_type, access_token, offset=0, limit=100, label=None):
    """
    [EN-US]
    Input's a playlist URL and returns a pandas DataFrame.
    
    [PT-BR]
    A entrada é um URL de uma playlist e retorna um DataFrame do pandas.
    
    Arguments:
        playlist_endpoint -- The Spotify URL of the playlist (O URL do Spotify da playlist).
        token_type -- How the access token may be used: always "Bearer"
                      (Como o token de acesso pode ser utilizado: sempre “Bearer”).
        access_token -- An access token that can be provided in subsequent calls, for example to Spotify Web API services
                        (Um token de acesso que pode ser fornecido em chamadas subsequentes, por exemplo, para serviços Spotify Web API).
        offset -- The index of the first item to return. Default: 0 (the first item). Use with limit to get the next set of items
                  (O índice do primeiro item a ser retornado. Padrão: 0 (o primeiro item). Use com limit para obter o próximo conjunto de itens).
        limit -- The maximum number of items to return. Default: 100. Minimum: 1. Maximum: 100
                 (O número máximo de itens a serem retornados. Padrão: 100. Mínimo: 1. Máximo: 100).
        label -- The label that the playlist will be classified by. Default: None. Label 1 for good playlist and 0 for bad playlist
                 (O label que a playlist será classificada. Padrão: None. Label 1 para playlist boa e 0 para playlist ruim).
        
    Return:
        df -- Pandas DataFrame with all tracks and the features of each track in the playlist
              (DataFrame pandas com todas as tracks e as features de cada track da playlist).
    """
    # Final list that will be transformed into DataFrame (Lista final que será transformada em DataFrame)
    examples = []
    # Tracks in JSON (Tracks em JSON)
    playlist = get_playlist(playlist=playlist_endpoint, token_type=token_type, access_token=access_token, offset=offset, limit=limit)
    
    # For loop through all playlist pages (Percorrendo todas as páginas da playlist)
    for page in range(offset, playlist['total'] + 1, limit):
        # Example size on page (Tamanho de exemplo na página)
        m = len(playlist['items'])
        # For loop through each example and adding the data to the final list (Percorrendo cada exemplo e adicionando os dados à lista final)
        for i in range(m):
            # Accessing each track (Acessando cada track)
            track = playlist['items'][i]['track']
            # Track features in JSON (Features da track em JSON)
            track_feature = track_features(track['id'], token_type, access_token) # Deprecated


            # Intermediate list for append each m example (Lista intermediária para adicionar cada exemplo m)
            example = [
                track['id'],
                track['name'],
                track['artists'][0]['name'],
                track['duration_ms'],
                track['popularity'],
                track_feature['danceability'],
                track_feature['energy'],
                track_feature['key'],
                track_feature['loudness'],
                track_feature['mode'],
                track_feature['speechiness'],
                track_feature['acousticness'],
                track_feature['instrumentalness'],
                track_feature['liveness'],
                track_feature['valence'],
                track_feature['tempo']
            ]

            # Adding to final list (Adicionando à lista final)
            examples.append(example)
        # If the page has more than 100 examples, you will read the next page
        # (Se a página tiver mais de 100 exemplos, lerá a próxima página)
        if m >= 100:
            # JSON tracks of the next offset (Tracks em JSON do próximo offset)
            playlist = get_playlist(playlist=playlist_endpoint, token_type=token_type, access_token=access_token, offset=page + limit, limit=limit)
    
    # DataFrame's features names (Nomes das features do DataFrame)
    features = ['id', 'name', 'artists', 'duration_ms', 'popularity', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
               'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    # Transforming into DataFrame (Transformando em DataFrame)
    df = pd.DataFrame(examples, columns=features)
    
    # Setting the labels (Definindo os labels)
    if label == 0:
        df['y'] = 0
    elif label == 1:
        df['y'] = 1
    
    return df
