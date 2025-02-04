from requests import post, get


def get_user(token_type, access_token):
    """
    [EN-US]
    Requests user data.
    
    [PT-BR]
    Requisita os dados do usuário.
    
    Arguments:
        token_type -- How the access token may be used: always "Bearer"
                      (Como o token de acesso pode ser utilizado: sempre “Bearer”).
        access_token -- An access token that can be provided in subsequent calls, for example to Spotify Web API services
                        (Um token de acesso que pode ser fornecido em chamadas subsequentes, por exemplo, para serviços Spotify Web API).
    
    Return:
        response.json()['id'] -- User ID (ID do usuário).
    """
    # Setting options for the request (Definindo as opções para a requisição)
    endpoint = f'https://api.spotify.com/v1/me'
    headers = {'headers': {
        'Authorization': token_type + ' ' + access_token
    }}
    
    # Send a get request (Enviando a requisição get)
    response = get(url=endpoint, headers=headers['headers'])
    # If the request status code is not 200 (Caso o status code da requisição não seja 200)
    if response.status_code != 200:
        print('Error! User data not extracted.')
    
    return response.json()['id']


def create_playlist(user_id, token_type, access_token):
    """
    [EN-US]
    Create a new playlist in the Spotify app.
    
    [PT-BR]
    Cria uma nova playlist no Spotify app.
    
    Arguments:
        user_id -- User ID where the playlist will be created
                    (ID do usuário onde será criada a playlist).
        token_type -- How the access token may be used: always "Bearer"
                      (Como o token de acesso pode ser utilizado: sempre “Bearer”).
        access_token -- An access token that can be provided in subsequent calls, for example to Spotify Web API services
                        (Um token de acesso que pode ser fornecido em chamadas subsequentes, por exemplo, para serviços Spotify Web API).
    
    Return:
        response.json()['id'] -- New playlist ID (ID da nova playlist).
    """
    # Setting options for the request (Definindo as opções para a requisição)
    url = f'https://api.spotify.com/v1/users/{user_id}/playlists'
    playlist_options = {
        'headers': {
            'Authorization': token_type + ' ' + access_token,
            'Content-Type': 'application/json'
        },
        'data': {
            'name': 'RecSys',
            'description': 'Recommended tracks',
            'public': True,
            'collaborative': False
        }
    }
    
    # Send a post request (Enviando a requisição post)
    request = post(url=url, headers=playlist_options['headers'], data=playlist_options['data'])
    # If the request status code is 200 (Caso o status code da requisição for 200)
    if request.status_code == 200:
        print('Playlist created successfully!')
    else:
        print('Playlist creation not authorized!')

    return request.json()['id']


def add_tracks(playlist_id, uris, token_type, access_token, position=0):
    """
    [EN-US]
    Adds recommended songs to a playlist.
    
    [PT-BR]
    Adiciona as músicas recomendadas em uma playlist.
    
    Arguments:
        playlist_id -- ID of the playlist where the songs will be added
                    (ID da playlist onde as músicas serão adicinadas).
        uris -- A list of Spotify URIs to add, can be track or episode URIs
                (Uma lista de URIs do Spotify para adicionar, podem ser URIs de track ou episódio).
        token_type -- How the access token may be used: always "Bearer"
                      (Como o token de acesso pode ser utilizado: sempre “Bearer”).
        access_token -- An access token that can be provided in subsequent calls, for example to Spotify Web API services
                        (Um token de acesso que pode ser fornecido em chamadas subsequentes, por exemplo, para serviços Spotify Web API).
        position -- The position to insert the items, a zero-based index
                    (A posição para inserir os itens, um índice de base zero).
    """
    # Setting options for the request (Definindo as opções para a requisição)
    url = f'https://api.spotify.com/v1/playlists/{playlist_id}/tracks'
    playlist_options = {
        'headers': {
            'Authorization': token_type + ' ' + access_token,
            'Content-Type': 'application/json'
        },
        'data': {
            'uris': uris,
            'position': position
        }
    }
    
    # Send a post request (Enviando a requisição post)
    request = post(url=url, headers=playlist_options['headers'], data=playlist_options['data'])
    # If the request status code is 200 (Caso o status code da requisição for 200)
    if request.status_code == 200:
        print('Tracks added to playlist successfully!')
    else:
        print('Error! Tracks not added to playlist.')
