import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)




print('Dowloading Sony 10% Model ')
download_file_from_google_drive('1a17FrYVJY7pGTw4qxurfxALY6PB1nP0A', 'checkpoint/Sony/model4k10.ckpt.data-00000-of-00001')
download_file_from_google_drive('1zTjqOaEhyoEcmQ2ANfmWOrBJJ-eSMT7v', 'checkpoint/Sony/model4k10.ckpt.meta')

print('Dowloading Sony 10% ablation Model')
download_file_from_google_drive('1iPgaDgKLmXmIjc-tWjhD2St5mZrGzqc5', 'checkpoint/Sony/modelAblation.ckpt.data-00000-of-00001')
download_file_from_google_drive('1WDBRIMuXNF0FaJ_iH3r4NrqxH_osDsBu', 'checkpoint/Sony/modelAblation.ckpt.meta')

print('Dowloading Sony 10% loss Model (84Mb)')
download_file_from_google_drive('WHOz8qF4Q5zfOwLXmOt7nHmyyfDUnMUi', 'checkpoint/Sony/model_loss4k10.ckpt.data-00000-of-00001')
download_file_from_google_drive('1a0CB3GKaQBO0WJbJoAgB81XccNaThxTg', 'checkpoint/Sony/model_loss4k10.ckpt.meta')

