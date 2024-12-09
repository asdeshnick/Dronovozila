import requests

def send_request():
    url = 'http://192.168.0.5:5000'  # Замените <IP_АДРЕС_СЕРВЕРА> на настоящий IP адрес вашего сервера
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            print("Ответ от сервера:", response.json().get('message'))  # Выводим сообщение "Hi!"
        else:
            print("Ошибка при получении ответа:", response.status_code)
    except Exception as e:
        print("Возникла ошибка:", str(e))

if __name__ == "__main__":
    print(3563)
    send_request()
    
