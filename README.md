# large-language-sommelier

## API для RAG-системы
### Запуск API для RAG-системы
```
python src/api/rag_api.py
```
### Пример использования
```
curl -X POST "http://localhost:8000/recommend/" -H "Content-Type: application/json" -d '{"question":"Посоветуйте красное сухое вино"}'
```

```
import requests
import json

BASE_URL = "http://localhost:8000"

endpoint = f"{BASE_URL}/recommend/"
payload = {
        "question": "Какое вино подойдет к рыбе?"
    }

try:
    response = requests.post(
        endpoint,
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    response.raise_for_status()
    
    result = response.json()
    print(result)

except requests.exceptions.RequestException as e:
    print(f"Error making request: {e}")
```

## TG bot
### Запуск
```
python app.py
```