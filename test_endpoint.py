import requests

url = "http://localhost:5000/analyze-sentiment"
payload = {"text": "I hate this product!"}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers)
print(response.status_code)
print(response.json())