import requests

url = "http://localhost:5000/health"
response = requests.get(url)
print(response.status_code)
print(response.json())