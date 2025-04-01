import requests

url = "http://localhost:5000/rate-limit-status"
response = requests.get(url)
print(response.status_code)
print(response.json()) #