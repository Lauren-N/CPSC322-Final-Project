import requests # a lib for HTTP requests
import json # a lib for parsing strings/JSON objects

url = "https://flask-app-demo.onrender.com/predict?"
# url = "http://127.0.0.1:5001/predict?"
# query terms
url += "level=Junior&lang=Java&tweets=yes&phd=no"
print(url)

# make the GET request
# https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods
response = requests.get(url)
# first check the status code
print("status code:", response.status_code)
# https://developer.mozilla.org/en-US/docs/Web/HTTP/Status#successful_responses
if response.status_code == 200:
    # OK
    # parse the message body JSON
    json_obj = json.loads(response.text)
    print(type(json_obj))
    print(json_obj)