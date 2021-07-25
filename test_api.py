import requests

url = "http://127.0.0.1:8000/uploadfile/"

payload={}
files=[
  ('file',('2.wav',open('C:/Users/dipesh.paul/Downloads/2.wav','rb'),'audio/wav'))
]
headers = {}

response = requests.request("POST", url, headers=headers, data=payload, files=files)

print(response.text)