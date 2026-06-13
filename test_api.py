import requests

url = "http://127.0.0.1:5000/predict"

image_path = "test.jpeg"  # change to your image file name

with open(image_path, "rb") as img:
    files = {"image": img}
    response = requests.post(url, files=files)

print(response.json())