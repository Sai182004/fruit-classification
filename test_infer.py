# test_infer.py
import os, requests, json

API = os.environ.get("API_URL", "http://127.0.0.1:7860/api/predict")
DIR = "samples"  # put test images here

if not os.path.exists(DIR):
    print("Create samples/ and add images to test.")
    exit(0)

for fname in os.listdir(DIR):
    if not fname.lower().endswith((".jpg",".jpeg",".png")): continue
    path = os.path.join(DIR, fname)
    with open(path, "rb") as fh:
        resp = requests.post(API, files={"file": (fname, fh, "image/jpeg")}, data={"top_k":"3"})
    try:
        print(fname, resp.status_code, json.dumps(resp.json(), indent=2))
    except Exception:
        print(fname, resp.status_code, resp.text)
