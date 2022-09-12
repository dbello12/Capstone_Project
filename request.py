import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Candidate_Observer_Multiplication_Mobilizer':2, 'Candidate_Observer_Multiplication_Discipler':9, 'Candidate_Observer_Multiplication_Server':6})

print(r.json())