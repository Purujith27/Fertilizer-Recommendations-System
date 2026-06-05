import pickle
import pandas as pd

model = pickle.load(open("model.pkl", "rb"))

cases = pd.DataFrame([
    [30,60,40,1,2,20,15,10],
    [25,50,35,2,1,15,20,5],
    [35,70,55,0,3,25,10,20]
])

predictions = model.predict(cases)

print("="*50)
print("PREDICTION RESULTS")
print("="*50)

for i,pred in enumerate(predictions):
    print(f"Case {i+1}: {pred}")