import pickle

with open("encoded_models_new/mrpc/ff.pkl", "rb") as f:
    data = pickle.load(f)

weights = data["weights"]

print("Number of layers:", len(weights))

for name, layer in weights.items():
    print(name, "shape:", getattr(layer, "shape", "no shape"))