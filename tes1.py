import pickle

with open("model.p", "rb") as f:
    model_dict = pickle.load(f)

model = model_dict["model"]

print("Classes:")
print(model.classes_)