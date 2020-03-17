import pickle

def save_model(path, model):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
            return model

    except FileNotFoundError:
        raise FileNotFoundError("model name %s not found" % path)