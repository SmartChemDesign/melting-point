import numpy as np
from megnet.models import MEGNetModel
from megnet.data.crystal import CrystalGraph 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import pickle
from sklearn.model_selection import train_test_split

NEW_SEED = 333

np.random.seed(NEW_SEED)

def load_megnet_dataset(pkl_path, test_size=0.2, seed=NEW_SEED):
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    structures = [d['structure'] for d in data]
    targets = np.array([d['melting_point'] for d in data])
    
    X_train, X_test, y_train, y_test = train_test_split(
        structures, targets, test_size=test_size, random_state=seed
    )
    
    print(f"Dataset: train={len(X_train)}, test={len(X_test)}")
    return X_train, X_test, y_train, y_test

NFEAT_BOND = 10
R_CUTOFF = 5.0 
GAUSSIAN_WIDTH = 0.5

graph_converter = CrystalGraph(cutoff=R_CUTOFF)


centers = np.linspace(0, R_CUTOFF + 1, NFEAT_BOND)


model = MEGNetModel(
    graph_converter=graph_converter,
    centers=centers,
    width=GAUSSIAN_WIDTH,
    lr=1e-3,
    batch_size=1,
    loss='mse'
)

X_train, X_test, y_train, y_test = load_megnet_dataset('m3gnet_dataset.pkl')

history = model.train(X_train, y_train, epochs=50, verbose=1)

print("\n final estimation ...")
y_pred = []

for struct in X_test:
    pred = model.predict_structure(struct)
    y_pred.append(float(np.squeeze(pred)))

y_pred = np.array(y_pred)
# print("PREDS:", y_pred)
# print()
# print("TESTS:", y_test)
# print()

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Test MAE:  {mae:.2f} K")
print(f"Test RMSE: {rmse:.2f} K")
print(f"Test R²:   {r2:.4f}")


