import logging
import os.path
import sys
from datetime import datetime
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_absolute_error
from torch import nn
from torch_geometric.nn import global_mean_pool, MFConv, GATv2Conv
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

sys.path.append(os.path.abspath("."))
from Source.data import balanced_train_valid_split, root_mean_squared_error
from Source.models.GCNN.trainer import GCNNTrainer
from Source.models.GCNN.featurizers import featurize_sdf, ConvMolFeaturizer, featurize_df
from Source.models.GCNN_FCNN.model import GCNN
from config import ROOT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
time_mark = str(datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]

cv_folds = 5
seed = 23
batch_size = 64
epochs = 1000
es_patience = 100
mode = "regression"
#path_to_sdf = ROOT_DIR / "Test_data/GCNN_data/logS_logP.sdf"
path_to_df = ROOT_DIR / "Test_data/GCNN_data/smiles_mp.csv"
valuename = "MP" ## указываем таргет
output_folder = ROOT_DIR / f"Output/{cv_folds}fold_{mode}_{time_mark}"

targets = ({"name": valuename,
            "mode": "regression",
            "dim": 1,
            "metrics": {
                "R2": (r2_score, {}),
                "RMSE": (root_mean_squared_error, {}),
                "MAE": (mean_absolute_error, {})
            },
            "loss": nn.MSELoss(),
            },)

model_parameters = {
    "pre_fc_params": {
        "hidden": (),
        "dropout": 0,
        "actf": nn.LeakyReLU(),
    },
    "hidden_conv": (128, 64,),
    "conv_dropout": 0.27936243337975536,
    "conv_actf": nn.LeakyReLU(),
    "conv_layer": GATv2Conv,
    "conv_parameters": None,
    "graph_pooling": global_mean_pool,
    "post_fc_params": {
        "hidden": (64,),
        "dropout": 0,
        "use_bn": False,
        "actf": nn.LeakyReLU(),
    },
}

df = pd.read_csv('df.csv')
df = df.iloc[:1000, :]
print(df)

processed_dataset = featurize_df(
    df=df,
    mol_featurizer=ConvMolFeaturizer(),
    target=valuename,
    column_name='SMILES')

train_graphs, test_graphs = train_test_split(processed_dataset)

logging.info("Splitting...")
folds = balanced_train_valid_split([train_graphs], n_folds=cv_folds,
                                   batch_size=batch_size,
                                   shuffle_every_epoch=True,
                                   seed=seed)

test_loader = DataLoader(test_graphs, batch_size=batch_size)


model = GCNN(
    node_features=next(iter(test_loader)).x.shape[-1],
    targets=targets,
    **model_parameters,
    optimizer=torch.optim.Adam,
    optimizer_parameters=None,
)

trainer = GCNNTrainer(
    model=model,
    train_valid_data=folds,
    test_data=test_loader,
    output_folder=output_folder,
    epochs=epochs,
    es_patience=es_patience,
    targets=targets,
    seed=seed,
)

trainer.train_cv_models()
