
#from sklearn.model_selection import train_test_split
#import pandas as pd
#import os
#import numpy as np
#from torch.utils.data import Dataset
#import torch
#from torch.utils.data import DataLoader
from kfp.dsl import component

@component(packages_to_install=["torch", "torchvision", "torchaudio", "mlflow"], pip_index_urls=["https://download.pytorch.org/whl/cpu"])
def train_model(mlflow_experiment_name='recommender', mlflow_run_id=None, mlflow_tags={},
                hot_reload_model_run_id=None,
                model_embedding_factors=20, model_learning_rate=1e-3,model_hidden_dims=256, model_dropout_rate=0.2,
                optimizer_step_size=10, optimizer_gamma=0.1,
                training_epochs=30,
                train_batch_size=64, test_batch_size=64, shuffle_training_data=True, shuffle_testing_data=True):
    input_params = {}
    for k, v in locals().items():
        if k == 'input_params':
            continue
        input_params[k] = v
    import torch
    from torch.autograd import Variable
    from torch.utils.data import DataLoader
    import mlflow
    from torchinfo import summary
    from mlflow.models import infer_signature
    from torch.utils.data import Dataset

    class datasetReader(Dataset):
        def __init__(self, df, dataset_name):
            self.df = df
            self.name = dataset_name
            print(f"{self.name} : {self.df.shape[0]}")

        def __len__(self):
            return self.df.shape[0]

        def __getitem__(self, idx):
            sd = self.df.iloc[idx]
            user = sd['user_id']
            item = sd['item_id']
            rating = sd['rating']
            return torch.tensor(user-1).long(), torch.tensor(item-1).long(), torch.tensor(rating).float()
    
    class MatrixFactorization(torch.nn.Module):
        def __init__(self, n_users, n_items, n_factors, hidden_dim, dropout_rate):
            super().__init__()
            self.n_items = n_items
            self.user_factors = torch.nn.Embedding(n_users+1, 
                                               n_factors,
                                               sparse=False)
            self.item_factors = torch.nn.Embedding(n_items+1, 
                                               n_factors,
                                               sparse=False)
        
            self.linear = torch.nn.Linear(in_features=n_factors, out_features=hidden_dim)
            self.linear2 = torch.nn.Linear(in_features=hidden_dim, out_features=1)
            self.dropout = torch.nn.Dropout(p=dropout_rate)
            self.relu = torch.nn.ReLU()
        
        def forward(self, user, item):
            user_embedding = self.user_factors(user)
            item_embedding = self.item_factors(item)
            embeddding_vector = torch.mul(user_embedding, item_embedding)
            x = self.relu(self.linear(embeddding_vector))
            x = self.dropout(x)
            rating = self.linear2(x)
            return rating
    
    dataset_map = get_datasets_local(split=['train', 'test'])
    
    if hot_reload_model_run_id is not None:
        model_uri = f"runs:/{hot_reload_model_run_id}/model"
        model = mlflow.pytorch.load_model(model_uri)
    else:
        model = MatrixFactorization(dataset_map['n_users'], dataset_map['n_items'], n_factors=model_embedding_factors, hidden_dim=model_hidden_dims, dropout_rate=model_dropout_rate)

    optimizer = torch.optim.SGD(model.parameters(), lr=model_learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=optimizer_step_size, gamma=optimizer_gamma)
    loss_func = torch.nn.L1Loss()
    train_dataloader = DataLoader(dataset_map['train'], batch_size=train_batch_size, shuffle=shuffle_training_data)
    test_dataloader = DataLoader(dataset_map['test'], batch_size=test_batch_size, shuffle=shuffle_testing_data)

    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://192.168.1.104:8080")

    # Create a new MLflow Experiment
    mlflow.set_experiment(mlflow_experiment_name)

    with mlflow.start_run(run_id=mlflow_run_id):
        for k,v in input_params.items():
            if 'mlflow_' not in k:
                mlflow.log_param(k, v)
        mlflow.log_param("loss_function", loss_func.__class__.__name__)
        #mlflow.log_param("metric_function", metric_fn.__class__.__name__,
        mlflow.log_param("optimizer", "SGD")
        mlflow.log_params({'n_user': dataset_map['n_users'], 'n_items': dataset_map['n_items']})
    
        for k,v in mlflow_tags.items():
            mlflow.set_tag(k, v)

        with open("model_summary.txt", "w") as f:
            f.write(str(summary(model)))
        mlflow.log_artifact("model_summary.txt")

        model_signature = None

        for train_iter in range(training_epochs):
            print(train_iter)
            model.train()
            t_loss = 0
            t_count = 0
            for row, col, rating in train_dataloader:
                # Predict and calculate loss
                #try:
                prediction = model(row, col)
                if model_signature is None:
                    model_signature = infer_signature({'user': row.cpu().detach().numpy(), 'movie': col.cpu().detach().numpy()}, prediction.cpu().detach().numpy())

                #except Exception as e:
                #print(f"R:{row}, C:{col}")
                loss = loss_func(prediction, rating.unsqueeze(1))
                t_loss += loss
                t_count += 1

                # Backpropagate
                loss.backward()

                # Update the parameters
                optimizer.step()
                optimizer.zero_grad()
            mlflow.log_metric("avg_training_loss", f"{(t_loss/t_count):3f}", step=train_iter)
            scheduler.step()
            model.eval()
            te_loss = 0
            te_count = 0
            print('Evaluating')
            with torch.no_grad():
                #HR, NDCG = metrics(model, test_dataloader, 5)
                for row, col,rating in test_dataloader:
                    prediction = model(row, col)
                    loss = loss_func(prediction, rating.unsqueeze(1))
                    te_loss += loss
                    te_count += 1
            mlflow.log_metric("avg_testing_loss", f"{(te_loss/te_count):3f}", step=train_iter)
            #print(f"HR: {HR} NDCG:{NDCG}")
            print(f"Test loss: {te_loss/te_count}")
            print(f"Train loss: {t_loss/t_count}")

        mlflow.pytorch.log_model(model, "model", signature=model_signature)