import bentoml
import mlflow
import torch
import numpy as np
from mlflow import MlflowClient

@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class RecommenderRunable:
    def __init__(self, registered_model_name='recommender_production', device='cpu'):
        mlflow.set_tracking_uri(uri="http://192.168.1.90:8080")
        client = MlflowClient()
        current_prod = client.get_model_version_by_alias(registered_model_name, "prod")
        model_uri = f"runs:/{current_prod.run_id}/model"
        print(model_uri)
        bentoml.mlflow.import_model("recommender", model_uri)
        bento_model = bentoml.mlflow.get("recommender:latest")
        mlflow_model_path = bento_model.path_of(bentoml.mlflow.MLFLOW_MODEL_FOLDER)

        self.model = mlflow.pytorch.load_model(mlflow_model_path)
        self.device = device
        self.model.to(self.device)
        self.model.eval()
    
    @bentoml.api
    def predict(self, user_id: int, top_k: int=10, ranked_movies:np.ndarray=None) -> np.ndarray:
        user_id = torch.tensor([user_id], dtype=torch.long).to(self.device)
        all_items = torch.arange(1, self.model.n_items + 1, dtype=torch.long).to(self.device)

        # Remove already ranked movies from the list of all items
        if ranked_movies is not None:
            ranked_movies = torch.tensor(ranked_movies, dtype=torch.long).to(self.device)
            unrated_items = all_items[~torch.isin(all_items, ranked_movies)]
        else:
            unrated_items = all_items
        
        user_ids = user_id.repeat(len(unrated_items))
        
        # Predict ratings for all unrated items
        with torch.no_grad():
            predictions = self.model(user_ids, unrated_items).squeeze()
        
        # Get the item with the highest predicted rating
        top_n_indices = torch.topk(predictions, top_k).indices
        recommended_items = unrated_items[top_n_indices].cpu().numpy()
        
        return recommended_items
