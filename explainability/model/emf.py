import torch
import torch.nn as nn
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm
from .utils import EMFLoss


class EMFModel:

    def __init__(
        self,
        learning_rate: float,
        reg_term: float,
        expl_reg_term: float,
        positive_threshold: float,
        latent_dim: int,
        epochs: int,
        knn: int,
    ):

        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.dataset = None
        self.dataset_metadata = None
        self.embedding_user = None
        self.embedding_item = None
        self.optimizer = None

        self.reg_term = reg_term
        self.expl_reg_term = expl_reg_term
        self.positive_threshold = positive_threshold
        self.knn = knn

        self.explainability_matrix = None
        self.sim_users = {}

        self.affine_output = nn.Linear(in_features=self.latent_dim, out_features=1)

        self.criterion = EMFLoss()

    def fit(self, dataset_metadata):

        self.dataset_metadata = dataset_metadata
        self.dataset = dataset_metadata.dataset

        # Convert to 0-based indexing at the start
        self.dataset["userId"] = self.dataset["userId"] - 1
        self.dataset["itemId"] = self.dataset["itemId"] - 1

        num_users = self.dataset_metadata.num_user
        num_items = self.dataset_metadata.num_item

        self.embedding_user = np.random.uniform(
            low=0, high=0.5 / self.latent_dim, size=(num_users, self.latent_dim)
        )

        self.embedding_item = np.random.uniform(
            low=0, high=0.5 / self.latent_dim, size=(num_items, self.latent_dim)
        )

        self.compute_explainability()

        with tqdm(total=self.epochs) as progress:
            for epoch in range(self.epochs):
                self.dataset = self.dataset.sample(frac=1)
                loss = []
                for _, row in self.dataset.iterrows():
                    p_ui = self.predict(row.userId, row.itemId)

                    e_ui = row.rating - p_ui

                    loss.append(e_ui**2)

                    # u′i=ui+η·(2·(rij−ui·vTj)·vj−β·ui−λ·sgn(ui−vj)·Eij)
                    delta_u = 2 * e_ui * self.embedding_item[row.itemId, :]
                    delta_u -= self.reg_term * self.embedding_user[row.userId, :]
                    temp = np.sign(
                        self.embedding_item[row.itemId, :]
                        - self.embedding_user[row.userId, :]
                    )
                    temp *= (
                        self.expl_reg_term
                        * self.explainability_matrix[row.userId, row.itemId]
                    )
                    delta_u -= temp

                    # v′j=vj+η·(2·(rij−ui·vTj)·ui−β·vj−λ·sgn(ui−vj)·Eij)
                    delta_v = 2 * e_ui * self.embedding_user[row.userId, :]
                    delta_v -= self.reg_term * self.embedding_item[row.itemId, :]
                    temp = np.sign(
                        self.embedding_user[row.userId, :]
                        - self.embedding_item[row.itemId, :]
                    )
                    temp *= (
                        self.expl_reg_term
                        * self.explainability_matrix[row.userId, row.itemId]
                    )
                    delta_v -= temp

                    self.embedding_user[row.userId, :] += self.learning_rate * delta_u
                    self.embedding_item[row.itemId, :] += self.learning_rate * delta_v

                progress.update(1)

                progress.set_postfix({"MSE": sum(loss) / len(loss)})
        return True

    def compute_explainability(self):
        # Add validation for user IDs
        if self.dataset["userId"].max() >= self.dataset_metadata.num_user:
            print("Warning: Converting 1-based user IDs to 0-based indexing")
            self.dataset["userId"] = self.dataset["userId"] - 1

        ds = self.dataset.pivot(index="userId", columns="itemId", values="rating")
        ds = ds.fillna(0)
        ds = sparse.csr_matrix(ds)
        sim_matrix = cosine_similarity(ds)
        min_val = sim_matrix.min() - 1

        for i in range(self.dataset_metadata.num_user):
            sim_matrix[i, i] = min_val

            knn_to_user_i = (-sim_matrix[i, :]).argsort()[: self.knn]
            self.sim_users[i] = knn_to_user_i

        self.explainability_matrix = np.zeros(
            (self.dataset_metadata.num_user, self.dataset_metadata.num_item)
        )

        filter_dataset_on_threshold = self.dataset[
            self.dataset["rating"] >= self.positive_threshold
        ]

        for i in range(self.dataset_metadata.num_user):
            knn_to_user_i = self.sim_users[i]

            rated_items_by_sim_users = filter_dataset_on_threshold[
                filter_dataset_on_threshold["userId"].isin(knn_to_user_i)
            ]

            sim_scores = rated_items_by_sim_users.groupby(by="itemId")
            sim_scores = sim_scores["rating"].sum()
            sim_scores = sim_scores.reset_index()

            # Adjust both user and item indices to be 0-based
            self.explainability_matrix[i, sim_scores.itemId - 1] = (
                sim_scores.rating.to_list()
            )

        self.explainability_matrix = MinMaxScaler().fit_transform(
            self.explainability_matrix
        )

    def predict(self, user_id, item_id):
        # Validate indices
        if isinstance(user_id, list):
            user_id = [
                u if u < self.dataset_metadata.num_user else u - 1 for u in user_id
            ]
        else:
            user_id = (
                user_id if user_id < self.dataset_metadata.num_user else user_id - 1
            )

        if isinstance(item_id, list):
            item_id = [
                i if i < self.dataset_metadata.num_item else i - 1 for i in item_id
            ]
        else:
            item_id = (
                item_id if item_id < self.dataset_metadata.num_item else item_id - 1
            )

        if isinstance(user_id, list) or isinstance(item_id, list):
            if not isinstance(user_id, list):
                user_id = [user_id]
            if not isinstance(item_id, list):
                item_id = [item_id]

            pred = [
                np.dot(self.embedding_user[u, :], self.embedding_item[i, :])
                for i in item_id
                for u in user_id
            ]
        else:
            pred = np.dot(
                self.embedding_user[user_id, :], self.embedding_item[item_id, :]
            )
        return pred

    def user_embedding(self):
        return self.embedding_user

    def item_embedding(self):
        return self.embedding_item
