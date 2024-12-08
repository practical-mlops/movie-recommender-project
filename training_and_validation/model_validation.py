def validate_model(model_run_id, top_k=50, threshold=3, val_batch_size=32):
    # https://pureai.substack.com/p/recommender-systems-with-pytorch
    from collections import defaultdict
    import torch
    import mlflow.pytorch
    import mlflow
    from sklearn.metrics import mean_squared_error

    mlflow.set_tracking_uri(uri="http://192.168.1.104:8080")

    model_uri = f"runs:/{model_run_id}/model"
    recommendation_model = mlflow.pytorch.load_model(model_uri)

    def calculate_precision_recall(user_ratings, k, threshold):
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum(true_r >= threshold for _, true_r in user_ratings)
        n_rec_k = sum(est >= threshold for est, _ in user_ratings[:k])
        n_rel_and_rec_k = sum((true_r >= threshold) and (est >= threshold) for est, true_r in user_ratings[:k])

        precision = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
        recall = n_rel_and_rec_k / n_rel if n_rel != 0 else 1
        return precision, recall

    user_ratings_comparison = defaultdict(list)

    dataset_map = get_datasets_local(split=['val'])
    val_dataloader = DataLoader(dataset_map['val'], batch_size=val_batch_size, shuffle=True)

    y_pred = []
    y_true = []

    recommendation_model.eval()

    with torch.no_grad():
        for users, movies, ratings in val_dataloader:
            output = recommendation_model(users, movies)

            y_pred.append(output.sum().item() / len(users))
            y_true.append(ratings.sum().item() / len(users))

            for user, pred, true in zip(users, output, ratings):
                user_ratings_comparison[user.item()].append((pred[0].item(), true.item()))

    user_precisions = dict()
    user_based_recalls = dict()

    k = top_k

    for user_id, user_ratings in user_ratings_comparison.items():
        precision, recall = calculate_precision_recall(user_ratings, k, threshold)
        user_precisions[user_id] = precision
        user_based_recalls[user_id] = recall


    average_precision = sum(prec for prec in user_precisions.values()) / len(user_precisions)
    average_recall = sum(rec for rec in user_based_recalls.values()) / len(user_based_recalls)
    rms = mean_squared_error(y_true, y_pred, squared=False)

    print(f"precision_{k}: {average_precision:.4f}")
    print(f"recall_{k}: {average_recall:.4f}")
    print(f"rms: {rms:.4f}")
    mlflow.log_metric(f"precision_{k}", average_precision, run_id=model_run_id)
    mlflow.log_metric(f"recall_{k}", average_recall, run_id=model_run_id)
    mlflow.log_metric("rms", rms, run_id=model_run_id)