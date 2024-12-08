from kfp.dsl import component

@component(packages_to_install=["mlflow"])
def promote_model_to_staging(
    model_run_id: str, 
    registered_model_name: str,
    rms_threshold: float, 
    precision_threshold: float,
    top_k: int,
    recall_threshold: float):

    import mlflow.pytorch
    import mlflow
    from mlflow import MlflowClient
    from mlflow.exceptions import RestException

    mlflow.set_tracking_uri(uri="http://192.168.1.104:8080")
    client = MlflowClient()

    current_staging = None
    try:
        current_staging = client.get_model_version_by_alias(registered_model_name, "staging")
    except RestException:
        print("No staging model found. Auto upgrade current run to staging.")
    
    if current_staging.run_id == model_run_id:
        print("Input run is already the current staging.")
        return
    
    if current_staging is not None:
        current_staging_model_data = client.get_run(current_staging.run_id).data.to_dictionary()
        staging_model_metrics = current_staging_model_data['metrics']

        new_model_data = client.get_run(model_run_id).data.to_dictionary()
        new_model_metrics = new_model_data['metrics']

        if (new_model_metrics['rms'] - staging_model_metrics['rms']) > rms_threshold:
            return

        if (new_model_metrics[f'precision_{top_k}'] - staging_model_metrics[f'precision_{top_k}']) < precision_threshold:
            return
        
        if (new_model_metrics[f'recall_{top_k}'] - staging_model_metrics[f'recall_{top_k}']) < recall_threshold:
            return

    result = mlflow.register_model(f"runs:/{model_run_id}/model", "recommender_production")
    client.set_registered_model_alias("recommender_production", "staging", result.version)