from kfp.dsl import component

@component(packages_to_install=["mlflow"])
def promote_model_to_staging(
    model_run_id, 
    registered_model_name='recommender_production',
    rms_threshold=0.0, 
    precision_threshold=-0.3, 
    recall_threshold=-0.2):

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

        if (new_model_metrics['precision_50'] - staging_model_metrics['precision_50']) < precision_threshold:
            return
        
        if (new_model_metrics['recall_50'] - staging_model_metrics['recall_50']) < recall_threshold:
            return

    result = mlflow.register_model(f"runs:/{model_run_id}/model", "recommender_production")
    client.set_registered_model_alias("recommender_production", "staging", result.version)