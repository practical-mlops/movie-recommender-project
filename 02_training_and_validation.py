import kfp.dsl as dsl
import kfp

from data_components import qa_data
from training_and_validation import (
    negative_sampling, get_datasets,
    promote_model_to_staging,
    validate_model,
    train_model
)

@dsl.pipeline(
  name='Model training pipeline',
  description='A pipeline to train models on the movielens dataset for recommenders'
)
def training_pipeline(
    minio_bucket:str='datasets',
    training_dataset_name:str = 'ml-25m',
    training_batch_size: int = 64,
    training_learning_rate:float = 0.001,
    model_embedding_factors: int = 20,
    model_hidden_dims:int = 256,
    training_epochs:int = 30,
    optimizer_step_size: float= 10.0,
    optimizer_gamma: float = 0.1,
    model_dropout_rate:float = 0.2,
    testing_batch_size: int = 64,
    shuffle_training_data:bool =True,
    shuffle_testing_data:bool =True,
    hot_reload_model_id: str | None = None,
    validation_top_k:int = 50,
    validation_threshold:int = 3,
    validation_batch_size: int = 32,
    model_promote_rms_threshold: float = 0.0,
    model_promote_precision_threshold: float = -0.3,
    model_promote_recall_threshold:float = -0.2,
    mlflow_experiment_name: str = 'recommender',
    mlflow_registered_model_name: str = 'recommender_production'):
    
    qa_op = qa_data(bucket=minio_bucket)
    
    train_data = get_datasets(
                    bucket=minio_bucket,
                    dataset=training_dataset_name,
                    split='train').after(qa_op)
    
    test_data = get_datasets(
                    bucket=minio_bucket,
                    dataset=training_dataset_name,
                    split='test').after(qa_op)
    
    val_data =  get_datasets(
                    bucket=minio_bucket,
                    dataset=training_dataset_name,
                    split='val').after(qa_op)
    
    negative_sample_training = negative_sampling(
                    num_ng_test=10,
                    input=train_data.output).after(train_data)
    
    training = train_model(
        mlflow_experiment_name=mlflow_experiment_name,
        mlflow_run_id=None,
        mlflow_tags={},
        hot_reload_model_run_id=hot_reload_model_id,
        model_embedding_factors=model_embedding_factors,
        model_learning_rate=training_learning_rate,
        model_hidden_dims=model_hidden_dims,
        model_dropout_rate=model_dropout_rate,
        optimizer_step_size=optimizer_step_size,
        optimizer_gamma=optimizer_gamma,
        training_epochs=training_epochs,
        train_batch_size=training_batch_size,
        test_batch_size=testing_batch_size,
        training_data=negative_sample_training.output,
        testing_data=test_data.output,
        shuffle_training_data=shuffle_training_data,
        shuffle_testing_data=shuffle_testing_data).after(negative_sample_training)
    
    val = validate_model(
        model_run_id=training.output,
        top_k=validation_top_k,
        threshold=validation_threshold,
        val_batch_size=validation_batch_size,
        validation_dataset=val_data.output
    ).after(training)
    
    promote_model_to_staging(
        model_run_id=training.output,
        registered_model_name=mlflow_registered_model_name,
        rms_threshold=model_promote_rms_threshold,
        precision_threshold=model_promote_precision_threshold,
        recall_threshold=model_promote_recall_threshold).after(val)

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path='compiled_pipelines/training_pipeline.yaml')