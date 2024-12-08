import kfp.dsl as dsl
import kfp

from data_components import qa_data

@dsl.pipeline(
  name='Model training pipeline',
  description='A pipeline to train models on the movielens dataset for recommenders'
)
def training_pipeline(
    minio_bucket:str='datasets',
    trainig_batch_size: int = 1,
    training_learning_rate:float = 0.001,
    training_factors: int = 20,
    optimizer_step_size: float= 25.0,
    optimizer_gamma: float = 0.1):
    
    qa_op = qa_data(bucket=minio_bucket)
    train_data = get_datasets(
                    bucket='datasets',
                    dataset='ml-25m',
                    split='train').after(qa_op)
    test_data = get_datasets(
                    bucket='datasets',
                    dataset='ml-25m',
                    split='test').after(qa_op)
    val_data =  get_datasets(
                    bucket='datasets',
                    dataset='ml-25m',
                    split='val').after(qa_op)
    
    negative_sample_training = negative_sampling(
                    num_ng_test=10,
                    input=train_data.output
    ).after(train_data)
    
    training = train_model(
        mlflow_experiment_name='recommender',
        mlflow_run_id=None,
        mlflow_tags={},
        hot_reload_model_run_id=None,
        model_embedding_factors=20,
        model_learning_rate=1e-3,
        model_hidden_dims=256,
        model_dropout_rate=0.2,
        optimizer_step_size=10, optimizer_gamma=0.1,
        training_epochs=30,
        train_batch_size=64,
        test_batch_size=64,
        training_data=negative_sample_training.output,
        testing_data=test_data.output,
        shuffle_training_data=True,
        shuffle_testing_data=True).after(negative_sample_training)
    
    val = validate_model(
        model_run_id=training.output,
        top_k=50,
        threshold=3,
        val_batch_size=32,
        validation_dataset=val_data.output
    ).after(training)
    
    promote_model_to_staging(
        model_run_id=training.output,
        registered_model_name='recommender_production',
        rms_threshold=0.0,
        precision_threshold=-0.3,
        recall_threshold=-0.2).after(val)

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path='training_pipeline.yaml')