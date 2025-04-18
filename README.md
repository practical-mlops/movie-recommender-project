# movie-recommender-project

## Setting up MLFlow

### Option 1: Installing MLFlow in the cluster and using cluster DNS
This project uses MLFLow for metrics logging and model storage. 
Install the MLFlow instance as mentioned in the appendix and modify the mlflow_uri with the cluster DNS in the training and validation pipeline.
This is the recommended method of installing MLFlow in a proper cluster.

### Option 2: Installing MLFlow locally
> This method is intended to be only for quick trials. Use the cluster install method in the appendix so that you do not loose critical data.

Alternatively, if you want to try this repo out, you can install and run MLFlow either locally or in another machine.
To do this
- Install MLFlow with 
```bash
pip install MLFlow
```
- Start the tracking server with 
```bash
mlflow server --port 8080
```
- Get the IP of the server with ip addr show and configure the mlflow_uri to point to http://IP:8080

### Modifying the MLFlow uri in code

Edit the following links to point to your new MFlow instance
- Input parameter in the training_and_validation pipeline. This can be set for a run when running the pipeline, but you can edit it in the pipeline definition to set the default value.
- tracking_uri in service.py in the serving directory
- the tracking_uri in the model inference notebook

## Building the pipelines

Build the pipelines by running 01_data_preparation_pipeline.py and 02_training_and_validation.py files. This should compile two yaml files in the compiled pipelines folder.

## Running the pipelines

Upload the pipeline yaml in compiled_pipelines to kubeflow using the UI.

Run the data preparation pipeline first to download and save the dataset to the local minio instance. 
After that, run the training and validation pipeline to train a model and save it to MLFlow. 
> Remember to tweak the number of epochs and keep it low when doing initial tests so that you can iterate quickly. 

There should be metrics logged while training is ongoing in MLFlow. At the end of training, the model will also be registered in MFLow.

## Testing the model

Configure the tracking_uri in the notebook in notebooks/model_inference_bentoml.ipynb to point to the MLFLow instance with the registered model. 
In the current configuration, the model will be registered with the name `recommender_production` and a version alias `prod`

Run the notebook to retrieve the prod model and run an inference locally. 