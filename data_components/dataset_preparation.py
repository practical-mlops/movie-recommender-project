import kfp.components as comp
from kfp import dsl
from kfp.dsl import Input, Output, Model, Dataset, OutputPath, Artifact


from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import zipfile
import boto3
from pyarrow import fs, parquet

@dsl.component(base_image="python:3.11", target_image="mlsysfromscratch/data_preprocessor:1.0.0", packages_to_install=["requests"])
def download_ml25m_data(output_path_one: Output[Artifact]):
    import requests
    url = 'https://files.grouplens.org/datasets/movielens/ml-25m.zip'
    response = requests.get(url, stream=True, verify=False)
    #file_size = int(response.headers.get("Content-Length", 0))
    #progress_bar = tqdm(total=file_size, unit="B", unit_scale=True)
    print(output_path_one.path)
    with open(output_path_one.path, 'wb') as file: 
        for chunk in response.iter_content(chunk_size=1024*1024): #D
            # Update the progress bar with the size of the downloaded chunk #D
            #progress_bar.update(len(chunk)) #D
            if chunk:
                file.write(chunk)

@dsl.component(base_image="python:3.11", target_image="mlsysfromscratch/data_preprocessor:1.0.0")            
def unzip_data(input_path: Input[Artifact], ratings_output_path: Output[Artifact], movies_output_path: Output[Artifact]):
    import zipfile

    with zipfile.ZipFile(input_path.path, 'r') as z:
        with open(ratings_output_path.path, 'wb') as f:
            f.write(z.read('ml-25m/ratings.csv'))
        with open(movies_output_path.path, 'wb') as f:
            f.write(z.read('ml-25m/movies.csv'))

@dsl.component(base_image="python:3.11", target_image="mlsysfromscratch/data_preprocessor:1.0.0", packages_to_install=["scikit-learn", "pandas", "fastparquet"])
def split_dataset(input_parquet: Input[Artifact], dataset_path: Output[Artifact], random_state: int = 42):
    from sklearn.model_selection import train_test_split
    import os
    import pandas as pd
    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10
    ratings_df = pd.read_parquet(input_parquet.path)

    # train is now 75% of the entire data set
    train, test = train_test_split(
        ratings_df,                                    
        test_size=1 - train_ratio,
        random_state=random_state)
    
    n_users = ratings_df.user_id.max()
    n_items = ratings_df.item_id.max()

    # test is now 10% of the initial data set
    # validation is now 15% of the initial data set
    val, test = train_test_split(   
        test,
        test_size=test_ratio / (test_ratio + validation_ratio),
        random_state=random_state)
    os.mkdir(dataset_path.path)
    train.to_parquet(os.path.join(dataset_path.path, 'train.parquet.gzip'), compression='gzip')
    test.to_parquet(os.path.join(dataset_path.path, 'test.parquet.gzip'), compression='gzip')
    val.to_parquet(os.path.join(dataset_path.path, 'val.parquet.gzip'), compression='gzip')

@dsl.component(base_image="python:3.11", target_image="mlsysfromscratch/data_preprocessor:1.0.0", packages_to_install=["scikit-learn", "pandas", "fastparquet"])
def csv_to_parquet(inputFile: Input[Artifact], output_path: Output[Artifact]):
    import pandas as pd
    df = pd.read_csv(inputFile.path, index_col=False)
    df.to_parquet(output_path.path, compression='gzip') 

@dsl.component(base_image="python:3.11", target_image="mlsysfromscratch/data_preprocessor:1.0.0", packages_to_install=["boto3"])
def put_to_minio(inputFile: Input[Artifact], upload_file_name:str='', bucket: str='datasets'):
    import boto3
    import os
    minio_client = boto3.client(                          
        's3',                                              
        endpoint_url='http://minio-service.kubeflow:9000',
        aws_access_key_id='minio',
        aws_secret_access_key='minio123') 
    try:
        minio_client.create_bucket(Bucket=bucket)
    except Exception as e:
        # Bucket already created.
        pass
    if os.path.isdir(inputFile.path):
        for file in os.listdir(inputFile.path):
            s3_path = os.path.join('ml-25m', file)
            minio_client.upload_file(os.path.join(inputFile.path, file), bucket, s3_path)
    else:
        if upload_file_name == '':
            _, file = os.path.split(inputFile.path)
        else:
            file = upload_file_name
        s3_path = os.path.join('ml-25m', file)
        minio_client.upload_file(inputFile.path, bucket, s3_path)

@dsl.component(base_image="python:3.11", target_image="mlsysfromscratch/data_preprocessor:1.0.0", packages_to_install=["pyarrow"]) 
def qa_data(bucket:str = 'datasets', dataset:str = 'ml-25m'):
    from pyarrow import fs, parquet
    print("Running QA")
    minio = fs.S3FileSystem(
        endpoint_override='http://minio-service.kubeflow:9000',
         access_key='minio',
         secret_key='minio123',
         scheme='http')
    train_parquet = minio.open_input_file(f'{bucket}/{dataset}/train.parquet.gzip')
    df = parquet.read_table(train_parquet).to_pandas()
    assert df.shape[1] == 4
    assert df.shape[0] >= 0.75 * 25 * 1e6
    print('QA passed!')