from kfp.dsl import Input, Output, Dataset, Artifact, component
from typing import Dict

@component(packages_to_install=["scikit-learn", "pandas", "fastparquet", "pyarrow"])
def get_dataset_metadata(bucket:str , dataset_name:str) -> Dict[str, int]:
    from pyarrow import fs, parquet
    valid_splits = ['test', 'train', 'val']
    data_map = {'n_users': 0, 'n_items': 0}
    minio = fs.S3FileSystem(
        endpoint_override='http://minio-service.kubeflow:9000',
         access_key='minio',
         secret_key='minio123',
         scheme='http')

    for valid_split in valid_splits:
        paraquet_data = minio.open_input_file(f'{bucket}/{dataset_name}/{valid_split}.parquet.gzip')
        df = parquet.read_table(paraquet_data).to_pandas()
        print(df.head())
        data_map['n_users'] = max(data_map['n_users'], int(df.userId.max()))
        data_map['n_items'] = max(data_map['n_items'], int(df.movieId.max()))
    
    #assert list(data_map.keys()) == split, f"Mismatched or invalid splits. Received {split} but can only process {valid_splits}"
    return data_map

@component(packages_to_install=["pandas", "fastparquet", "numpy", "pyarrow"])
def get_dataset(bucket: str, dataset_name: str, split: str, output_dataset: Output[Dataset]):
    from pyarrow import fs, parquet
    import pandas as pd

    assert split in ['test', 'train', 'val']
    minio = fs.S3FileSystem(
        endpoint_override='http://minio-service.kubeflow:9000',
         access_key='minio',
         secret_key='minio123',
         scheme='http')
    paraquet_data = minio.open_input_file(f'{bucket}/{dataset_name}/{split}.parquet.gzip')
    df = parquet.read_table(paraquet_data).to_pandas()
    pd.to_pickle(df, output_dataset.path)


@component(packages_to_install=["pandas", "fastparquet", "numpy", "pyarrow"])
def negative_sampling(num_ng_test:int, bucket:str , dataset_name:str, split: str, negative_sampled_dataset: Output[Dataset]):
    import pandas as pd
    from pyarrow import fs, parquet
    import numpy as np

    minio = fs.S3FileSystem(
        endpoint_override='http://minio-service.kubeflow:9000',
         access_key='minio',
         secret_key='minio123',
         scheme='http')
    paraquet_data = minio.open_input_file(f'{bucket}/{dataset_name}/{split}.parquet.gzip')
    ratings = parquet.read_table(paraquet_data).to_pandas()
    item_pool = set(ratings['movieId'].unique())
    interact_status = (
			ratings.groupby('userId')['movieId']
			.apply(set)
			.reset_index()
			.rename(columns={'movieId': 'interacted_items'}))
    interact_status['negative_samples'] = interact_status['interacted_items'].apply(lambda x: np.random.choice(list(item_pool - x), num_ng_test))
    interact_status['rating'] = 0.0
    interact_status['timestamp'] = 1051631039
    interact_status = interact_status.drop(columns=['interacted_items']).explode('negative_samples').rename(columns={'negative_samples':'movieId'})
    #ret = ratings.append(interact_status, ignore_index=True)
    ret = pd.concat([ratings, interact_status], ignore_index=True)
    pd.to_pickle(ret, negative_sampled_dataset.path)