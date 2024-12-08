from kfp.dsl import Input, Output, Dataset, Artifact, component

@component(packages_to_install=["scikit-learn", "pandas", "fastparquet"])
def get_datasets(bucket:str , dataset_name:str , split: str, dataset: Output[Dataset]):
    from pyarrow import fs, parquet
    valid_splits = ['test', 'train', 'val']
    data_map = {}
    minio = fs.S3FileSystem(
        endpoint_override='http://minio-service.kubeflow:9000',
         access_key='minio',
         secret_key='minio123',
         scheme='http')

    for valid_split in ['train', 'test', 'val']:
        paraquet_data = minio.open_input_file(f'{bucket}/{dataset_name}/{valid_split}.parquet.gzip')
        df = parquet.read_table(paraquet_data).to_pandas()
        data_map['n_users'] = max(data_map['n_users'], df.user_id.max())
        data_map['n_items'] = max(data_map['n_items'], df.item_id.max())
        if valid_split == split:
            data_map[valid_split] = datasetReader(df, dataset_name=dataset_split)
    
    assert list(data_map.keys()) == split, f"Mismatched or invalid splits. Received {split} but can only process {valid_splits}"
    return data_map

@component(packages_to_install=["pandas"])
def negative_sampling(num_ng_test:int , input: Input[Dataset], sampled_dataset: Output[Dataset]):
    import pandas as pd
    ratings = pd.read_pickle(input.path)
    item_pool = set(ratings['item_id'].unique())
    interact_status = (
			ratings.groupby('user_id')['item_id']
			.apply(set)
			.reset_index()
			.rename(columns={'item_id': 'interacted_items'}))
    interact_status['negative_samples'] = interact_status['interacted_items'].apply(lambda x: np.random.choice(list(item_pool - x), num_ng_test))
    interact_status['rating'] = 0.0
    interact_status['timestamp'] = 1051631039
    interact_status = interact_status.drop(columns=['interacted_items']).explode('negative_samples').rename(columns={'negative_samples':'item_id'})
    #ret = ratings.append(interact_status, ignore_index=True)
    ret = pd.concat([ratings, interact_status], ignore_index=True)
    pd.to_pickle(ret, sampled_dataset.path)