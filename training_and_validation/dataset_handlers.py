

class datasetReader(Dataset):
    def __init__(self, df, dataset_name):
        self.df = df
        self.name = dataset_name
        print(f"{self.name} : {self.df.shape[0]}")

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        sd = self.df.iloc[idx]
        user = sd['user_id']
        item = sd['item_id']
        rating = sd['rating']
        return torch.tensor(user-1).long(), torch.tensor(item-1).long(), torch.tensor(rating).float()

### Testing code.
def negative_sampling(ratings, num_ng_test=10):
    print('x')
    item_pool = set(ratings['item_id'].unique())
    print('y')
    interact_status = (
			ratings.groupby('user_id')['item_id']
			.apply(set)
			.reset_index()
			.rename(columns={'item_id': 'interacted_items'}))
    print('z')
    interact_status['negative_samples'] = interact_status['interacted_items'].apply(lambda x: np.random.choice(list(item_pool - x), num_ng_test))
    interact_status['rating'] = 0.0
    interact_status['timestamp'] = 1051631039
    interact_status = interact_status.drop(columns=['interacted_items']).explode('negative_samples').rename(columns={'negative_samples':'item_id'})
    print('a')
    #ret = ratings.append(interact_status, ignore_index=True)
    ret = pd.concat([ratings, interact_status], ignore_index=True)
    return ret

def split_dataset(path_to_ml_25m = "/Users/shanoop/Downloads/ml-25m",  random_state: int = 42):
    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10
	
    num_ng_test = 10
    num_ng = 10

    names = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings_df = pd.read_csv(os.path.join(path_to_ml_25m, 'ratings.csv'),names=names, index_col=False, skiprows=1)
    ratings_df = ratings_df.iloc[:6400]
    ratings_df = negative_sampling(ratings_df)
	

    n_users = ratings_df.user_id.max()
    n_items = ratings_df.item_id.max()

    # train is now 75% of the entire data set
    train, test = train_test_split(
        ratings_df,                                    
        test_size=1 - train_ratio,
        random_state=random_state)
	
    # preprocess
    #train_ratings, test_ratings = leave_one_out(preprocess_ratings)

	

    # test is now 10% of the initial data set
    # validation is now 15% of the initial data set
    val, test = train_test_split(   
        test,
        test_size=test_ratio / (test_ratio + validation_ratio),
        random_state=random_state)
	
    
    return train, test, val, (n_users, n_items)

def get_datasets_local(bucket='datasets', dataset='ml-25m', split=['test', 'train', 'val']):
    train, test, val, (n_users, n_items) = split_dataset()

    data_map = {'n_users': n_users, 'n_items': n_items, 'train': train, 'test': test, 'val': val}

    for dataset_name in ['train', 'test', 'val']:
        data_map[dataset_name] = datasetReader(data_map[dataset_name], dataset_name=dataset_name)
    #    data_map['n_users'] = max(data_map['n_users'], df.user_id.max())
    #    data_map['n_items'] = max(data_map['n_items'], df.item_id.max())
    
    assert data_map['n_users'] == n_users
    assert data_map['n_items'] == n_items
    return data_map