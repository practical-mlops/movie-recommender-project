

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