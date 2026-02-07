import pandas as pd
from tqdm import tqdm

# with open('electronics_CORE_15.jsonl', 'r', encoding='utf-8') as f:
#    lines = [line for line in f if line.strip()]  # ignora righe vuote o solo spazi

# dataset = pd.read_json(io.StringIO("".join(lines)), lines=True)
# columns: ['rating', 'title', 'text', 'images', 'asin', 'parent_asin',
# 'user_id', 'timestamp', 'helpful_vote', 'verified_purchase']
dataset = pd.read_json("electronics_CORE_15.jsonl", lines=True)
print(len(dataset))
print(dataset.columns)

# sort by user_id and timestamp
sorted_dataset = dataset.sort_values(by=["user_id", "timestamp"])

# print(len(sorted_dataset['rating'].isin([4,5])))
print(len(sorted_dataset[sorted_dataset["rating"].isin([5])]))
print(len(sorted_dataset[sorted_dataset["rating"].isin([1, 2, 3, 4])]))


# binarize ratings
sorted_dataset["rating_bin"] = sorted_dataset["rating"].apply(
    lambda x: 1 if x == 5 else 0
)


# build user/item maps
map_users = {user_id: i for i, user_id in enumerate(sorted_dataset["user_id"].unique())}
map_items = {item_id: i for i, item_id in enumerate(sorted_dataset["asin"].unique())}

# Crea mapping asin -> parent_asin dal dataset
asin_to_parent = {}
for _, row in sorted_dataset.iterrows():
    asin = row["asin"]
    parent_asin = row["parent_asin"]
    if asin not in asin_to_parent:
        asin_to_parent[asin] = parent_asin


# save maps as tsv files
map_users_df = pd.DataFrame(list(map_users.items()), columns=["user_id", "user_index"])
map_users_df.to_csv("maps/user_map.tsv", sep="\t", index=False)


# add parent_asin al mapping degli item
map_items_data = []
for item_id, item_index in map_items.items():
    parent_asin = asin_to_parent.get(
        item_id, item_id
    )  # fallback su asin se parent_asin non trovato
    map_items_data.append((item_id, item_index, parent_asin))

map_items_df = pd.DataFrame(
    map_items_data, columns=["item_id", "item_index", "parent_asin"]
)
map_items_df.to_csv("maps/item_map.tsv", sep="\t", index=False)

# build interaction tuples (user_id, item_id, binary rating)
ratings = []
for _, row in tqdm(sorted_dataset.iterrows(), total=len(sorted_dataset)):
    user_id = map_users[row["user_id"]]
    item_id = map_items[row["asin"]]
    ratings.append((user_id, item_id, row["rating_bin"]))

# build DataFrame
ratings_recbole = pd.DataFrame(
    ratings, columns=["user_id:token", "item_id:token", "rating:float"]
)

# shuffle dataset befor splitting into train, valid, test sets
ratings_recbole = ratings_recbole.sample(frac=1, random_state=42).reset_index(drop=True)


# split into train, valid, test sets
train_size = int(0.8 * len(ratings_recbole))
valid_size = int(0.1 * len(ratings_recbole))
train_data = ratings_recbole.iloc[:train_size]
valid_data = ratings_recbole.iloc[train_size : train_size + valid_size]
test_data = ratings_recbole.iloc[train_size + valid_size :]

# save DataFrames to tsv files
train_data.to_csv(
    "../dataset/amazon_elec/amazon_elec.train.inter", sep="\t", index=False
)
valid_data.to_csv(
    "../dataset/amazon_elec/amazon_elec.valid.inter", sep="\t", index=False
)
test_data.to_csv("../dataset/amazon_elec/amazon_elec.test.inter", sep="\t", index=False)

# save the sorted dataset with binary ratings
ratings_recbole.to_csv(
    "../dataset/amazon_elec/amazon_elec.inter", sep="\t", index=False
)

# print dataset statistics
print(f"Total interactions: {len(ratings_recbole)}")
print(f"Train set size: {len(train_data)}")
print(f"Validation set size: {len(valid_data)}")
print(f"Test set size: {len(test_data)}")