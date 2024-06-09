import pandas as pd
import psycopg2
import torch
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from torch.utils.data import DataLoader, Dataset


class MovieDataset(Dataset):
    def __init__(self, users, movies, ratings, device):
        self.users = users
        self.movies = movies
        self.ratings = ratings
        self.device = device

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        users = self.users[idx]
        movies = self.movies[idx]
        ratings = self.ratings[idx]
        return_dict = {
            "users": torch.tensor(users, dtype=torch.long).to(self.device),
            "movies": torch.tensor(movies, dtype=torch.long).to(self.device),
            "ratings": torch.tensor(ratings, dtype=torch.long).to(self.device),
        }
        return return_dict


def _create_movie_dataset(df: pd.DataFrame, device):
    dataset = MovieDataset(
        users=df["userIdEncoded"].values,
        movies=df["movieIdEncoded"].values,
        ratings=df["rating"].values,
        device=device,
    )
    return dataset


def load_dataframe(
    host: str, database: str, user: str, password: str, port: int
) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=host,
        database=database,
        user=user,
        password=password,
        port=port,
    )
    conn.set_session(autocommit=True)
    engine = create_engine(
        f"postgresql://{user}:{password}@{host}:{port}/{database}",
        connect_args={"sslmode": "require"},
    )
    df = pd.read_sql("SELECT * FROM ratings", con=engine)
    return df


def create_dataloaders(
    df_train: pd.DataFrame, df_test: pd.DataFrame, device: str, batch_size: int
) -> tuple[DataLoader, DataLoader]:
    train_dataset = _create_movie_dataset(df_train, device)
    test_dataset = _create_movie_dataset(df_test, device)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def create_train_test_split(
    df: pd.DataFrame, test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df["rating"].values
    )

    return df_train, df_test
