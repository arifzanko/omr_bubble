from steps.data_ingestion import ingest_df
from steps.data_split import split_df
from steps.model_train import model_train
from steps.model_track import model_track

def train_pipeline():
    ingest_df()
    split_df()
    model_train()
    model_track()