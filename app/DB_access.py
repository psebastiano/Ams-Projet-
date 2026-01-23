import os
from pymongo import MongoClient

# --- Configuration ---
MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb+srv://byanismci_db_user:ciFm8mSSBfSB6GOh@cluster0.tdoyk6j.mongodb.net/multisport"
)
MONGODB_DB = os.getenv("MONGODB_DB", "multisport")


class DatabaseMongo:
    def __init__(self) -> None:
        self.client = MongoClient(MONGODB_URI)
        self.db = self.client[MONGODB_DB]
    
    def get_collection(self, collection_name):
        return self.db[collection_name]
    
    def close(self):
        self.client.close()
