import logging
from tinydb import TinyDB
from tinydb.database import Table, StorageProxy, Document
from tinydb import Query

# Tables: Counter, Images | Signals | Risks

# Risks
# id, class, img, start, end, xmin, ymin, xmax , ymax

# Signals
# name, color_arr
# ======================
# BACKGROUND = (0, 0, 0)
# DANGER = (0, 0, 255)  # RED
# WARNING = (0, 128, 255)  # ORANGE
# CAUTION = (0, 255, 255)  # YELLOW
# NO_IMMEDIATE_DANGER = (255, 0, 0)  # BLUE

# Images
# name, 

# Counter
# img, risk, 

class Xaidb:
    def __init__(self, db_name):
        self.db = TinyDB(db_name)

    def get_tiny_instance(self):
        return self.db

    def create_table(self, name):
        logging.info("create table")
        self.db.table(name)


