import logging
from tinydb import TinyDB, Query, where
from tinydb.operations import delete, increment
from tinydb.database import Table, StorageProxy, Document
# Tables: Counter, Images | Signals | Risks

# Risks
# id, class, img, start, end, xmin, ymin, xmax , ymax

# Signals
# type, color_arr
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
    '''Database for XAI. This provides a wrapper around the chosen database
    which is tinydb in this case
    '''
    def __init__(self, db_name):
        self.db = TinyDB(db_name)

    def get_tiny_instance(self):
        return self.db

    def get_all_signals(self):
        logging.debug("[Xaidb] Getting all signal classifications")
        signals = {}
        for sig in self.db.table('Signals').all():
            signals[sig["type"]] = sig["colour"]
        return signals

    def get_signal(self, name):
        logging.debug("[Xaidb] Getting Signal [%s]", name)
        table = self.db.table('Signals')
        sig = table.search(where('type') == name)[0]
        return {sig["type"]: sig["colour"]}

    def get_image_counter(self):
        logging.debug("[Xaidb] Getting Image Counter")
        table = self.db.table('Counter').all()
        return table[0]["Image"]

    def increment_image_counter(self):
        logging.debug("[Xaidb] Incrementing Image Counter")
        table = self.db.table('Counter')
        table.update(increment("Image"), doc_ids=[1])

    def get_risk_counter(self):
        logging.debug("[Xaidb] Getting Risk Counter")
        table = self.db.table('Counter').all()
        return table[0]["Risk"]

    def increment_risk_counter(self):
        logging.debug("[Xaidb] Incrementing Risk Counter")
        table = self.db.table('Counter')
        table.update(increment("Risk"), doc_ids=[1])