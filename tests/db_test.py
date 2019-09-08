from os import sys, path
import json
import logging
import logging.config
sys.path.append(path.abspath(path.join(path.dirname(__file__), '..')))

from db_utils.Xaidb import Xaidb

def setup_log(log_config):
    '''Setup logging
    '''
    with open(log_config, encoding='utf-8-sig') as conf_file:
        jc = json.load(conf_file)
        logging.config.dictConfig(jc["db_test"])

def test_signals(db):
    print(db.get_all_signals())
    print(db.get_signal("danger"))
    

def test_counter(db):
    print(db.get_image_counter())
    print(db.get_risk_counter())

if __name__ == "__main__":
    setup_log("../configuration/log_config.json")
    db = Xaidb("../database/xai_db.json")
    # =================================
    test_signals(db)
    test_counter(db)
    # =================================
    
    # tiny = db.get_tiny_instance()
    # counterTable = tiny.table("Counter")
    # counterTable.insert({"Image": 0, "Risk": 0})
    # db.increment_image_counter()
    # t = db.get_image_counter()
    # print(t)
    # print(s)
    # print()
