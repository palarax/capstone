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

if __name__ == "__main__":
    setup_log("../configuration/log_config.json")
    db = Xaidb("../database/xai_db.json")
    # db.create_table("ilya")
    tiny = db.get_tiny_instance()
    print(tiny.tables())
