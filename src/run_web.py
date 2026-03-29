import json, time
from data_manager import DataManager
from web_dashboard import create_app

with open("config.json","r") as f:
    cfg = json.load(f)

dm = DataManager(cfg)

def dummy_next_frame(timeout: float = 1.0):
    return None  # no live stream; UI stays clean

app = create_app(dm.db_path, dm.latest_assets, dummy_next_frame)
import os
port=int(os.getenv('WEB_PORT','5000'))
app.run(host='0.0.0.0', port=port, threaded=True, use_reloader=False)
