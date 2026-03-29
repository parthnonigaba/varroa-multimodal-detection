import os, json
from datetime import datetime
from data_manager import DataManager
from audio_monitor import AudioHealthMonitor

with open("/home/parthgaba/bee_monitoring/config.json","r") as f:
    cfg = json.load(f)

dm = DataManager(cfg)
am = AudioHealthMonitor(cfg)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = cfg.get("storage",{}).get("audio_dir","data/audio")
os.makedirs(out_dir, exist_ok=True)
wav = os.path.join(out_dir, f"audio_{ts}.wav")

if am._record(wav):
    label, conf = am.analyze(wav)
    unhealthy = (label == am.unhealthy_label and conf >= am.unhealthy_threshold)
    dm.save_audio_analysis(datetime.now(), wav, label, conf, unhealthy)
    if unhealthy:
        dm.save_event(datetime.now(), "unhealthy_audio", {"label": label, "confidence": conf}, wav)
    print("OK", wav, label, conf, unhealthy)
else:
    print("REC_FAIL")
