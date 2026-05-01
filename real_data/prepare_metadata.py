import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import json
import os


config_path = Path("config/config.json")
config = json.load(open(config_path))

metadata_path = config['metadata_path']

columns = ['L.poj.', "FV_LEWA_MCA.3", "FV_PRAWA_MCA.3"]
df = pd.read_excel(metadata_path, skiprows=6)[columns]


# patient id column
df["patient_id"] = df["L.poj."].apply(
    lambda x: f"PAC_{int(x):02d}" if x < 51 else None
)

df.to_csv("config/cbfv_config.csv", index=False)