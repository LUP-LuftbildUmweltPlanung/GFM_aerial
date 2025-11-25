import numpy as np
import pandas as pd
from pathlib import Path
import os
def calc_rmse_from_csv(csv_path, out_csv):
    df = pd.read_csv(csv_path)

    df_new = np.sqrt(df.loc[:,df.columns.str.contains("l2")])

    df_new.to_csv(out_csv)

    print(df_new)



input_folders = r"PATH" # sth like ...\stats

for csv_file in os.listdir(input_folders):
    f = Path(input_folders)
    name = Path(csv_file)
    out_csv = f / f"{name.stem}_rsme{name.suffix}"
    in_csv = f / csv_file
    print(out_csv)
    print(in_csv)
    calc_rmse_from_csv(in_csv, out_csv)
