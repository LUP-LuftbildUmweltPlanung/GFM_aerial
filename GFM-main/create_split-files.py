import random
import pathlib

root = pathlib.Path(r"/home/embedding/Data_Center/Vera/Finetuning_datasets/4_bands/Images/") #end with /Images

train, val, test = [], [], []

for dir in pathlib.Path.glob(root, "*/"):
    print(dir)
    tif_files = [str(f) for f in pathlib.Path.rglob(root/dir,r"*.tif") if str(f).endswith(('.tif', '.tiff'))]
    print(len(tif_files))
    random.shuffle(tif_files)
    n = len(tif_files)
    train += tif_files[:int(n*0.7)]
    #val   += images[int(n*0.7):int(n*0.85)]
    val   += tif_files[int(n*0.7):int(n)]
    #test  += images[int(n*0.85):]

with open("4-band-train.txt", "w") as f:
    f.write("\n".join(train))
with open("4-band-val.txt", "w") as f:
    f.write("\n".join(val))
with open("4-band-test.txt", "w") as f:
    f.write("\n".join(test))
