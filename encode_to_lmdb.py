import lmdb
import os
import rasterio
from safetensors.numpy import save, load
import numpy as np


def read_tif_bands(tif_path):
    """
    Liest eine TIFF-Datei ein und gibt ein Dictionary mit den Bändern zurück.

    :param tif_path: Pfad zur TIFF-Datei
    :return: Dictionary {Bandname: NumPy-Array}
    """
    band_dict = {}
    key = os.path.basename(tif_path).replace(".tif", "")

    with rasterio.open(tif_path) as src:
        for i in range(1, src.count + 1):  # Bänder starten bei 1 in rasterio
            band_name = str(i)  # Beispiel: "B01", "B02", ...
            band_dict[band_name] = src.read(i)
    return key, band_dict


def save_bands_to_safetensor(bands_dict):
    """
    Speichert alle Bänder als Safetensor-Format in ein Dictionary.

    :param bands_dict: Dictionary {Bandname: NumPy-Array}
    :return: Bytes-Objekt mit Safetensor-Daten
    """
    return save(bands_dict)


def write_to_lmdb(db, key, safetensor_data):
    """
    Schreibt ein mehrdimensionales Safetensor-Objekt in eine LMDB-Datenbank.

    Falls die LMDB zu klein ist, wird `map_size` automatisch verdoppelt.

    :param db: LMDB-Umgebung
    :param key: Schlüssel für den Safetensor (z. B. Name der TIFF-Datei)
    :param bands_dict: Dictionary mit {Bandname: NumPy-Array}, das gespeichert werden soll
    """
    success = False

    while not success:
        txn = db.begin(write=True)
        try:
            txn.put(key.encode(), safetensor_data)  # Key zu Bytes umwandeln
            txn.commit()
            success = True
            print(f"✅ TIFF '{key}' erfolgreich in LMDB gespeichert!")
        except lmdb.MapFullError:
            txn.abort()  # Transaktion abbrechen
            curr_limit = db.info()['map_size']
            new_limit = curr_limit * 2
            print(f"⚠️ Speicher voll! Verdopple LMDB-Größe auf {new_limit >> 20}MB ...")
            db.set_mapsize(new_limit)  # Speichergröße erhöhen


def read_from_lmdb(db, key):
    """
    Liest ein Safetensor aus der LMDB-Datenbank aus.

    :param lmdb_path: Pfad zur LMDB-Datenbank
    :param key: Schlüssel des gespeicherten Safetensors
    :return: Dictionary mit geladenen Bändern als NumPy-Arrays
    """
    with db.begin() as txn:
        safetensor_data = txn.get(key.encode())
    #env.close()

    if safetensor_data is None:
        print(f"❌ Kein Eintrag für '{key}' in LMDB gefunden!")
        return None
    return load(safetensor_data)


def create_or_open_lmdb(lmdb_path, size=None):
    """
    Erstellt eine neue LMDB-Datenbank oder öffnet eine bestehende.

    - Falls `size` gegeben ist, wird dieser Wert genutzt.
    - Falls die LMDB existiert und `size` nicht gegeben ist, wird die bestehende `map_size` genutzt.
    - Falls die LMDB nicht existiert und `size` nicht gegeben ist, wird ein Standardwert (10MB) verwendet.

    :param lmdb_path: Pfad zur LMDB-Datenbank
    :param size: Maximale Speichergröße in Bytes (optional)
    :return: LMDB-Umgebung (db)
    """
    if os.path.exists(lmdb_path):
        print(f"⚡ Öffne bestehende LMDB: {lmdb_path}")

        # Bestehende DB öffnen, um aktuelle Größe zu ermitteln
        temp_env = lmdb.open(lmdb_path, readonly=True)
        existing_size = temp_env.info()['map_size']
        temp_env.close()

        # Falls size explizit gegeben wurde, nutze diesen Wert
        map_size = size if size else existing_size
        print(f"📏 Verwende existierende map_size: {map_size >> 20}MB")

        return lmdb.open(lmdb_path, map_size=map_size)
    else:
        # Falls `size` nicht gegeben ist, nutze Standardwert von 10MB
        default_size = 10 * 1024 * 1024
        map_size = size if size else default_size
        print(f"🆕 Erstelle neue LMDB: {lmdb_path} mit {map_size >> 20}MB Speicher")

        return lmdb.open(lmdb_path, map_size=map_size)


def read_all_from_lmdb(db):
    """
    Liest alle Safetensor-Daten aus einer LMDB-Datenbank und gibt sie als Dictionary zurück.

    :param lmdb_path: Pfad zur LMDB-Datenbank
    :return: Dictionary {TIFF-Name: {Bandname: NumPy-Array}}
    """
    all_data = {}

    # LMDB im Read-Only-Modus öffnen
    db = lmdb.open(lmdb_path, readonly=True)

    with db.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            key_str = key.decode()  # Key (TIFF-Name) als String
            safetensor_data = load(value)  # Safetensor-Daten dekodieren
            all_data[key_str] = safetensor_data

    db.close()
    return all_data


"""def read_specific_key_from_lmdb(db, key):
    
    #Liest einen spezifischen Key aus der LMDB-Datenbank.
    #:param db: LMDB-Umgebung
    #:param key: Schlüssel (String)
    
    with db.begin() as txn:
        value = txn.get(key.encode())
        print(f"🔍 Gelesener Wert für '{key}':", value.decode() if value else "❌ Nicht gefunden")

def list_all_entries(db):
    
    #Listet alle Key-Value-Paare in der LMDB-Datenbank auf.
    #:param db: LMDB-Umgebung
    
    with db.begin() as txn:
        cursor = txn.cursor()
        print("\n📂 **Alle Einträge in LMDB:**")
        for key, value in cursor:
            print(f"🔹 {key.decode()} -> {value.decode()}")

def write_to_lmdb(db, key, value):
    
    #Write (key, value) to LMDB, automatically increasing the map size if needed.
    #:param db: LMDB environment
    #:param key: Key (should be string, gets converted to bytes)
    #:param value: Value (must be bytes)
    
    success = False
    while not success:
        txn = db.begin(write=True)
        try:
            txn.put(key.encode(), value.encode())  # Key explizit in Bytes umwandeln
            txn.commit()
            success = True
        except lmdb.MapFullError:
            txn.abort()
            curr_limit = db.info()['map_size']
            new_limit = curr_limit * 2
            print(f">>> Doubling LMDB map size to {new_limit >> 20}MB ...")
            db.set_mapsize(new_limit)  # Speichergröße verdoppeln
"""


def process_tiff_folder(folder_path, lmdb_path):
    """
    Liest alle TIFF-Dateien aus einem Ordner und speichert sie als Safetensors in einer LMDB-Datei.

    :param folder_path: Pfad zum Ordner mit den TIFF-Dateien
    :param lmdb_path: Pfad zur LMDB-Datenbank
    """
    file_list = [f for f in os.listdir(folder_path) if f.endswith(".tif")]
    print(f"📂 {len(file_list)} TIFF-Dateien gefunden. Starte LMDB-Speicherung...\n")

    db = create_or_open_lmdb(lmdb_path)

    for tif_file in file_list:
        tif_path = os.path.join(folder_path, tif_file)
        key, bands_dict = read_tif_bands(tif_path)  # TIFF-Daten extrahieren
        bands_dict_safetensor = save_bands_to_safetensor(bands_dict)
        write_to_lmdb(db, key, bands_dict_safetensor)  # In LMDB speichern
        print(f"✅ {key} gespeichert mit {len(bands_dict_safetensor)} Bändern")

    db.close()
    print("\n✅ Alle TIFF-Dateien erfolgreich in LMDB gespeichert!\n")

    all_images = read_all_from_lmdb(lmdb_path)

    print("🔑 Gespeicherte Keys & Bänder in LMDB:")
    for tif_name, bands in all_images.items():
        print(f"🖼️ {tif_name}: {list(bands.keys())}")

# -------------------------------
# Beispielhafte Nutzung
# -------------------------------
folder_path = "/home/embedding/Data_Center/Vera/GeoPile/GeoPileV0/folder1/folder2"  # Ändere den Pfad zum Ordner mit TIFF-Dateien
lmdb_path = "/home/embedding/Data_Center/Vera/GeoPile/GeoPileV0/sample_lmdb.lmdb"  # Name der LMDB-Datei

process_tiff_folder(folder_path, lmdb_path)



"""lmdb_path = "/home/embedding/Data_Center/Vera/GFM_aerial/test_lmdb3.lmdb"
tif_path = "/home/embedding/Data_Center/Vera/GeoPile/GeoPileV0/folder1/folder2/Proesa_DBUNE_2020_29.tif"
key, band_dict = read_tif_bands(tif_path)
band_dict_safetensor = save_bands_to_safetensor(band_dict)
new_lmdb = create_or_open_lmdb(lmdb_path)
write_to_lmdb(new_lmdb, key, band_dict_safetensor)
#img_safetensor = read_from_lmdb(new_lmdb, key)
img_safetensors = read_all_from_lmdb(new_lmdb)
print(img_safetensors.items().items())
for band, array in img_safetensors.items().items():
    print(f"{band}: {array.shape}")"""