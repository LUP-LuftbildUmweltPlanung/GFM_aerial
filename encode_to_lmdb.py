import lmdb
import os

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


def read_specific_key_from_lmdb(db, key):
    """
    Liest einen spezifischen Key aus der LMDB-Datenbank.

    :param db: LMDB-Umgebung
    :param key: Schlüssel (String)
    """
    with db.begin() as txn:
        value = txn.get(key.encode())
        print(f"🔍 Gelesener Wert für '{key}':", value.decode() if value else "❌ Nicht gefunden")


def list_all_entries(db):
    """
    Listet alle Key-Value-Paare in der LMDB-Datenbank auf.

    :param db: LMDB-Umgebung
    """
    with db.begin() as txn:
        cursor = txn.cursor()
        print("\n📂 **Alle Einträge in LMDB:**")
        for key, value in cursor:
            print(f"🔹 {key.decode()} -> {value.decode()}")



def write_to_lmdb(db, key, value):
    """
    Write (key, value) to LMDB, automatically increasing the map size if needed.
    :param db: LMDB environment
    :param key: Key (should be string, gets converted to bytes)
    :param value: Value (must be bytes)
    """
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


lmdb_path = "/home/embedding/Data_Center/Vera/GFM_aerial/test_lmdb2.lmdb"
new_lmdb = create_or_open_lmdb(lmdb_path)
write_to_lmdb(new_lmdb, "4","test4")
read_specific_key_from_lmdb(new_lmdb, "3")