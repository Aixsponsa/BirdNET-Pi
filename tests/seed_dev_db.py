import sqlite3
from datetime import datetime, timedelta
import os

db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts', 'birds.db')

conn = sqlite3.connect(db_path)
cur = conn.cursor()

cur.execute("DROP TABLE IF EXISTS detections;")
cur.execute("""
CREATE TABLE IF NOT EXISTS detections (
  Date DATE,
  Time TIME,
  Sci_Name VARCHAR(100) NOT NULL,
  Com_Name VARCHAR(100) NOT NULL,
  Confidence FLOAT,
  Lat FLOAT,
  Lon FLOAT,
  Cutoff FLOAT,
  Week INT,
  Sens FLOAT,
  Overlap FLOAT,
  File_Name VARCHAR(100) NOT NULL);
""")
cur.execute('CREATE INDEX IF NOT EXISTS "detections_Com_Name" ON "detections" ("Com_Name");')
cur.execute('CREATE INDEX IF NOT EXISTS "detections_Sci_Name" ON "detections" ("Sci_Name");')
cur.execute('CREATE INDEX IF NOT EXISTS "detections_Date_Time" ON "detections" ("Date" DESC, "Time" DESC);')

# Also create images cache table so image provider doesn't fail
cur.execute("""
CREATE TABLE IF NOT EXISTS images (
  sci_name VARCHAR(63) NOT NULL PRIMARY KEY,
  com_en_name VARCHAR(63) NOT NULL,
  image_url TEXT NOT NULL,
  title TEXT NOT NULL,
  id TEXT NOT NULL UNIQUE,
  author_url TEXT NOT NULL,
  license_url TEXT NOT NULL,
  date_created DATE
);
""")

birds = [
    ("Thryothorus ludovicianus", "Carolina Wren", 0.88, "https://upload.wikimedia.org/wikipedia/commons/thumb/6/64/Carolina_Wren_in_PP_%2849842%29.jpg/640px-Carolina_Wren_in_PP_%2849842%29.jpg"),
    ("Cardinalis cardinalis", "Northern Cardinal", 0.94, "https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Northern_Cardinal_Male-27527-2.jpg/640px-Northern_Cardinal_Male-27527-2.jpg"),
    ("Cyanocitta cristata", "Blue Jay", 0.79, "https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/Blue_Jay_in_PP_%2830026%29.jpg/640px-Blue_Jay_in_PP_%2830026%29.jpg"),
    ("Turdus migratorius", "American Robin", 0.91, "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b8/Turdus-migratorius-002.jpg/640px-Turdus-migratorius-002.jpg"),
    ("Zenaida macroura", "Mourning Dove", 0.84, "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/Mourning_Dove_2006.jpg/640px-Mourning_Dove_2006.jpg"),
    ("Poecile atricapillus", "Black-capped Chickadee", 0.76, "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Poecile-atricapilla-001.jpg/640px-Poecile-atricapilla-001.jpg"),
]

today = datetime.now()
today_str = today.strftime("%Y-%m-%d")

# Populate images cache
for idx, (sci, com, conf, img_url) in enumerate(birds):
    cur.execute("""
    INSERT OR REPLACE INTO images (sci_name, com_en_name, image_url, title, id, author_url, license_url, date_created)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (sci, com, img_url, com, str(idx + 1000), "https://wikipedia.org", "https://creativecommons.org/licenses/by-sa/4.0/", today_str))

# Insert detections for today
for i in range(25):
    sci, com, conf, _ = birds[i % len(birds)]
    t = (today - timedelta(minutes=i * 12)).strftime("%H:%M:%S")
    fname = f"birdnet-{today_str}-{t.replace(':', '-')}.wav"
    cur.execute("""
    INSERT INTO detections (Date, Time, Sci_Name, Com_Name, Confidence, Lat, Lon, Cutoff, Week, Sens, Overlap, File_Name)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (today_str, t, sci, com, conf, 40.0, -75.0, 0.7, 40, 1.0, 0.0, fname))

# Insert some historical detections
for day_offset in range(1, 10):
    d_str = (today - timedelta(days=day_offset)).strftime("%Y-%m-%d")
    for i in range(10):
        sci, com, conf, _ = birds[i % len(birds)]
        t = (today - timedelta(hours=i)).strftime("%H:%M:%S")
        fname = f"birdnet-{d_str}-{t.replace(':', '-')}.wav"
        cur.execute("""
        INSERT INTO detections (Date, Time, Sci_Name, Com_Name, Confidence, Lat, Lon, Cutoff, Week, Sens, Overlap, File_Name)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (d_str, t, sci, com, conf, 40.0, -75.0, 0.7, 40, 1.0, 0.0, fname))

conn.commit()
conn.close()
print(f"Successfully seeded development database at {db_path}")
