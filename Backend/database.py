import sqlite3

conn = sqlite3.connect('smart_da.db',check_same_thread=False)
cursor = conn.cursor()

#create dataset if doesnot exist 
def create_db():
    cursor.execute(""" 
            CREATE TABLE IF NOT EXISTS Datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_name TEXT,
            dataset_file TEXT,
            description TEXT,
            audio_path TEXT
                   )""")
    conn.commit()
    conn.close()



def get_connection():
    conn = sqlite3.connect('smart_da.db')
    cursor = conn.cursor()
    return conn , cursor


