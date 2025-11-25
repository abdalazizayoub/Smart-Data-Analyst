# database.py
import sqlite3

def create_db():
    conn = sqlite3.connect('smart_da.db')
    cursor = conn.cursor()
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
    print("Database table created successfully")

def get_connection():
    conn = sqlite3.connect('smart_da.db')
    cursor = conn.cursor()
    return conn, cursor