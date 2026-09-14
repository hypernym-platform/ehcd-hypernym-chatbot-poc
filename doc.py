import sqlite3
import os

class Documents:
    def __init__(self, db_path=None, directory_path='doc-policy'):
        # Default under DATA_ROOT (a mounted volume) rather than the image, which
        # is read-only in Kubernetes. `directory_path` is read-only and stays put.
        self.db_path = db_path or os.getenv(
            "DOCUMENTS_DB_PATH",
            os.path.join(os.getenv("DATA_ROOT", "./data"), "documents.db"),
        )
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self.directory_path = directory_path
        self.init_db()  

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                path TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()

    def save_local_files_to_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM documents')  

        for filename in os.listdir(self.directory_path):
            file_path = os.path.join(self.directory_path, filename)
            if os.path.isfile(file_path):
                cursor.execute('INSERT INTO documents (name, path) VALUES (?, ?)', (filename, file_path))
        
        conn.commit()
        conn.close()

    def fetch_documents(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, path FROM documents')
        documents = cursor.fetchall()
        conn.close()
        return documents

    def get_document_path(self, document_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT name, path FROM documents WHERE id = ?', (document_id,))
        document = cursor.fetchone()
        conn.close()
        return document
