import sqlite3
import pandas as pd
import zipfile
import io
import os

conn = sqlite3.connect('eng_subtitles_database.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
print(cursor.fetchall())
print(".....................................")


cursor.execute("PRAGMA table_info('zipfiles')")
cols = cursor.fetchall()
for col in cols:
    print(col[1])

print(".....................................")
 
df = pd.read_sql_query("""SELECT * FROM zipfiles LIMIT 100""", conn)

for i in range(100):
    binary_data = df.iloc[i, 2]

    # Decompress the binary data using the zipfile module
    with io.BytesIO(binary_data) as f:
        with zipfile.ZipFile(f, 'r') as zip_file:
            # Reading only one file in the ZIP archive
            subtitle_content = zip_file.read(zip_file.namelist()[0])

    # Now 'subtitle_content' should contain the extracted subtitle content
    df.loc[i,"file_content"]=subtitle_content.decode('latin-1')
    
    
column_name = "file_content"

# Directory to save SRT files
output_dir = "subtitles2"
os.makedirs(output_dir, exist_ok=True)

# Save only the required column's data as .srt files
for index, row in df.iterrows():
    file_path = os.path.join(output_dir, f"subtitle_{index+1}.srt")  # Naming files sequentially
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(str(row[column_name]))  

print(f"Saved {len(df)} subtitle files in '{output_dir}' directory.")