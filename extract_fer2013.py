import zipfile
import os

# Specify the path to the ZIP file
zip_path = 'C:\\Users\\lucky\\facial_expression_recognition\\fer2013\\fer2013.zip'
extract_to = 'C:\\Users\\lucky\\facial_expression_recognition\\fer2013\\data'

# Check if the ZIP file exists
if os.path.exists(zip_path):
    print("Zip file found. Proceeding with extraction...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print(f"Extraction completed. Files are extracted to {extract_to}")
else:
    print("Zip file not found. Please check the file path.")
# Returns False because the first key is false.
# For dictionaries the all() function checks the keys, not the values.