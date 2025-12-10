import os
import shutil
from pathlib import Path

# Standard ultralytics settings path on Windows
config_dir = Path(os.environ['APPDATA']) / 'Ultralytics'
config_file = config_dir / 'settings.yaml'

print(f"Checking for config at: {config_file}")

if config_file.exists():
    print("Found settings.yaml. Deleting it to force reset...")
    try:
        os.remove(config_file)
        print("Deleted settings.yaml successfully.")
    except Exception as e:
        print(f"Failed to delete settings.yaml: {e}")
else:
    print("settings.yaml not found.")

# Also check for fonts just in case
font_file = config_dir / 'Arial.ttf'
if font_file.exists():
    try:
        os.remove(font_file)
        print("Deleted cached font file.")
    except:
        pass

print("Ultralytics config reset complete.")
