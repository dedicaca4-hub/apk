"""
configure_android.py
Helper script untuk mengkonfigurasi AndroidManifest.xml di Capacitor project
menambahkan izin kamera dan hardware camera.
"""

import sys
from pathlib import Path

manifest_path = Path("android/app/src/main/AndroidManifest.xml")

if not manifest_path.exists():
    print(f"[WARNING] {manifest_path} belum ada.")
    sys.exit(0)

with open(manifest_path, "r", encoding="utf-8") as f:
    content = f.read()

permissions_block = """    <uses-permission android:name="android.permission.CAMERA" />
    <uses-permission android:name="android.permission.INTERNET" />
    <uses-feature android:name="android.hardware.camera" android:required="false" />
    <uses-feature android:name="android.hardware.camera.autofocus" android:required="false" />
<application"""

if "android.permission.CAMERA" not in content:
    content = content.replace("<application", permissions_block)
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("[INFO] AndroidManifest.xml berhasil dikonfigurasi dengan izin Kamera!")
else:
    print("[INFO] Izin kamera sudah ada di AndroidManifest.xml.")
