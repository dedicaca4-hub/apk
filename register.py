"""
register.py
Skrip pendaftaran wajah (Face Enrollment).
Memungkinkan pengguna mendaftarkan wajah baru ke database baik via kamera langsung
maupun mengimpor file foto yang sudah ada.

Penggunaan:
    1. Menu Interaktif:
       python register.py

    2. Ambil foto langsung via kamera:
       python register.py --name "Budi Santoso"

    3. Ambil dari kamera HP:
       python register.py --name "Siti Rahma" --source http://192.168.1.50:8080/video

    4. Impor dari file foto di laptop:
       python register.py --import "C:/Users/.../foto.jpg" --name "Joko"
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from face_system import FaceSystem


def capture_from_camera(source, person_name: str, targets_dir: str = "targets", model_name: str = "buffalo_sc"):
    """Membuka kamera dengan preview interaktif untuk mengambil foto wajah terbaik."""
    print(f"\n[INFO] Menginisialisasi model InsightFace ({model_name})...")
    face_system = FaceSystem(model_name=model_name, targets_dir=targets_dir)

    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print(f"[ERROR] Gagal membuka kamera sumber: {source}")
        return False

    window_name = f"Pendaftaran Wajah: {person_name} (Tekan SPASI untuk Ambil Foto)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"\n[READY] Kamera aktif!")
    print("Instruksi:")
    print("  • Posisikan wajah Anda di tengah layar dan hadap ke depan.")
    print("  • Tekan [SPASI] untuk mengambil foto.")
    print("  • Tekan [Q] atau [ESC] untuk membatalkan.\n")

    captured = False
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        h, w, _ = frame.shape
        display = frame.copy()

        # Deteksi wajah secara live untuk memberi panduan
        faces = face_system.app.get(frame)
        status_color = (0, 0, 255)
        status_text = "Mencari wajah..."

        if len(faces) == 1:
            status_color = (0, 255, 0)
            status_text = "Wajah terdeteksi! Tekan [SPASI] untuk Ambil Foto"
            # Gambar kotak panduan
            f = faces[0]
            x1, y1, x2, y2 = f.bbox.astype(int)
            cv2.rectangle(display, (x1, y1), (x2, y2), status_color, 2)
        elif len(faces) > 1:
            status_color = (0, 165, 255)
            status_text = "Peringatan: Terdeteksi lebih dari 1 wajah!"

        # Banner info atas
        cv2.rectangle(display, (0, 0), (w, 50), (20, 20, 20), -1)
        cv2.putText(display, f"Mendaftarkan: {person_name}", (15, 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display, status_text, (15, 43), cv2.FONT_HERSHEY_DUPLEX, 0.45, status_color, 1)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF

        if key in [ord("q"), ord("Q"), 27]:
            print("[INFO] Pendaftaran dibatalkan.")
            break
        elif key == 32:  # SPASI
            if len(faces) == 0:
                print("[WARNING] Tidak ada wajah terdeteksi! Mohon hadap ke kamera.")
                continue
            if len(faces) > 1:
                print("[WARNING] Terdeteksi lebih dari 1 orang! Pastikan hanya ada Anda di depan kamera.")
                continue

            ok, msg = face_system.register_face_from_frame(frame, person_name)
            print(f"\n[{'BERHASIL' if ok else 'GAGAL'}] {msg}")
            if ok:
                captured = True
                # Tampilkan efek flash visual sebentar
                flash = np.full_like(display, 255)
                cv2.imshow(window_name, flash)
                cv2.waitKey(150)
                time.sleep(0.5)
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured


def import_from_file(file_path: str, person_name: str, targets_dir: str = "targets", model_name: str = "buffalo_sc"):
    """Mengimpor foto yang sudah ada di komputer dan memverifikasi apakah ada wajah."""
    path = Path(file_path)
    if not path.is_file():
        print(f"[ERROR] File tidak ditemukan: {file_path}")
        return False

    print(f"[INFO] Memverifikasi deteksi wajah pada file: {path.name}...")
    face_system = FaceSystem(model_name=model_name, targets_dir=targets_dir)

    with open(path, "rb") as f:
        img_bytes = bytearray(f.read())
    img = cv2.imdecode(np.asarray(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        print("[ERROR] File bukan gambar yang valid.")
        return False

    faces = face_system.app.get(img)
    if len(faces) == 0:
        print("[ERROR] Wajah tidak ditemukan pada gambar tersebut.")
        return False

    dest_dir = Path(targets_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / f"{person_name.replace(' ', '_')}{path.suffix.lower()}"

    shutil.copy(str(path), str(dest_file))
    print(f"[BERHASIL] Foto berhasil diimpor sebagai: {dest_file.name}")
    print(f"Total wajah terdeteksi pada foto: {len(faces)}")
    return True


def list_registered_faces(targets_dir: str = "targets"):
    """Menampilkan daftar orang yang saat ini terdaftar di folder targets."""
    p = Path(targets_dir)
    if not p.exists():
        print("[INFO] Belum ada database targets.")
        return

    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    registered = set()

    for item in p.iterdir():
        if item.is_file() and item.suffix.lower() in valid_ext:
            registered.add(item.stem.replace("_", " "))
        elif item.is_dir():
            registered.add(item.name.replace("_", " "))

    print("\n" + "=" * 50)
    print(f"  DAFTAR WAJAH TERDAFTAR ({len(registered)} Orang)")
    print("=" * 50)
    if not registered:
        print("  (Database masih kosong)")
    else:
        for i, name in enumerate(sorted(registered), 1):
            print(f"  {i}. {name}")
    print("=" * 50 + "\n")


def interactive_menu():
    """Tampilan menu interaktif jika skrip dijalankan tanpa argumen."""
    while True:
        print("\n" + "=" * 50)
        print("       MENU PENDAFTARAN WAJAH (ENROLLMENT)")
        print("=" * 50)
        print("  1. Ambil foto langsung dari Webcam / Kamera HP")
        print("  2. Impor foto dari file di laptop/komputer")
        print("  3. Lihat daftar wajah yang sudah terdaftar")
        print("  4. Keluar")
        print("=" * 50)
        choice = input("Pilih menu (1-4): ").strip()

        if choice == "1":
            name = input(">> Masukkan Nama Orang: ").strip()
            if not name:
                print("[ERROR] Nama tidak boleh kosong!")
                continue
            source = input(">> Sumber kamera (tekan Enter untuk Webcam laptop default / isi URL HP): ").strip()
            if not source:
                source = "0"
            capture_from_camera(source, name)

        elif choice == "2":
            file_path = input(">> Masukkan path file foto (misal: C:/foto/budi.jpg): ").strip(' "\'')
            if not file_path:
                continue
            name = input(">> Masukkan Nama Orang untuk foto ini: ").strip()
            if not name:
                print("[ERROR] Nama tidak boleh kosong!")
                continue
            import_from_file(file_path, name)

        elif choice == "3":
            list_registered_faces()

        elif choice == "4":
            print("[INFO] Keluar dari menu.")
            break
        else:
            print("[WARNING] Pilihan tidak valid.")


def main():
    parser = argparse.ArgumentParser(description="Tool Pendaftaran Wajah InsightFace")
    parser.add_argument("--name", type=str, default=None, help="Nama orang yang akan didaftarkan")
    parser.add_argument("--source", type=str, default="0", help="Sumber kamera (0 untuk webcam, atau URL HP)")
    parser.add_argument("--import", dest="import_path", type=str, default=None, help="Path file foto untuk diimpor")
    parser.add_argument("--targets", type=str, default="targets", help="Direktori database target")
    parser.add_argument("--list", action="store_true", help="Tampilkan daftar wajah terdaftar")

    args = parser.parse_args()

    if args.list:
        list_registered_faces(args.targets)
    elif args.import_path:
        if not args.name:
            args.name = Path(args.import_path).stem
        import_from_file(args.import_path, args.name, args.targets)
    elif args.name:
        capture_from_camera(args.source, args.name, args.targets)
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
