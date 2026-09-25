"""
app.py
Aplikasi Real-Time Face Recognition menggunakan InsightFace.
Mendukung webcam laptop bawaan maupun kamera HP via IP Webcam / RTSP stream.

Penggunaan:
    Webcam laptop default:
        python app.py

    Kamera HP (IP Webcam Android / iOS):
        python app.py --source http://192.168.1.50:50050/video

    Menggunakan model buffalo_l untuk akurasi maksimal:
        python app.py --model buffalo_l --threshold 0.55
"""

import argparse
import os
import sys
import time
from typing import Optional

import cv2
import numpy as np
from face_system import FaceSystem

# Coba import dialog GUI untuk registrasi nama cepat
try:
    import tkinter as tk
    from tkinter import simpledialog
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False


def ask_person_name_gui() -> Optional[str]:
    """Membuka dialog popup kecil untuk memasukkan nama saat tombol 'S' ditekan."""
    if HAS_TKINTER:
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            name = simpledialog.askstring("Pendaftaran Wajah", "Masukkan Nama Lengkap:")
            root.destroy()
            return name
        except Exception:
            pass

    # Fallback ke terminal jika dialog GUI gagal/tidak tersedia
    print("\n" + "=" * 40)
    name = input(">> Masukkan Nama untuk Wajah ini: ")
    print("=" * 40)
    return name


def apply_rotation(frame: np.ndarray, angle: int) -> np.ndarray:
    """Merotasi frame sesuai derajat (90, 180, 270, atau -90)."""
    norm_angle = angle % 360
    if norm_angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif norm_angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif norm_angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="FaceAI - Aplikasi Real-Time Face Recognition & Age Detector (InsightFace)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Sumber video: '0' untuk webcam bawaan laptop, atau URL seperti 'http://192.168.1.50:50050/video' untuk kamera HP",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="buffalo_sc",
        choices=["buffalo_sc", "buffalo_l"],
        help="Model InsightFace: 'buffalo_sc' (cepat di CPU) atau 'buffalo_l' (sangat presisi + Tebak Umur & Gender)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Batas kemiripan Cosine Similarity (0.45 - 0.65 disarankan). Nilai lebih tinggi = verifikasi lebih ketat",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="targets",
        help="Folder penyimpanan foto referensi wajah terdaftar",
    )
    parser.add_argument(
        "--rotate",
        type=int,
        default=0,
        choices=[0, 90, 180, 270, -90, -180, -270],
        help="Rotasi video dalam derajat: 90, 180, 270, atau -90 (sangat berguna untuk kamera HP portrait)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Gunakan akselerasi GPU (memerlukan CUDA & onnxruntime-gpu)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Ubah resolusi lebar video (opsional, misal 1280 atau 640)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Ubah resolusi tinggi video (opsional, misal 720 atau 480)",
    )
    parser.add_argument(
        "--skip-detection",
        type=int,
        default=0,
        help="Proses deteksi setiap N frame untuk menghemat daya CPU (0 = proses setiap frame)",
    )
    parser.add_argument(
        "--hide-age",
        action="store_true",
        help="Sembunyikan estimasi usia wajah pada tampilan kamera",
    )
    parser.add_argument(
        "--hide-gender",
        action="store_true",
        help="Sembunyikan estimasi jenis kelamin pada tampilan kamera",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("\n" + "=" * 60)
    print("             FaceAI - REAL-TIME FACE RECOGNITION")
    print("=" * 60)
    print(f"• Model        : {args.model}")
    print(f"• Threshold    : {args.threshold}")
    print(f"• Sumber Video : {args.source}")
    print(f"• Database Dir : {args.targets}/")
    print(f"• Akselerasi   : {'GPU (CUDA)' if args.gpu else 'CPU'}")
    print(f"• Estimasi Usia: {'Nonaktif' if args.hide_age else 'Aktif'}")
    current_rotation = args.rotate % 360
    if current_rotation != 0:
        print(f"• Rotasi Awal  : {current_rotation}° (dapat ditekan 'O' untuk memutar)")
    print("=" * 60 + "\n")

    # Inisialisasi FaceSystem
    try:
        face_system = FaceSystem(
            model_name=args.model,
            threshold=args.threshold,
            targets_dir=args.targets,
            use_gpu=args.gpu,
        )
    except Exception as e:
        print(f"[FATAL] Gagal menginisialisasi FaceSystem: {e}")
        sys.exit(1)

    # Parsing sumber kamera (integer untuk webcam atau string untuk URL)
    source = int(args.source) if args.source.isdigit() else args.source
    source_label = "Webcam Laptop" if str(source) == "0" else ("Kamera " + str(source) if isinstance(source, int) else "Kamera HP / Stream")

    print(f"[INFO] Membuka kamera ({source_label})...")
    cap = cv2.VideoCapture(source)

    # Set resolusi jika diberikan
    if args.width and args.height and isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"\n[ERROR] Tidak dapat mengakses sumber video: {args.source}")
        if isinstance(source, int):
            print("Tips:")
            print("1. Pastikan webcam laptop tidak sedang digunakan oleh aplikasi lain (Zoom, Teams, Kamera Windows).")
            print("2. Coba ganti indeks kamera: python app.py --source 1")
        else:
            print("Tips Kamera HP:")
            print("1. Pastikan HP dan laptop terhubung ke jaringan WiFi yang sama.")
            print("2. Pastikan server aplikasi di HP (seperti IP Webcam) sudah di-Start.")
            print(f"3. Buka URL {args.source} di browser laptop untuk menguji apakah streaming aktif.")
        sys.exit(1)

    window_name = f"FaceAI ({args.model})"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n[READY] Aplikasi berjalan!")
    print("Navigasi Tombol:")
    print("  • [Q] atau [ESC] : Keluar")
    print("  • [S]           : Daftarkan wajah yang sedang ada di layar")
    print("  • [A]           : Tampilkan / Sembunyikan estimasi usia wajah")
    print("  • [G]           : Tampilkan / Sembunyikan jenis kelamin (gender)")
    print("  • [R]           : Reload database wajah dari folder targets/")
    print("  • [O]           : Putar rotasi layar (-90° / 90° live)")
    print("  • [H]           : Tampilkan / Sembunyikan HUD atas")
    print("  • [L]           : Tampilkan / Sembunyikan titik landmark wajah\n")

    fps_tracker = []
    frame_count = 0
    cached_results = []
    show_hud = True
    show_landmarks = True
    show_age = not args.hide_age
    show_gender = not args.hide_gender
    status_message = ""
    status_time = 0.0

    try:
        while True:
            t_start = time.time()
            ret, frame = cap.read()

            if not ret or frame is None:
                print("\n[WARNING] Tidak ada frame dari kamera. Menunggu koneksi kembali...")
                time.sleep(0.5)
                # Coba baca lagi
                ret, frame = cap.read()
                if not ret:
                    print("[INFO] Koneksi video terputus. Keluar.")
                    break

            # Terapkan rotasi jika diatur (misal kamera HP / RTSP portrait)
            if current_rotation != 0:
                frame = apply_rotation(frame, current_rotation)

            # Opsional resize frame jika resolusi terlalu besar untuk mempercepat proses CPU
            if args.width and args.height and not isinstance(source, int):
                frame = cv2.resize(frame, (args.width, args.height))

            # Proses deteksi wajah (setiap frame atau dengan skip frame)
            if args.skip_detection <= 0 or (frame_count % (args.skip_detection + 1) == 0):
                cached_results = face_system.recognize_frame(frame)
            frame_count += 1

            # Hitung rata-rata FPS
            t_elapsed = time.time() - t_start
            current_fps = 1.0 / t_elapsed if t_elapsed > 0 else 30.0
            fps_tracker.append(current_fps)
            if len(fps_tracker) > 15:
                fps_tracker.pop(0)
            avg_fps = sum(fps_tracker) / len(fps_tracker)

            # Render overlay visual
            display_frame = face_system.draw_overlay(
                frame=frame,
                results=cached_results,
                fps=avg_fps,
                source_name=source_label,
                show_landmarks=show_landmarks,
                show_hud=show_hud,
                show_age=show_age,
                show_gender=show_gender,
                rotation=current_rotation,
            )

            # Tampilkan pesan status sementara jika ada (misal setelah registrasi wajah)
            if time.time() - status_time < 3.5 and status_message:
                banner_color = (0, 180, 0) if "berhasil" in status_message.lower() else (0, 0, 220)
                cv2.rectangle(display_frame, (10, display_frame.shape[0] - 65), (display_frame.shape[1] - 10, display_frame.shape[0] - 35), (20, 20, 20), -1)
                cv2.rectangle(display_frame, (10, display_frame.shape[0] - 65), ( display_frame.shape[1] - 10, display_frame.shape[0] - 35), banner_color, 1)
                cv2.putText(
                    display_frame,
                    status_message,
                    (20, display_frame.shape[0] - 44),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.50,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow(window_name, display_frame)

            # Handle input keyboard (wait 1ms)
            key = cv2.waitKey(1) & 0xFF

            if key in [ord("q"), ord("Q"), 27]:  # 27 = ESC
                print("[INFO] Pengguna meminta keluar...")
                break

            elif key in [ord("s"), ord("S")]:
                # Freeze frame sebentar dan minta nama
                name = ask_person_name_gui()
                if name and name.strip():
                    ok, msg = face_system.register_face_from_frame(frame, name)
                    status_message = msg
                    status_time = time.time()
                    print(f"[{'SUCCESS' if ok else 'FAILED'}] {msg}")
                else:
                    print("[INFO] Pendaftaran dibatalkan.")

            elif key in [ord("a"), ord("A")]:
                show_age = not show_age
                status_message = f"Pendeteksi Usia: {'AKTIF' if show_age else 'NONAKTIF'}"
                status_time = time.time()
                print(f"[INFO] {status_message}")

            elif key in [ord("g"), ord("G")]:
                show_gender = not show_gender
                status_message = f"Pendeteksi Gender: {'AKTIF' if show_gender else 'NONAKTIF'}"
                status_time = time.time()
                print(f"[INFO] {status_message}")

            elif key in [ord("r"), ord("R")]:
                count = face_system.reload_targets()
                status_message = f"Database di-reload: {count} wajah aktif"
                status_time = time.time()
                print(f"[INFO] {status_message}")

            elif key in [ord("o"), ord("O")]:
                # Putar rotasi -90 derajat live
                current_rotation = (current_rotation - 90) % 360
                status_message = f"Rotasi layar diubah ke: {current_rotation}°"
                status_time = time.time()
                print(f"[INFO] {status_message}")

            elif key in [ord("h"), ord("H")]:
                show_hud = not show_hud

            elif key in [ord("l"), ord("L")]:
                show_landmarks = not show_landmarks

    except KeyboardInterrupt:
        print("\n[INFO] Dihentikan oleh pengguna (Ctrl+C).")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Kamera dan jendela telah ditutup dengan aman. Selesai.")


if __name__ == "__main__":
    main()
