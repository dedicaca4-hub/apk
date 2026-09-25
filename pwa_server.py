"""
pwa_server.py
Server HTTPS & HTTP lokal untuk menjalankan Progressive Web App (PWA) FaceAI.
Memungkinkan smartphone terhubung via Wi-Fi lokal dan mendapatkan akses kamera secara legal di browser mobile.

Penggunaan:
    python pwa_server.py
    python pwa_server.py --port 50050
    python pwa_server.py --http   (mode HTTP biasa tanpa SSL)
"""

import argparse
import base64
import http.server
import json
import os
import socket
import ssl
import subprocess
import sys
from pathlib import Path


def get_local_ip() -> str:
    """Mendapatkan alamat IP lokal (LAN/Wi-Fi) laptop."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def ensure_ssl_cert(pwa_dir: Path) -> tuple[Path, Path]:
    """Memastikan sertifikat self-signed SSL cert.pem dan key.pem tersedia."""
    cert_file = pwa_dir / "cert.pem"
    key_file = pwa_dir / "key.pem"

    if cert_file.exists() and key_file.exists():
        return cert_file, key_file

    # Cari openssl di instalasi Git for Windows atau system PATH
    openssl_candidates = [
        r"C:\Users\PC\AppData\Local\hermes\git\usr\bin\openssl.exe",
        r"C:\Users\PC\AppData\Local\hermes\git\mingw64\bin\openssl.exe",
        "openssl",
    ]

    openssl_bin = None
    for cand in openssl_candidates:
        if os.path.exists(cand):
            openssl_bin = cand
            break

    if openssl_bin:
        print("[INFO] Membuat sertifikat SSL lokal otomatis...")
        cmd = [
            openssl_bin, "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", str(key_file), "-out", str(cert_file),
            "-days", "365", "-nodes",
            "-subj", "/CN=ipweb-PWA"
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("[INFO] Sertifikat SSL berhasil dibuat.")
            return cert_file, key_file
        except Exception as e:
            print(f"[WARNING] Gagal membuat sertifikat via openssl: {e}")

    return None, None


class PWAHandler(http.server.SimpleHTTPRequestHandler):
    """Handler khusus untuk melayani file PWA dan endpoint sinkronisasi target."""

    def __init__(self, *args, **kwargs):
        pwa_dir = str(Path(__file__).parent / "pwa")
        super().__init__(*args, directory=pwa_dir, **kwargs)

    def end_headers(self):
        # Header penting untuk WebAssembly dan PWA
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

    def guess_type(self, path):
        # Pastikan MIME types untuk .onnx, .wasm, dan .json benar
        if str(path).endswith(".onnx"):
            return "application/octet-stream"
        elif str(path).endswith(".wasm"):
            return "application/wasm"
        elif str(path).endswith(".json") or str(path).endswith(".webmanifest"):
            return "application/manifest+json"
        return super().guess_type(path)

    def do_GET(self):
        # API untuk membaca database foto yang ada di folder targets/ laptop
        if self.path == "/api/targets":
            self.handle_api_targets()
            return
        super().do_GET()

    def handle_api_targets(self):
        targets_dir = Path(__file__).parent / "targets"
        valid_ext = {".jpg", ".jpeg", ".png", ".webp"}
        targets_data = []

        if targets_dir.exists():
            emb_map = {}
            try:
                from face_system import FaceSystem
                fs = FaceSystem(model_name="buffalo_sc", targets_dir=str(targets_dir))
                for kf in fs.known_faces:
                    emb_map[kf["file"]] = kf["embedding"].tolist()
            except Exception as e:
                print(f"[WARNING] Gagal mengekstrak embedding untuk /api/targets: {e}")

            for item in targets_dir.iterdir():
                if item.is_file() and item.suffix.lower() in valid_ext:
                    name = item.stem.replace("_", " ")
                    try:
                        with open(item, "rb") as f:
                            b64 = base64.b64encode(f.read()).decode("utf-8")
                        mime = "image/png" if item.suffix.lower() == ".png" else "image/jpeg"
                        targets_data.append({
                            "name": name,
                            "photo": f"data:{mime};base64,{b64}",
                            "embedding": emb_map.get(item.name, []),
                        })
                    except Exception:
                        pass

        body = json.dumps({"targets": targets_data}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description="Server PWA ipweb untuk HP Android & iOS")
    parser.add_argument("--port", type=int, default=50050, help="Port server HTTPS (default: 50050)")
    parser.add_argument("--http", action="store_true", help="Jalankan di mode HTTP biasa tanpa SSL (port default: 50050)")
    args = parser.parse_args()

    port = args.port
    local_ip = get_local_ip()
    pwa_dir = Path(__file__).parent / "pwa"

    use_ssl = not args.http
    cert_file, key_file = None, None

    if use_ssl:
        cert_file, key_file = ensure_ssl_cert(pwa_dir)
        if not cert_file or not cert_file.exists():
            print("[WARNING] File sertifikat SSL tidak ditemukan. Beralih ke mode HTTP.")
            use_ssl = False

    server_address = ("0.0.0.0", port)
    httpd = http.server.ThreadingHTTPServer(server_address, PWAHandler)

    protocol = "https" if use_ssl else "http"
    if use_ssl:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=str(cert_file), keyfile=str(key_file))
        httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
            sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except Exception:
            pass

    print("\n" + "=" * 65)
    print("      IPWEB - PWA STANDALONE ON-DEVICE MOBILE SERVER")
    print("=" * 65)
    print(f"* Status Server   : BERJALAN ({protocol.upper()})")
    print(f"* Alamat Lokal PC : {protocol}://localhost:{port}")
    print(f"* Alamat HP (LAN) : {protocol}://{local_ip}:{port}")
    print("=" * 65)
    print("\n[CARA MENGHUBUNGKAN DARI HP ANDROID]")
    print(f"1. Pastikan HP dan Laptop terhubung ke Wi-Fi / Hotspot yang sama.")
    print(f"2. Buka Google Chrome di HP Anda.")
    print(f"3. Masukkan alamat URL berikut:")
    print(f"\n      >>   {protocol}://{local_ip}:{port}   <<\n")
    if use_ssl:
        print("4. Jika muncul peringatan 'Your connection is not private' (karena SSL lokal):")
        print("   - Klik 'Advanced' (Lanjutan)")
        print(f"   - Pilih 'Proceed to {local_ip} (unsafe)'")
    print("5. Izinkan akses kamera HP saat muncul pop-up izin.")
    print("6. Tekan tombol menu Chrome (titik 3 di kanan atas) -> Pilih 'Tambahkan ke Layar Utama' (Install App).")
    print("   Aplikasi ipweb akan terpasang di HP Anda dan dapat dibuka seperti aplikasi APK!")
    print("\nTekan Ctrl+C di terminal ini untuk mematikan server.")
    print("-" * 65 + "\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Server PWA dihentikan.")
        httpd.server_close()
        sys.exit(0)


if __name__ == "__main__":
    main()
