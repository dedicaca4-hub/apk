"""
face_system.py
Modul inti pengenalan wajah menggunakan InsightFace.
Menangani inisialisasi model, pemindaian database target, kalkulasi kemiripan (cosine similarity),
dan visualisasi overlay hasil deteksi pada frame video.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import insightface
from insightface.app import FaceAnalysis
import numpy as np


class FaceSystem:
    def __init__(
        self,
        model_name: str = "buffalo_sc",
        threshold: float = 0.50,
        targets_dir: str = "targets",
        det_size: Tuple[int, int] = (640, 640),
        use_gpu: bool = False,
    ):
        """
        Inisialisasi sistem pengenalan wajah.
        :param model_name: Nama model InsightFace ('buffalo_sc' untuk CPU/ringan, 'buffalo_l' untuk akurasi maksimal).
        :param threshold: Batas minimal skor similarity (0.0 - 1.0). Di atas threshold dianggap cocok.
        :param targets_dir: Direktori tempat menyimpan foto wajah yang didaftarkan.
        :param det_size: Resolusi input deteksi wajah (default: (640, 640)).
        :param use_gpu: Aktifkan jika memiliki CUDA / onnxruntime-gpu.
        """
        self.model_name = model_name
        self.threshold = threshold
        self.targets_dir = Path(targets_dir)
        self.det_size = det_size
        self.use_gpu = use_gpu

        # Format gambar yang didukung
        self.valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        # Struktur data wajah terdaftar: list dict {"name": str, "embedding": np.ndarray}
        self.known_faces: List[Dict[str, any]] = []

        # Inisialisasi model InsightFace
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        print(f"[INFO] Memuat model InsightFace '{model_name}' (providers: {providers})...")
        allowed_modules = ["detection", "recognition", "genderage"] if model_name == "buffalo_l" else None
        if allowed_modules:
            self.app = FaceAnalysis(name=model_name, providers=providers, allowed_modules=allowed_modules)
        else:
            self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=0, det_size=det_size)
        print("[INFO] Model InsightFace siap digunakan.")

        # Muat database wajah yang tersimpan di folder targets
        self.reload_targets()

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        """Normalisasi vektor embedding ke satuan magnitudo (unit norm) untuk kalkulasi cosine similarity cepat."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def reload_targets(self) -> int:
        """
        Memindai ulang folder targets dan mengekstrak embedding dari setiap foto yang ditemukan.
        Mendukung file langsung: targets/Budi.jpg
        Maupun subfolder: targets/Budi/1.jpg, targets/Budi/2.jpg
        """
        self.known_faces.clear()
        if not self.targets_dir.exists():
            self.targets_dir.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Folder '{self.targets_dir}' dibuat. Silakan tambahkan foto ke folder ini.")
            return 0

        print(f"[INFO] Memindai database wajah dari '{self.targets_dir}'...")
        count = 0

        # Cari file di root targets/
        for item in self.targets_dir.iterdir():
            if item.is_file() and item.suffix.lower() in self.valid_extensions:
                name = item.stem.replace("_", " ").strip()
                if self._register_image_file(item, name):
                    count += 1
            elif item.is_dir():
                # Subfolder: nama folder adalah nama orang
                person_name = item.name.replace("_", " ").strip()
                for sub_item in item.iterdir():
                    if sub_item.is_file() and sub_item.suffix.lower() in self.valid_extensions:
                        if self._register_image_file(sub_item, person_name):
                            count += 1

        print(f"[INFO] Selesai memuat database: {len(self.known_faces)} wajah ({self.get_registered_person_count()} orang unik).")
        return len(self.known_faces)

    def _register_image_file(self, file_path: Path, person_name: str) -> bool:
        """Mengekstrak embedding wajah dari sebuah file gambar."""
        try:
            # Gunakan cv2.imdecode agar mendukung path dengan karakter khusus/spasi di Windows
            with open(file_path, "rb") as f:
                img_bytes = bytearray(f.read())
            img = cv2.imdecode(np.asarray(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

            if img is None:
                print(f"[WARNING] Gagal membaca gambar: {file_path.name}")
                return False

            faces = self.app.get(img)
            if len(faces) == 0:
                print(f"[WARNING] Tidak ada wajah terdeteksi pada {file_path.name}")
                return False

            # Ambil wajah dengan ukuran deteksi terbesar jika ada lebih dari 1 wajah
            best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            embedding = self._normalize(best_face.embedding)

            self.known_faces.append({
                "name": person_name,
                "embedding": embedding,
                "file": file_path.name
            })
            return True
        except Exception as e:
            print(f"[ERROR] Terjadi kesalahan saat memproses {file_path.name}: {e}")
            return False

    def register_face_from_frame(self, frame: np.ndarray, person_name: str) -> Tuple[bool, str]:
        """
        Mendaftarkan wajah langsung dari frame kamera (live enrollment) dan menyimpannya ke folder targets.
        """
        person_name = person_name.strip()
        if not person_name:
            return False, "Nama tidak boleh kosong."

        faces = self.app.get(frame)
        if len(faces) == 0:
            return False, "Wajah tidak terdeteksi di kamera."
        if len(faces) > 1:
            return False, "Terdeteksi lebih dari 1 wajah! Pastikan hanya ada 1 orang di depan kamera."

        face = faces[0]
        embedding = self._normalize(face.embedding)

        # Simpan foto ke folder targets
        safe_name = person_name.replace(" ", "_")
        target_path = self.targets_dir / f"{safe_name}.jpg"

        # Simpan frame asli
        success = cv2.imwrite(str(target_path), frame)
        if not success:
            return False, f"Gagal menyimpan file ke {target_path}"

        # Tambahkan ke database aktif di memori
        self.known_faces.append({
            "name": person_name,
            "embedding": embedding,
            "file": target_path.name
        })

        return True, f"Wajah '{person_name}' berhasil didaftarkan dan disimpan ke {target_path.name}!"

    def get_registered_person_count(self) -> int:
        """Mengembalikan jumlah nama orang unik yang terdaftar."""
        unique_names = {item["name"] for item in self.known_faces}
        return len(unique_names)

    def recognize_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Mendeteksi semua wajah di dalam frame dan mencocokkannya dengan database wajah.
        Mengembalikan list dictionary berisi informasi setiap wajah:
        - bbox: [x1, y1, x2, y2]
        - name: nama orang atau 'Unknown'
        - similarity: skor kemiripan (0.0 - 1.0)
        - is_known: boolean True jika di atas threshold
        - landmarks: 5 titik kunci wajah (mata, hidung, mulut)
        """
        faces = self.app.get(frame)
        results = []

        if len(faces) == 0:
            return results

        # Jika database kosong, semua wajah ditandai Unknown
        if len(self.known_faces) == 0:
            for face in faces:
                results.append({
                    "bbox": face.bbox.astype(int),
                    "name": "Unknown",
                    "similarity": 0.0,
                    "is_known": False,
                    "landmarks": face.kps.astype(int) if face.kps is not None else None,
                })
            return results

        # Matriks embedding database untuk perbandingan vektor cepat
        known_matrix = np.array([item["embedding"] for item in self.known_faces])  # shape: (N, 512)

        for face in faces:
            query_emb = self._normalize(face.embedding)

            # Cosine similarity serentak terhadap seluruh database: dot product
            sims = np.dot(known_matrix, query_emb)
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])
            best_match = self.known_faces[best_idx]

            is_known = best_score >= self.threshold
            label = best_match["name"] if is_known else "Unknown"

            age = int(round(face.age)) if hasattr(face, "age") and face.age is not None else None
            gender = ("L" if face.gender == 1 else "P") if hasattr(face, "gender") and face.gender is not None else None

            results.append({
                "bbox": face.bbox.astype(int),
                "name": label,
                "similarity": max(0.0, min(1.0, best_score)),
                "is_known": is_known,
                "landmarks": face.kps.astype(int) if face.kps is not None else None,
                "age": age,
                "gender": gender,
            })

        return results

    def draw_overlay(
        self,
        frame: np.ndarray,
        results: List[Dict],
        fps: Optional[float] = None,
        source_name: str = "Webcam",
        show_landmarks: bool = True,
        show_hud: bool = True,
    ) -> np.ndarray:
        """
        Menggambar visual bounding box modern, label nama, score similarity, dan HUD status di atas frame.
        """
        out = frame.copy()
        h, w, _ = out.shape

        # Warna (BGR)
        COLOR_KNOWN = (118, 230, 0)     # Hijau Neon / Emerald
        COLOR_UNKNOWN = (60, 60, 240)    # Merah / Coral
        COLOR_BG_TAG = (20, 24, 28)      # Gelap untuk background teks
        COLOR_TEXT = (255, 255, 255)     # Putih
        COLOR_ACCENT = (255, 191, 0)     # Emas / Amber untuk info

        for res in results:
            x1, y1, x2, y2 = res["bbox"]
            name = res["name"]
            score = res["similarity"]
            is_known = res["is_known"]
            landmarks = res["landmarks"]
            age = res.get("age")
            gender = res.get("gender")

            color = COLOR_KNOWN if is_known else COLOR_UNKNOWN

            # Pastikan koordinat dalam batas frame
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            # 1. Gambar Bounding Box dengan aksen sudut (Corner Brackets) bergaya futuristik
            self._draw_corner_box(out, x1, y1, x2, y2, color, thickness=2, corner_len=18)

            # 2. Gambar 5 titik kunci wajah (landmarks) tipis
            if show_landmarks and landmarks is not None:
                for pt in landmarks:
                    cv2.circle(out, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)

            # 3. Label Tag (Nama + Estimasi Umur/Gender + Persentase Kemiripan)
            attr_str = f" [{age}th, {gender}]" if age is not None and gender is not None else ""
            if is_known:
                text = f"{name}{attr_str} ({score * 100:.1f}%)"
            else:
                text = f"Unknown{attr_str} ({score * 100:.1f}%)" if score > 0.1 else f"Unknown{attr_str}"

            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.55
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

            # Posisikan tag di atas bounding box jika muat, jika tidak di bawah
            pad = 6
            tag_y2 = y1 - 6 if y1 - (text_h + pad * 2) > 0 else y2 + (text_h + pad * 2) + 6
            tag_y1 = tag_y2 - (text_h + pad * 2)
            tag_x1 = x1
            tag_x2 = x1 + text_w + pad * 2

            # Background pill/tag transparan
            sub_img = out[max(0, tag_y1):min(h, tag_y2), max(0, tag_x1):min(w, tag_x2)]
            if sub_img.shape[0] > 0 and sub_img.shape[1] > 0:
                overlay = sub_img.copy()
                overlay[:] = COLOR_BG_TAG
                cv2.addWeighted(overlay, 0.75, sub_img, 0.25, 0, sub_img)

            # Garis strip warna di samping tag
            cv2.rectangle(out, (tag_x1, tag_y1), (tag_x1 + 4, tag_y2), color, -1)

            # Teks nama
            text_pos = (tag_x1 + pad + 2, tag_y2 - pad - baseline // 2)
            cv2.putText(out, text, text_pos, font, font_scale, COLOR_TEXT, thickness, cv2.LINE_AA)

        # 4. HUD Banner di bagian atas (Status sistem)
        if show_hud:
            hud_h = 36
            hud_bg = out[0:hud_h, 0:w].copy()
            hud_color = np.full_like(hud_bg, (15, 18, 22), dtype=np.uint8)
            cv2.addWeighted(hud_color, 0.75, hud_bg, 0.25, 0, hud_bg)
            out[0:hud_h, 0:w] = hud_bg

            # Garis pemisah bawah HUD tipis
            cv2.line(out, (0, hud_h), (w, hud_h), (60, 70, 80), 1)

            # Info kiri: FPS & Model
            fps_str = f"FPS: {fps:.1f}" if fps is not None else "FPS: --"
            info_left = f"{fps_str}  |  Model: {self.model_name}  |  Thresh: {self.threshold:.2f}"
            cv2.putText(out, info_left, (12, 23), cv2.FONT_HERSHEY_DUPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

            # Info kanan: Database & Source
            reg_count = self.get_registered_person_count()
            info_right = f"Database: {reg_count} Wajah  |  Sumber: {source_name}"
            (ir_w, _), _ = cv2.getTextSize(info_right, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
            cv2.putText(out, info_right, (w - ir_w - 12, 23), cv2.FONT_HERSHEY_DUPLEX, 0.45, COLOR_ACCENT, 1, cv2.LINE_AA)

            # Footer tipis petunjuk tombol di bawah layar
            footer_text = "[Q] Keluar   [S] Daftar Wajah   [R] Reload DB   [H] Toggle HUD"
            (ft_w, _), _ = cv2.getTextSize(footer_text, cv2.FONT_HERSHEY_DUPLEX, 0.40, 1)
            f_y = h - 10
            # Background transparan tipis untuk footer
            cv2.putText(out, footer_text, ((w - ft_w) // 2, f_y), cv2.FONT_HERSHEY_DUPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

        return out

    def _draw_corner_box(
        self,
        img: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: Tuple[int, int, int],
        thickness: int = 2,
        corner_len: int = 16,
    ):
        """Menggambar kotak pembatas dengan aksen sudut tebal futuristik."""
        # Garis tipis kotak penuh
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

        # Panjang sudut tidak boleh melebihi setengah dimensi
        cl_x = min(corner_len, (x2 - x1) // 3)
        cl_y = min(corner_len, (y2 - y1) // 3)

        t = thickness
        # Sudut Kiri Atas
        cv2.line(img, (x1, y1), (x1 + cl_x, y1), color, t, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1, y1 + cl_y), color, t, cv2.LINE_AA)

        # Sudut Kanan Atas
        cv2.line(img, (x2, y1), (x2 - cl_x, y1), color, t, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2, y1 + cl_y), color, t, cv2.LINE_AA)

        # Sudut Kiri Bawah
        cv2.line(img, (x1, y2), (x1 + cl_x, y2), color, t, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1, y2 - cl_y), color, t, cv2.LINE_AA)

        # Sudut Kanan Bawah
        cv2.line(img, (x2, y2), (x2 - cl_x, y2), color, t, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2, y2 - cl_y), color, t, cv2.LINE_AA)
