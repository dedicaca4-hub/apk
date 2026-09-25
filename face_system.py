"""
face_system.py
Modul inti pengenalan wajah menggunakan InsightFace.
Menangani inisialisasi model, pemindaian database target, kalkulasi kemiripan (cosine similarity),
stabilisasi tracking (usia, gender, identitas), dan visualisasi overlay hasil deteksi pada frame video.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

# Redam warning dari dependensi scikit-image pada insightface.face_align
warnings.filterwarnings("ignore", category=FutureWarning)

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
        det_thresh: float = 0.45,
        use_gpu: bool = False,
    ):
        """
        Inisialisasi sistem pengenalan wajah.
        :param model_name: Nama model InsightFace ('buffalo_sc' untuk CPU/ringan, 'buffalo_l' untuk akurasi maksimal).
        :param threshold: Batas minimal skor similarity (0.0 - 1.0). Di atas threshold dianggap cocok.
        :param targets_dir: Direktori tempat menyimpan foto wajah yang didaftarkan.
        :param det_size: Resolusi input deteksi wajah (default: (640, 640)).
        :param det_thresh: Batas sensitivitas detektor wajah (default: 0.45 untuk akurasi pose/pencahayaan variatif).
        :param use_gpu: Aktifkan jika memiliki CUDA / onnxruntime-gpu.
        """
        self.model_name = model_name
        self.threshold = threshold
        self.targets_dir = Path(targets_dir)
        self.det_size = det_size
        self.det_thresh = det_thresh
        self.use_gpu = use_gpu

        # Format gambar yang didukung
        self.valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        # Database wajah: list sample individu dan kamus centroid fitur per orang
        self.known_faces: List[Dict[str, any]] = []
        self.known_centroids: Dict[str, np.ndarray] = {}

        # Tracker serbaguna untuk menstabilkan estimasi usia, gender, dan identitas antar-frame
        self._tracks: Dict[int, Dict[str, any]] = {}
        self._next_track_id: int = 0
        self._frame_index: int = 0

        # Pastikan modul estimasi usia & gender tersedia (hanya ~1.3 MB)
        self._ensure_genderage_model(model_name)

        # Inisialisasi model InsightFace
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        print(f"[INFO] Memuat model InsightFace '{model_name}' (providers: {providers})...")
        allowed_modules = ["detection", "recognition", "genderage"]
        self.app = FaceAnalysis(name=model_name, providers=providers, allowed_modules=allowed_modules)
        self.app.prepare(ctx_id=0, det_thresh=det_thresh, det_size=det_size)
        print("[INFO] Model InsightFace siap digunakan (Deteksi Wajah, Pengenalan, dan Estimasi Usia aktif).")

        # Muat database wajah yang tersimpan di folder targets
        self.reload_targets()

    @staticmethod
    def _ensure_genderage_model(model_name: str) -> bool:
        """
        Memastikan bobot model genderage.onnx (~1.3 MB) tersedia di folder model InsightFace.
        Model ini memungkinkan estimasi usia dan gender bekerja bahkan pada model ringan (buffalo_sc).
        """
        model_dir = Path.home() / ".insightface" / "models" / model_name
        ga_file = model_dir / "genderage.onnx"
        if ga_file.exists():
            return True

        model_dir.mkdir(parents=True, exist_ok=True)
        url = "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/genderage.onnx"
        print(f"[INFO] Mengunduh modul estimasi usia wajah (~1.3 MB) untuk '{model_name}'...")
        try:
            import urllib.request
            urllib.request.urlretrieve(url, str(ga_file))
            print(f"[INFO] Modul usia berhasil diunduh ke: {ga_file.name}")
            return True
        except Exception as e:
            print(f"[WARNING] Tidak dapat mengunduh model usia otomatis ({e}). Estimasi usia mungkin tidak aktif.")
            return False

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        """Normalisasi vektor embedding ke satuan magnitudo (unit norm) untuk kalkulasi cosine similarity cepat."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def _rebuild_centroids(self):
        """Membangun centroid embedding (vektor rata-rata ternormalisasi) untuk setiap nama orang terdaftar."""
        name_groups: Dict[str, List[np.ndarray]] = {}
        for item in self.known_faces:
            name_groups.setdefault(item["name"], []).append(item["embedding"])

        self.known_centroids.clear()
        for name, embs in name_groups.items():
            mean_vec = np.mean(embs, axis=0)
            self.known_centroids[name] = self._normalize(mean_vec)

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

        for item in self.targets_dir.iterdir():
            if item.is_file() and item.suffix.lower() in self.valid_extensions:
                name = item.stem.replace("_", " ").strip()
                if self._register_image_file(item, name):
                    count += 1
            elif item.is_dir():
                person_name = item.name.replace("_", " ").strip()
                for sub_item in item.iterdir():
                    if sub_item.is_file() and sub_item.suffix.lower() in self.valid_extensions:
                        if self._register_image_file(sub_item, person_name):
                            count += 1

        self._rebuild_centroids()
        print(f"[INFO] Selesai memuat database: {len(self.known_faces)} sampel wajah ({self.get_registered_person_count()} orang unik).")
        return len(self.known_faces)

    def _register_image_file(self, file_path: Path, person_name: str) -> bool:
        """Mengekstrak embedding wajah dari sebuah file gambar."""
        try:
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
        Mendaftarkan wajah langsung dari frame kamera (live enrollment) dengan validasi kualitas.
        Menyimpan foto aman di Windows dan mengupdate database aktif secara instan.
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

        # Validasi kualitas deteksi untuk memastikan akurasi maksimal
        w = face.bbox[2] - face.bbox[0]
        h = face.bbox[3] - face.bbox[1]
        if w < 70 or h < 70:
            return False, "Ukuran wajah terlalu kecil/jauh di kamera. Silakan mendekat ke kamera."

        det_score = getattr(face, "det_score", 1.0)
        if det_score < 0.55:
            return False, "Kualitas deteksi wajah rendah (pencahayaan kurang / wajah miring). Posisikan wajah tegak lurus."

        embedding = self._normalize(face.embedding)

        # Simpan foto ke folder targets secara aman di Windows (mendukung Unicode/spasi)
        safe_name = person_name.replace(" ", "_")
        target_path = self.targets_dir / f"{safe_name}.jpg"

        is_ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not is_ok:
            return False, "Gagal meng-encode frame foto ke format JPEG."

        try:
            with open(target_path, "wb") as f:
                f.write(buf)
        except Exception as e:
            return False, f"Gagal menyimpan file ke {target_path.name}: {e}"

        # Perbarui memori: hapus data lama jika nama yang sama didaftarkan ulang
        self.known_faces = [item for item in self.known_faces if item["name"] != person_name]
        self.known_faces.append({
            "name": person_name,
            "embedding": embedding,
            "file": target_path.name
        })
        self._rebuild_centroids()

        return True, f"Wajah '{person_name}' berhasil didaftarkan dan disimpan ke {target_path.name}!"

    def get_registered_person_count(self) -> int:
        """Mengembalikan jumlah nama orang unik yang terdaftar."""
        unique_names = {item["name"] for item in self.known_faces}
        return len(unique_names)

    @staticmethod
    def _calc_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
        """Menghitung Intersection over Union (IoU) antara dua bounding box [x1, y1, x2, y2]."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return float(iou)

    def _match_or_create_track(
        self,
        bbox: np.ndarray,
        raw_age: Optional[float],
        raw_gender: Optional[int],
        detected_name: str,
        match_score: float,
        is_match: bool,
    ) -> Tuple[Optional[int], Optional[str], str, float, bool]:
        """
        Asosiasi multi-frame tracking untuk stabilisasi usia, gender, dan identitas pengenalan.
        Mencegah flicker nama antar-frame akibat goyangan kamera atau perubahan pose sesaat.
        """
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        diag = max(1.0, np.sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2))

        best_id = None
        best_score = -1.0

        for track_id, data in self._tracks.items():
            prev_bbox = data["bbox"]
            iou = self._calc_iou(bbox, prev_bbox)
            prev_cx = (prev_bbox[0] + prev_bbox[2]) / 2.0
            prev_cy = (prev_bbox[1] + prev_bbox[3]) / 2.0
            center_dist = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) / diag

            # Kriteria pencocokan: IoU signifikan atau pergeseran pusat wajah wajar
            if iou > 0.25 or (iou > 0.10 and center_dist < 0.35):
                track_match_score = iou - center_dist * 0.5
                if track_match_score > best_score:
                    best_score = track_match_score
                    best_id = track_id

        # Hitung gender numerik (1 = Laki-laki, 0 = Perempuan)
        male_val = 1.0 if raw_gender == 1 else (0.0 if raw_gender == 0 else 0.5)

        if best_id is not None:
            track = self._tracks[best_id]

            # 1. Stabilisasi Usia (EMA 85% riwayat + 15% frame baru)
            if raw_age is not None:
                smoothed_age_val = 0.85 * track["age"] + 0.15 * raw_age
            else:
                smoothed_age_val = track["age"]
            track["age"] = smoothed_age_val
            out_age = int(round(smoothed_age_val))

            # 2. Stabilisasi Gender (EMA probabilitas pria/wanita)
            if raw_gender is not None:
                track["gender_prob"] = 0.85 * track["gender_prob"] + 0.15 * male_val
            out_gender = "L" if track["gender_prob"] >= 0.50 else "P"

            # 3. Stabilisasi Skor Kemiripan
            track["score"] = 0.70 * track["score"] + 0.30 * match_score

            # 4. Stabilisasi Identitas (Menghilangkan flicker Unknown sesaat saat pengguna sudah terkonfirmasi)
            if is_match:
                track["name_history"].append(detected_name)
                track["consecutive_known"] = min(20, track["consecutive_known"] + 1)
            else:
                track["name_history"].append("Unknown")
                track["consecutive_known"] = max(0, track["consecutive_known"] - 1)

            if len(track["name_history"]) > 6:
                track["name_history"].pop(0)

            # Jika track sudah terkonfirmasi sebagai orang yang sama dalam beberapa frame,
            # pertahankan nama meski ada penurunan sesaat (misal menoleh sedikit)
            final_name = detected_name
            final_is_known = is_match
            if not is_match and track["consecutive_known"] >= 2:
                # Periksa nama dominan dalam riwayat recent
                known_names = [n for n in track["name_history"] if n != "Unknown"]
                if known_names and match_score >= (self.threshold - 0.08):
                    final_name = max(set(known_names), key=known_names.count)
                    final_is_known = True

            track["bbox"] = bbox
            track["last_seen"] = self._frame_index
            return out_age, out_gender, final_name, track["score"], final_is_known

        else:
            # Buat track baru
            new_id = self._next_track_id
            self._next_track_id += 1
            init_age = float(raw_age) if raw_age is not None else 25.0
            init_gender_prob = male_val
            self._tracks[new_id] = {
                "bbox": bbox,
                "age": init_age,
                "gender_prob": init_gender_prob,
                "score": match_score,
                "name_history": [detected_name],
                "consecutive_known": 1 if is_match else 0,
                "last_seen": self._frame_index,
            }
            out_age = int(round(init_age)) if raw_age is not None else None
            out_gender = ("L" if init_gender_prob >= 0.50 else "P") if raw_gender is not None else None
            return out_age, out_gender, detected_name, match_score, is_match

    def _cleanup_tracks(self):
        """Membersihkan track yang sudah tidak terlihat lebih dari 45 frame (~1.5 - 2 detik)."""
        stale_ids = [
            tid for tid, d in self._tracks.items()
            if self._frame_index - d["last_seen"] > 45
        ]
        for tid in stale_ids:
            del self._tracks[tid]

    def recognize_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Mendeteksi semua wajah di dalam frame dan mencocokkannya dengan database wajah.
        Menggunakan pencocokan gabungan (Max-Sample + Centroid) dan Multi-Frame Temporal Stabilization.
        """
        self._frame_index += 1  # Increment tepat satu kali per frame video
        faces = self.app.get(frame)
        results = []

        if len(faces) == 0:
            if self._frame_index % 30 == 0:
                self._cleanup_tracks()
            return results

        # Kelompokkan data target per orang untuk pencocokan multi-sampel dan centroid
        known_names = list(self.known_centroids.keys())

        for face in faces:
            bbox_int = face.bbox.astype(int)
            raw_age = float(face.age) if hasattr(face, "age") and face.age is not None else None
            raw_gender = int(face.gender) if hasattr(face, "gender") and face.gender is not None else None

            # Jika database target kosong
            if not known_names:
                age, gender, name, score, is_known = self._match_or_create_track(
                    bbox=bbox_int,
                    raw_age=raw_age,
                    raw_gender=raw_gender,
                    detected_name="Unknown",
                    match_score=0.0,
                    is_match=False,
                )
                results.append({
                    "bbox": bbox_int,
                    "name": "Unknown",
                    "similarity": 0.0,
                    "is_known": False,
                    "landmarks": face.kps.astype(int) if face.kps is not None else None,
                    "age": age,
                    "raw_age": raw_age,
                    "gender": gender,
                })
                continue

            # Query embedding
            query_emb = self._normalize(face.embedding)

            # Hitung skor kemiripan terhadap setiap orang terdaftar
            best_person = "Unknown"
            best_score = -1.0

            for person in known_names:
                # 1. Skor terhadap centroid (representasi rata-rata wajah)
                centroid_sim = float(np.dot(self.known_centroids[person], query_emb))

                # 2. Skor terhadap semua sampel individual orang tersebut
                person_samples = [item["embedding"] for item in self.known_faces if item["name"] == person]
                max_sample_sim = float(np.max(np.dot(person_samples, query_emb)))

                # Nilai kemiripan tertinggi antara sampel spesifik dan centroid umum
                combined_sim = max(max_sample_sim, centroid_sim)
                if combined_sim > best_score:
                    best_score = combined_sim
                    best_person = person

            raw_is_known = best_score >= self.threshold
            detected_label = best_person if raw_is_known else "Unknown"

            # Stabilisasi multi-frame (usia, gender, dan identitas anti-flicker)
            age, gender, final_name, smoothed_score, final_is_known = self._match_or_create_track(
                bbox=bbox_int,
                raw_age=raw_age,
                raw_gender=raw_gender,
                detected_name=detected_label,
                match_score=best_score,
                is_match=raw_is_known,
            )

            results.append({
                "bbox": bbox_int,
                "name": final_name,
                "similarity": max(0.0, min(1.0, smoothed_score)),
                "raw_similarity": max(0.0, min(1.0, best_score)),
                "is_known": final_is_known,
                "landmarks": face.kps.astype(int) if face.kps is not None else None,
                "age": age,
                "raw_age": raw_age,
                "gender": gender,
            })

        if self._frame_index % 30 == 0:
            self._cleanup_tracks()

        return results

    @staticmethod
    def _calc_calibrated_confidence(score: float, threshold: float) -> int:
        """
        Mengonversi skor cosine similarity (biasanya 0.20 - 0.75) ke persentase keyakinan intuitif.
        - Di atas threshold: 70% - 99%
        - Di bawah threshold: 5% - 49%
        """
        if score >= threshold:
            # Skor dari threshold hingga 0.75 dipetakan ke 70% - 99%
            norm_range = max(0.05, 0.75 - threshold)
            scaled = 70.0 + min(29.0, ((score - threshold) / norm_range) * 29.0)
            return int(round(scaled))
        else:
            scaled = (score / max(0.01, threshold)) * 48.0
            return max(5, int(round(scaled)))

    def draw_overlay(
        self,
        frame: np.ndarray,
        results: List[Dict],
        fps: Optional[float] = None,
        source_name: str = "Webcam",
        show_landmarks: bool = True,
        show_hud: bool = True,
        show_age: bool = True,
        show_gender: bool = True,
        rotation: int = 0,
        is_mirrored: bool = False,
    ) -> np.ndarray:
        """
        Menggambar visual bounding box modern, label nama, score similarity, estimasi usia, dan HUD status di atas frame.
        """
        out = frame.copy()
        h, w, _ = out.shape

        # Skema Warna Modern (BGR)
        COLOR_KNOWN = (118, 230, 0)     # Emerald Neon
        COLOR_UNKNOWN = (60, 60, 240)    # Coral Red
        COLOR_BG_TAG = (20, 24, 28)      # Dark Tag Background
        COLOR_TEXT = (255, 255, 255)     # Putih
        COLOR_ACCENT = (255, 191, 0)     # Amber / Gold

        for res in results:
            x1, y1, x2, y2 = res["bbox"]
            name = res["name"]
            score = res["similarity"]
            is_known = res["is_known"]
            landmarks = res["landmarks"]
            age = res.get("age")
            gender = res.get("gender")

            color = COLOR_KNOWN if is_known else COLOR_UNKNOWN

            # Batas koordinat
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            # 1. Gambar Bounding Box futuristik dengan aksen sudut
            self._draw_corner_box(out, x1, y1, x2, y2, color, thickness=2, corner_len=18)

            # 2. Gambar 5 titik kunci wajah (landmarks) tipis
            if show_landmarks and landmarks is not None:
                for pt in landmarks:
                    cv2.circle(out, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)

            # 3. Label Tag (Nama + Usia / Gender + Persentase Keyakinan Terkalibrasi + Cosine Score)
            attr_parts = []
            if show_age and age is not None:
                attr_parts.append(f"{age}th")
            if show_gender and gender is not None:
                attr_parts.append(gender)
            attr_str = f" [{', '.join(attr_parts)}]" if attr_parts else ""

            conf_pct = self._calc_calibrated_confidence(score, self.threshold)

            if is_known:
                text = f"{name}{attr_str} ({conf_pct}% | {score:.2f})"
            else:
                if score > 0.15:
                    text = f"Unknown{attr_str} ({score:.2f})"
                else:
                    text = f"Unknown{attr_str}"

            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.52
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

            # Posisikan tag di atas kotak jika memungkinkan
            pad = 6
            tag_y2 = y1 - 6 if y1 - (text_h + pad * 2) > 0 else y2 + (text_h + pad * 2) + 6
            tag_y1 = tag_y2 - (text_h + pad * 2)
            tag_x1 = x1
            tag_x2 = x1 + text_w + pad * 2

            sub_img = out[max(0, tag_y1):min(h, tag_y2), max(0, tag_x1):min(w, tag_x2)]
            if sub_img.shape[0] > 0 and sub_img.shape[1] > 0:
                overlay = sub_img.copy()
                overlay[:] = COLOR_BG_TAG
                cv2.addWeighted(overlay, 0.75, sub_img, 0.25, 0, sub_img)

            # Aksen warna tag
            cv2.rectangle(out, (tag_x1, tag_y1), (tag_x1 + 4, tag_y2), color, -1)
            text_pos = (tag_x1 + pad + 2, tag_y2 - pad - baseline // 2)
            cv2.putText(out, text, text_pos, font, font_scale, COLOR_TEXT, thickness, cv2.LINE_AA)

        # 4. HUD Banner di bagian atas (Status sistem)
        if show_hud:
            hud_h = 36
            hud_bg = out[0:hud_h, 0:w].copy()
            hud_color = np.full_like(hud_bg, (15, 18, 22), dtype=np.uint8)
            cv2.addWeighted(hud_color, 0.75, hud_bg, 0.25, 0, hud_bg)
            out[0:hud_h, 0:w] = hud_bg

            cv2.line(out, (0, hud_h), (w, hud_h), (60, 70, 80), 1)

            fps_str = f"FPS: {fps:.1f}" if fps is not None else "FPS: --"
            rot_str = f" | Rot: {rotation}°" if rotation != 0 else ""
            mir_str = " | Cermin" if is_mirrored else ""
            age_str = f" | Usia: {'ON' if show_age else 'OFF'}"
            info_left = f"FaceAI | {fps_str} | {self.model_name}{rot_str}{mir_str}{age_str} | Thresh: {self.threshold:.2f}"
            cv2.putText(out, info_left, (12, 23), cv2.FONT_HERSHEY_DUPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

            reg_count = self.get_registered_person_count()
            info_right = f"Database: {reg_count} Wajah | Sumber: {source_name}"
            (ir_w, _), _ = cv2.getTextSize(info_right, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
            cv2.putText(out, info_right, (w - ir_w - 12, 23), cv2.FONT_HERSHEY_DUPLEX, 0.45, COLOR_ACCENT, 1, cv2.LINE_AA)

            # Footer lengkap tombol pintasan keyboard
            footer_text = "[Q] Keluar  [S] Daftar  [A] Usia  [G] Gender  [R] Reload  [O] Rotasi  [M] Cermin  [H] HUD  [L] Titik"
            (ft_w, _), _ = cv2.getTextSize(footer_text, cv2.FONT_HERSHEY_DUPLEX, 0.40, 1)
            f_y = h - 10
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
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

        cl_x = min(corner_len, (x2 - x1) // 3)
        cl_y = min(corner_len, (y2 - y1) // 3)

        t = thickness
        # Kiri Atas
        cv2.line(img, (x1, y1), (x1 + cl_x, y1), color, t, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1, y1 + cl_y), color, t, cv2.LINE_AA)
        # Kanan Atas
        cv2.line(img, (x2, y1), (x2 - cl_x, y1), color, t, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2, y1 + cl_y), color, t, cv2.LINE_AA)
        # Kiri Bawah
        cv2.line(img, (x1, y2), (x1 + cl_x, y2), color, t, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1, y2 - cl_y), color, t, cv2.LINE_AA)
        # Kanan Bawah
        cv2.line(img, (x2, y2), (x2 - cl_x, y2), color, t, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2, y2 - cl_y), color, t, cv2.LINE_AA)
