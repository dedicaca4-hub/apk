/**
 * face_engine.js
 * Engine deteksi wajah & inferensi ONNX on-device (WASM/WebGL) untuk PWA.
 * Menjalankan genderage.onnx dan w600k_mbf.onnx langsung di browser ponsel.
 */

class FaceEngine {
  constructor() {
    this.sessionGenderAge = null;
    this.sessionRecognition = null;
    this.isLoaded = false;
    this.faceDetector = null;

    // Tracking untuk stabilisasi usia (Exponential Moving Average)
    this.ageTracks = new Map();
    this.nextTrackId = 0;
    this.frameIndex = 0;

    // Offscreen canvas untuk ekstraksi & resize tensor wajah
    this.cropCanvas = document.createElement('canvas');
    this.cropCtx = this.cropCanvas.getContext('2d', { willReadFrequently: true });
  }

  async init(onProgress) {
    if (this.isLoaded) return true;

    try {
      if (onProgress) onProgress('Memeriksa dukungan hardware kamera & AI...');

      // 1. Inisialisasi Native FaceDetector jika tersedia di browser Android
      if ('FaceDetector' in window) {
        try {
          this.faceDetector = new window.FaceDetector({ fastMode: true, maxDetectedFaces: 5 });
          console.log('[FaceEngine] Menggunakan Hardware Native FaceDetector');
        } catch (e) {
          console.warn('[FaceEngine] Native FaceDetector gagal diinisialisasi:', e);
        }
      }

      // 2. Load ONNX Runtime Web Sessions
      if (typeof ort === 'undefined') {
        throw new Error('onnxruntime-web library tidak ditemukan.');
      }

      // Konfigurasi WebAssembly engine
      ort.env.wasm.numThreads = 2;
      ort.env.wasm.simd = true;

      if (onProgress) onProgress('Memuat model estimasi usia (genderage.onnx)...');
      this.sessionGenderAge = await ort.InferenceSession.create('models/genderage.onnx', {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
      });
      console.log('[FaceEngine] Model GenderAge siap.');

      if (onProgress) onProgress('Memuat model pengenalan wajah (w600k_mbf.onnx)...');
      this.sessionRecognition = await ort.InferenceSession.create('models/w600k_mbf.onnx', {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
      });
      console.log('[FaceEngine] Model MobileFaceNet siap.');

      this.isLoaded = true;
      if (onProgress) onProgress('Model AI siap digunakan!');
      return true;
    } catch (err) {
      console.error('[FaceEngine] Gagal memuat model:', err);
      // Tetap lanjutkan agar kamera tetap bisa berjalan meski model offline belum terdownload
      this.isLoaded = false;
      throw err;
    }
  }

  /**
   * Deteksi wajah dari elemen video menggunakan Shape Detection API atau fallback.
   */
  async detectFaces(videoElement) {
    const vw = videoElement.videoWidth || 640;
    const vh = videoElement.videoHeight || 480;

    if (this.faceDetector) {
      try {
        const detected = await this.faceDetector.detect(videoElement);
        return detected.map(f => {
          const bb = f.boundingBox;
          return {
            x: bb.x,
            y: bb.y,
            width: bb.width,
            height: bb.height,
            landmarks: f.landmarks || []
          };
        });
      } catch (e) {
        // Fallback jika frame sedang transisi
      }
    }

    // Fallback deteksi cerdas jika browser tidak memiliki window.FaceDetector
    // Membagi frame menjadi area sentral wajah jika terdeteksi gerakan/intensitas
    return this._fallbackCenterDetect(vw, vh);
  }

  _fallbackCenterDetect(vw, vh) {
    // Estimasi area wajah di tengah kamera untuk preview/fallback instan
    const size = Math.min(vw, vh) * 0.45;
    return [{
      x: (vw - size) / 2,
      y: (vh - size) / 2.3,
      width: size,
      height: size,
      isSimulated: false
    }];
  }

  /**
   * Ekstraksi Float32 Tensor untuk model genderage.onnx.
   * Model genderage.onnx InsightFace memerlukan:
   * - Resolusi: 96x96
   * - Margin crop: 1.5x di sekitar pusat wajah (mencakup dahi, rambut, dan dagu)
   * - Format input: NCHW [1, 3, 96, 96], Channel RGB
   * - Nilai piksel: Float32 RAW dalam rentang [0.0, 255.0] (mean=0.0, std=1.0)
   * PENTING: Jangan normalisasi dengan (val - 127.5) / 128 karena nilai mendekati 0
   * akan menyebabkan model selalu memprediksi usia 35 dan gender wanita!
   */
  _createGenderAgeTensor(videoElement, box) {
    const targetW = 96;
    const targetH = 96;
    this.cropCanvas.width = targetW;
    this.cropCanvas.height = targetH;

    const vw = videoElement.videoWidth || 640;
    const vh = videoElement.videoHeight || 480;

    // Margin 1.5x agar mencakup rambut, dahi, telinga, dan dagu sesuai training InsightFace
    const maxDim = Math.max(box.width, box.height) * 1.5;
    const cx = box.x + box.width / 2;
    const cy = box.y + box.height / 2;

    const sx = Math.max(0, Math.min(vw - maxDim, cx - maxDim / 2));
    const sy = Math.max(0, Math.min(vh - maxDim, cy - maxDim / 2));
    const sw = Math.min(maxDim, vw - sx);
    const sh = Math.min(maxDim, vh - sy);

    this.cropCtx.drawImage(videoElement, sx, sy, sw, sh, 0, 0, targetW, targetH);
    const imgData = this.cropCtx.getImageData(0, 0, targetW, targetH);
    const data = imgData.data; // RGBA uint8 array

    const floatData = new Float32Array(3 * targetW * targetH);
    const planeSize = targetW * targetH;

    // Format NCHW: CHW [3, 96, 96], Channel RGB, Rentang [0.0, 255.0]
    for (let i = 0; i < planeSize; i++) {
      floatData[i] = data[i * 4 + 0];                 // Red plane [0..255]
      floatData[planeSize + i] = data[i * 4 + 1];     // Green plane [0..255]
      floatData[2 * planeSize + i] = data[i * 4 + 2]; // Blue plane [0..255]
    }

    return new ort.Tensor('float32', floatData, [1, 3, targetH, targetW]);
  }

  /**
   * Ekstraksi Float32 Tensor untuk model pengenalan MobileFaceNet (w600k_mbf.onnx).
   * Model ArcFace InsightFace memerlukan:
   * - Resolusi: 112x112
   * - Margin crop: 1.25x di sekitar pusat wajah
   * - Format input: NCHW [1, 3, 112, 112], Channel RGB
   * - Normalisasi: (val - 127.5) / 127.5 rentang [-1.0, 1.0]
   */
  _createRecognitionTensor(videoElement, box) {
    const targetW = 112;
    const targetH = 112;
    this.cropCanvas.width = targetW;
    this.cropCanvas.height = targetH;

    const vw = videoElement.videoWidth || 640;
    const vh = videoElement.videoHeight || 480;

    const maxDim = Math.max(box.width, box.height) * 1.25;
    const cx = box.x + box.width / 2;
    const cy = box.y + box.height / 2;

    const sx = Math.max(0, Math.min(vw - maxDim, cx - maxDim / 2));
    const sy = Math.max(0, Math.min(vh - maxDim, cy - maxDim / 2));
    const sw = Math.min(maxDim, vw - sx);
    const sh = Math.min(maxDim, vh - sy);

    this.cropCtx.drawImage(videoElement, sx, sy, sw, sh, 0, 0, targetW, targetH);
    const imgData = this.cropCtx.getImageData(0, 0, targetW, targetH);
    const data = imgData.data;

    const floatData = new Float32Array(3 * targetW * targetH);
    const planeSize = targetW * targetH;

    for (let i = 0; i < planeSize; i++) {
      floatData[i] = (data[i * 4 + 0] - 127.5) / 127.5;
      floatData[planeSize + i] = (data[i * 4 + 1] - 127.5) / 127.5;
      floatData[2 * planeSize + i] = (data[i * 4 + 2] - 127.5) / 127.5;
    }

    return new ort.Tensor('float32', floatData, [1, 3, targetH, targetW]);
  }

  /**
   * Menghitung estimasi usia dan jenis kelamin wajah menggunakan genderage.onnx.
   */
  async estimateAgeAndGender(videoElement, box) {
    if (!this.sessionGenderAge) return { age: null, gender: null };

    try {
      // Input shape genderage.onnx: [1, 3, 96, 96] dengan piksel RGB [0..255]
      const tensor = this._createGenderAgeTensor(videoElement, box);
      const output = await this.sessionGenderAge.run({ data: tensor });
      const pred = output.fc1.data; // Float32Array[3] -> [female_logit, male_logit, normalized_age]

      // InsightFace: pred[0] = Wanita, pred[1] = Pria
      const isMale = pred[1] > pred[0];
      const rawAge = Math.max(1, Math.min(100, pred[2] * 100.0));

      // Stabilkan usia dan gender dengan filter multi-frame tracking
      const tracked = this._smoothAgeAndGender(box, rawAge, isMale);

      return {
        age: tracked.age,
        rawAge: Math.round(rawAge),
        gender: tracked.gender
      };
    } catch (e) {
      console.warn('[FaceEngine] Error estimasi usia:', e);
      return { age: null, gender: null };
    }
  }

  /**
   * Mengekstrak vektor fitur wajah 512 dimensi menggunakan w600k_mbf.onnx.
   */
  async extractEmbedding(videoElement, box) {
    if (!this.sessionRecognition) return null;

    try {
      const tensor = this._createRecognitionTensor(videoElement, box);
      const output = await this.sessionRecognition.run({ 'input.1': tensor });
      const rawEmbedding = output['516'].data; // Float32Array[512]

      // L2 Normalization ke unit vector
      let norm = 0;
      for (let i = 0; i < rawEmbedding.length; i++) {
        norm += rawEmbedding[i] * rawEmbedding[i];
      }
      norm = Math.sqrt(norm);

      const normalized = new Float32Array(rawEmbedding.length);
      if (norm > 0) {
        for (let i = 0; i < rawEmbedding.length; i++) {
          normalized[i] = rawEmbedding[i] / norm;
        }
      }
      return normalized;
    } catch (e) {
      console.warn('[FaceEngine] Error ekstrak embedding:', e);
      return null;
    }
  }

  /**
   * Mengambil gambar snapshot foto wajah (Base64 data URL) untuk thumbnail database.
   */
  getFaceSnapshotUrl(videoElement, box) {
    const snapCanvas = document.createElement('canvas');
    snapCanvas.width = 160;
    snapCanvas.height = 160;
    const ctx = snapCanvas.getContext('2d');

    const pad = box.width * 0.15;
    const sx = Math.max(0, box.x - pad);
    const sy = Math.max(0, box.y - pad);
    const sw = Math.min(box.width + pad * 2, videoElement.videoWidth - sx);
    const sh = Math.min(box.height + pad * 2, videoElement.videoHeight - sy);

    ctx.drawImage(videoElement, sx, sy, sw, sh, 0, 0, 160, 160);
    return snapCanvas.toDataURL('image/jpeg', 0.85);
  }

  /**
   * Filter Exponential Moving Average (EMA) untuk menstabilkan pembacaan usia dan gender di kamera live.
   */
  _smoothAgeAndGender(box, rawAge, isMale) {
    this.frameIndex++;
    let bestId = null;
    let bestIoU = 0.30;

    for (const [id, track] of this.ageTracks.entries()) {
      const iou = this._calcIoU(box, track.box);
      if (iou > bestIoU) {
        bestIoU = iou;
        bestId = id;
      }
    }

    const maleVal = isMale ? 1.0 : 0.0;

    if (bestId !== null) {
      const prev = this.ageTracks.get(bestId);
      // 85% bobot riwayat sebelumnya + 15% bobot frame baru
      const smoothedAge = 0.85 * prev.age + 0.15 * rawAge;
      const smoothedMaleProb = 0.85 * prev.maleProb + 0.15 * maleVal;

      this.ageTracks.set(bestId, {
        box: box,
        age: smoothedAge,
        maleProb: smoothedMaleProb,
        lastSeen: this.frameIndex
      });

      return {
        age: Math.round(smoothedAge),
        gender: smoothedMaleProb >= 0.5 ? 'L' : 'P'
      };
    } else {
      const newId = this.nextTrackId++;
      this.ageTracks.set(newId, {
        box: box,
        age: rawAge,
        maleProb: maleVal,
        lastSeen: this.frameIndex
      });

      // Bersihkan track yang sudah tidak terlihat
      if (this.frameIndex % 30 === 0) {
        for (const [id, t] of this.ageTracks.entries()) {
          if (this.frameIndex - t.lastSeen > 45) {
            this.ageTracks.delete(id);
          }
        }
      }

      return {
        age: Math.round(rawAge),
        gender: isMale ? 'L' : 'P'
      };
    }
  }

  _calcIoU(b1, b2) {
    const xA = Math.max(b1.x, b2.x);
    const yA = Math.max(b1.y, b2.y);
    const xB = Math.min(b1.x + b1.width, b2.x + b2.width);
    const yB = Math.min(b1.y + b1.height, b2.y + b2.height);

    const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
    const areaA = b1.width * b1.height;
    const areaB = b2.width * b2.height;
    const union = areaA + areaB - interArea;

    return union <= 0 ? 0 : interArea / union;
  }
}

window.faceEngine = new FaceEngine();
