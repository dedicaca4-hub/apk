/**
 * app.js
 * Controller utama antarmuka PWA FaceAI: Kamera, Canvas Overlay,
 * Event Handler, dan Alur Pendaftaran Wajah.
 */

document.addEventListener('DOMContentLoaded', async () => {
  // Elements
  const video = document.getElementById('camera-video');
  const canvas = document.getElementById('overlay-canvas');
  const ctx = canvas.getContext('2d');

  const fpsDisplay = document.getElementById('fps-val');
  const dbCountDisplay = document.getElementById('db-count');
  const statusPill = document.getElementById('status-pill');
  const statusText = document.getElementById('status-text');

  const btnShutter = document.getElementById('btn-shutter');
  const btnFlipCam = document.getElementById('btn-flip-cam');
  const btnToggleAge = document.getElementById('btn-toggle-age');
  const btnToggleGender = document.getElementById('btn-toggle-gender');
  const btnOpenDb = document.getElementById('btn-open-db');
  const btnInstall = document.getElementById('btn-install');

  // Modals
  const enrollModal = document.getElementById('enroll-modal');
  const enrollClose = document.getElementById('enroll-close');
  const enrollCancel = document.getElementById('enroll-cancel');
  const enrollSave = document.getElementById('enroll-save');
  const enrollNameInput = document.getElementById('enroll-name-input');
  const enrollThumb = document.getElementById('enroll-thumb');
  const enrollAgeBadge = document.getElementById('enroll-age-badge');

  const dbModal = document.getElementById('db-modal');
  const dbClose = document.getElementById('db-close');
  const dbList = document.getElementById('db-list');
  const dbClearAll = document.getElementById('db-clear-all');

  const toast = document.getElementById('toast-msg');
  const installBanner = document.getElementById('install-banner');
  const installBannerBtn = document.getElementById('install-banner-btn');

  // State
  let currentFacingMode = 'user'; // 'user' (depan) atau 'environment' (belakang)
  let currentStream = null;
  let isRunning = false;
  let showAge = true;
  let showGender = true;
  let deferredPrompt = null;
  let lastCapturedFace = null;

  // FPS Tracker
  let lastTime = performance.now();
  let frameCount = 0;
  let currentFps = 0;

  // PWA Install Event Handler
  window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    deferredPrompt = e;
    installBanner.classList.add('active');
    if (btnInstall) btnInstall.style.display = 'flex';
  });

  if (installBannerBtn) {
    installBannerBtn.addEventListener('click', async () => {
      if (deferredPrompt) {
        deferredPrompt.prompt();
        const { outcome } = await deferredPrompt.userChoice;
        console.log('[PWA] User choice:', outcome);
        deferredPrompt = null;
        installBanner.classList.remove('active');
      }
    });
  }

  // Toast Helper
  function showToast(message, isError = false) {
    toast.textContent = message;
    toast.className = `toast-msg show ${isError ? 'error' : ''}`;
    setTimeout(() => {
      toast.className = 'toast-msg';
    }, 3200);
  }

  // Update Database Counter
  async function updateDbBadge() {
    try {
      const faces = await window.faceDB.getAllFaces();
      dbCountDisplay.textContent = `${faces.length} Wajah`;
    } catch (e) {
      dbCountDisplay.textContent = '0 Wajah';
    }
  }

  // Inisialisasi Kamera
  async function startCamera(facingMode = 'user') {
    if (currentStream) {
      currentStream.getTracks().forEach(track => track.stop());
    }

    try {
      statusText.textContent = 'Membuka kamera...';
      const constraints = {
        audio: false,
        video: {
          facingMode: { ideal: facingMode },
          width: { ideal: 1280 },
          height: { ideal: 720 }
        }
      };

      currentStream = await navigator.mediaDevices.getUserMedia(constraints);
      video.srcObject = currentStream;

      if (facingMode === 'environment') {
        video.classList.add('back-camera');
      } else {
        video.classList.remove('back-camera');
      }

      await new Promise(resolve => {
        video.onloadedmetadata = () => {
          video.play();
          resolve();
        };
      });

      resizeCanvas();
      statusText.textContent = 'Live AI Aktif';
      return true;
    } catch (err) {
      console.error('[Kamera] Gagal membuka kamera:', err);
      showToast('Gagal mengakses kamera: ' + err.message, true);
      statusText.textContent = 'Kamera Error';
      return false;
    }
  }

  function resizeCanvas() {
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
  }

  window.addEventListener('resize', resizeCanvas);

  // Render Corner Box Futuristik
  function drawCornerBox(x, y, w, h, color, thickness = 3, cornerLen = 20) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    // Kotak tipis transparan
    ctx.strokeRect(x, y, w, h);

    ctx.lineWidth = thickness;
    ctx.lineCap = 'round';

    const clX = Math.min(cornerLen, w / 3);
    const clY = Math.min(cornerLen, h / 3);

    // Kiri Atas
    ctx.beginPath();
    ctx.moveTo(x, y + clY);
    ctx.lineTo(x, y);
    ctx.lineTo(x + clX, y);
    ctx.stroke();

    // Kanan Atas
    ctx.beginPath();
    ctx.moveTo(x + w - clX, y);
    ctx.lineTo(x + w, y);
    ctx.lineTo(x + w, y + clY);
    ctx.stroke();

    // Kiri Bawah
    ctx.beginPath();
    ctx.moveTo(x, y + h - clY);
    ctx.lineTo(x, y + h);
    ctx.lineTo(x + clX, y + h);
    ctx.stroke();

    // Kanan Bawah
    ctx.beginPath();
    ctx.moveTo(x + w - clX, y + h);
    ctx.lineTo(x + w, y + h);
    ctx.lineTo(x + w, y + h - clY);
    ctx.stroke();
  }

  // Draw Futuristic Tag
  function drawTag(text, x, y, color) {
    ctx.font = '600 13px Outfit, sans-serif';
    const textWidth = ctx.measureText(text).width;
    const padH = 10;
    const padV = 6;
    const tagH = 24;
    const tagW = textWidth + padH * 2 + 6;

    let tagX = x;
    let tagY = y - tagH - 8;
    if (tagY < 10) tagY = y + 8; // Geser ke bawah jika mentok atas

    // Background pill
    ctx.fillStyle = 'rgba(12, 17, 26, 0.88)';
    ctx.beginPath();
    ctx.roundRect(tagX, tagY, tagW, tagH, 6);
    ctx.fill();

    // Border
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Color indicator strip di samping kiri
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.roundRect(tagX + 2, tagY + 2, 4, tagH - 4, 3);
    ctx.fill();

    // Text
    ctx.fillStyle = '#ffffff';
    ctx.fillText(text, tagX + padH + 4, tagY + 16);
  }

  // Main Real-time Detection & Recognition Loop
  let isProcessing = false;
  async function processFrame() {
    if (!isRunning) return;

    // Hitung FPS
    frameCount++;
    const now = performance.now();
    if (now - lastTime >= 1000) {
      currentFps = (frameCount * 1000) / (now - lastTime);
      fpsDisplay.textContent = currentFps.toFixed(1);
      frameCount = 0;
      lastTime = now;
    }

    if (!isProcessing && video.readyState === 4) {
      isProcessing = true;

      try {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Mirroring jika kamera depan
        const isMirrored = currentFacingMode === 'user';

        ctx.save();
        if (isMirrored) {
          ctx.translate(canvas.width, 0);
          ctx.scale(-1, 1);
        }

        // 1. Deteksi Wajah
        const faces = await window.faceEngine.detectFaces(video);

        for (const face of faces) {
          // Bounding Box
          const box = {
            x: face.x,
            y: face.y,
            width: face.width,
            height: face.height
          };

          // 2. Estimasi Usia & Gender
          let ageInfo = null;
          let emb = null;

          if (window.faceEngine.isLoaded) {
            ageInfo = await window.faceEngine.estimateAgeAndGender(video, box);
            emb = await window.faceEngine.extractEmbedding(video, box);
          }

          // Simpan data wajah terakhir untuk registrasi
          lastCapturedFace = {
            box: box,
            age: ageInfo ? ageInfo.age : 25,
            gender: ageInfo ? ageInfo.gender : 'L',
            embedding: emb,
            snapshot: window.faceEngine.getFaceSnapshotUrl(video, box)
          };

          // 3. Pencocokan Wajah dengan Database
          let match = { isKnown: false, name: 'Unknown', score: 0.0 };
          if (emb) {
            match = await window.faceDB.matchFace(emb, 0.48);
          }

          // Format Teks Tag
          const attrParts = [];
          if (showAge && ageInfo && ageInfo.age !== null) {
            attrParts.push(`${ageInfo.age}th`);
          }
          if (showGender && ageInfo && ageInfo.gender) {
            attrParts.push(ageInfo.gender);
          }

          const attrStr = attrParts.length > 0 ? ` [${attrParts.join(', ')}]` : '';
          const tagLabel = match.isKnown
            ? `${match.name}${attrStr} (${(match.score * 100).toFixed(0)}%)`
            : `Unknown${attrStr}`;

          const boxColor = match.isKnown ? '#00e599' : '#ff4757'; // Neon Green / Coral Red

          // Gambar corner box dan tag
          drawCornerBox(box.x, box.y, box.width, box.height, boxColor, 3, 22);
          drawTag(tagLabel, box.x, box.y, boxColor);
        }

        ctx.restore();
      } catch (err) {
        console.warn('[Loop] Error:', err);
      } finally {
        isProcessing = false;
      }
    }

    requestAnimationFrame(processFrame);
  }

  // Setup Event Listeners

  // 1. Tombol Shutter (Daftarkan Wajah)
  btnShutter.addEventListener('click', () => {
    if (!lastCapturedFace) {
      showToast('Wajah tidak terdeteksi di kamera!', true);
      return;
    }

    enrollThumb.src = lastCapturedFace.snapshot;
    enrollAgeBadge.textContent = `Estimasi Usia: ~${lastCapturedFace.age} th (${lastCapturedFace.gender === 'L' ? 'Pria' : 'Wanita'})`;
    enrollNameInput.value = '';
    enrollModal.classList.add('active');
    enrollNameInput.focus();
  });

  enrollClose.addEventListener('click', () => enrollModal.classList.remove('active'));
  enrollCancel.addEventListener('click', () => enrollModal.classList.remove('active'));

  enrollSave.addEventListener('click', async () => {
    const name = enrollNameInput.value.trim();
    if (!name) {
      showToast('Masukkan nama orang terlebih dahulu!', true);
      return;
    }

    if (!lastCapturedFace) return;

    try {
      await window.faceDB.addFace({
        name: name,
        embedding: lastCapturedFace.embedding,
        photo: lastCapturedFace.snapshot,
        age: lastCapturedFace.age,
        gender: lastCapturedFace.gender
      });

      enrollModal.classList.remove('active');
      showToast(`Wajah '${name}' berhasil didaftarkan!`);
      updateDbBadge();
    } catch (e) {
      showToast('Gagal menyimpan wajah: ' + e.message, true);
    }
  });

  // 2. Tombol Flip Kamera (Depan / Belakang)
  btnFlipCam.addEventListener('click', async () => {
    currentFacingMode = currentFacingMode === 'user' ? 'environment' : 'user';
    showToast(`Beralih ke kamera ${currentFacingMode === 'user' ? 'depan' : 'belakang'}`);
    await startCamera(currentFacingMode);
  });

  // 3. Toggle Tampilan Usia
  btnToggleAge.addEventListener('click', () => {
    showAge = !showAge;
    btnToggleAge.classList.toggle('active', showAge);
    showToast(`Pendeteksi Usia: ${showAge ? 'AKTIF' : 'NONAKTIF'}`);
  });

  // 4. Toggle Tampilan Gender
  btnToggleGender.addEventListener('click', () => {
    showGender = !showGender;
    btnToggleGender.classList.toggle('active', showGender);
    showToast(`Pendeteksi Gender: ${showGender ? 'AKTIF' : 'NONAKTIF'}`);
  });

  // 5. Database Modal
  btnOpenDb.addEventListener('click', async () => {
    await renderDbList();
    dbModal.classList.add('active');
  });

  dbClose.addEventListener('click', () => dbModal.classList.remove('active'));

  async function renderDbList() {
    const faces = await window.faceDB.getAllFaces();
    dbList.innerHTML = '';

    if (faces.length === 0) {
      dbList.innerHTML = '<div class="empty-state">Belum ada wajah yang didaftarkan.<br>Gunakan tombol bulat di tengah kamera untuk mendaftar.</div>';
      return;
    }

    faces.forEach(item => {
      const el = document.createElement('div');
      el.className = 'db-item';
      el.innerHTML = `
        <div class="db-item-left">
          <img src="${item.photo || 'icons/icon-192.png'}" class="db-thumb" alt="${item.name}">
          <div>
            <div class="db-info-name">${item.name}</div>
            <div class="db-info-meta">Usia: ~${item.age || '-'} th | Gender: ${item.gender || '-'}</div>
          </div>
        </div>
        <button class="btn-delete-item" data-id="${item.id}">Hapus</button>
      `;

      el.querySelector('.btn-delete-item').addEventListener('click', async (e) => {
        const id = e.target.getAttribute('data-id');
        await window.faceDB.deleteFace(id);
        renderDbList();
        updateDbBadge();
        showToast('Wajah berhasil dihapus.');
      });

      dbList.appendChild(el);
    });
  }

  dbClearAll.addEventListener('click', async () => {
    if (confirm('Hapus semua database wajah yang tersimpan?')) {
      await window.faceDB.clearAll();
      renderDbList();
      updateDbBadge();
      showToast('Database wajah dikosongkan.');
    }
  });

  // Inisialisasi Aplikasi Lengkap
  async function initApp() {
    btnToggleAge.classList.add('active');
    btnToggleGender.classList.add('active');
    updateDbBadge();

    // 1. Jalankan kamera duluan agar user langsung melihat video stream
    const camOk = await startCamera(currentFacingMode);
    if (!camOk) return;

    isRunning = true;
    requestAnimationFrame(processFrame);

    // 2. Load model ONNX di background
    try {
      await window.faceEngine.init((msg) => {
        statusText.textContent = msg;
      });
      showToast('Model AI On-Device berhasil dimuat!');

      // 3. Auto-sync database targets dari laptop jika database HP masih kosong
      const existing = await window.faceDB.getAllFaces();
      if (existing.length === 0) {
        try {
          const resp = await fetch('/api/targets');
          if (resp.ok) {
            const data = await resp.json();
            if (data.targets && data.targets.length > 0) {
              for (const t of data.targets) {
                await window.faceDB.addFace({
                  name: t.name,
                  photo: t.photo,
                  embedding: []
                });
              }
              updateDbBadge();
            }
          }
        } catch (syncErr) {
          // Mode offline
        }
      }
    } catch (e) {
      console.warn('[App] Gagal memuat onnxruntime secara lokal:', e);
      statusText.textContent = 'Mode Deteksi Dasar';
      showToast('Model AI berjalan dalam mode fallback browser.', true);
    }
  }

  // Daftarkan Service Worker
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('sw.js').then(() => {
      console.log('[PWA] Service Worker registered.');
    }).catch(e => console.warn('[PWA] Service worker registration error:', e));
  }

  initApp();
});
