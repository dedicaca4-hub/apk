/**
 * db.js
 * Manajemen IndexedDB lokal untuk menyimpan wajah, foto, vektor embedding 512D,
 * dan estimasi usia secara on-device di HP tanpa memerlukan koneksi server.
 */

const DB_NAME = 'FaceAI_DB';
const DB_VERSION = 1;
const STORE_NAME = 'registered_faces';

class FaceDatabase {
  constructor() {
    this.db = null;
  }

  async init() {
    if (this.db) return this.db;

    return new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);

      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          const store = db.createObjectStore(STORE_NAME, { keyPath: 'id', autoIncrement: true });
          store.createIndex('name', 'name', { unique: false });
        }
      };

      request.onsuccess = (event) => {
        this.db = event.target.result;
        resolve(this.db);
      };

      request.onerror = (event) => {
        console.error('[FaceDB] IndexedDB error:', event.target.error);
        reject(event.target.error);
      };
    });
  }

  async getAllFaces() {
    await this.init();
    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([STORE_NAME], 'readonly');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.getAll();

      request.onsuccess = () => resolve(request.result || []);
      request.onerror = (e) => reject(e.target.error);
    });
  }

  async addFace(faceData) {
    await this.init();
    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([STORE_NAME], 'readwrite');
      const store = transaction.objectStore(STORE_NAME);
      const record = {
        name: faceData.name.trim(),
        embedding: Array.from(faceData.embedding || []),
        photo: faceData.photo || '',
        age: faceData.age || null,
        gender: faceData.gender || null,
        createdAt: new Date().toISOString()
      };

      const request = store.add(record);
      request.onsuccess = () => resolve(request.result);
      request.onerror = (e) => reject(e.target.error);
    });
  }

  async deleteFace(id) {
    await this.init();
    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([STORE_NAME], 'readwrite');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.delete(Number(id));

      request.onsuccess = () => resolve(true);
      request.onerror = (e) => reject(e.target.error);
    });
  }

  async clearAll() {
    await this.init();
    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([STORE_NAME], 'readwrite');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.clear();

      request.onsuccess = () => resolve(true);
      request.onerror = (e) => reject(e.target.error);
    });
  }

  /**
   * Menghitung Cosine Similarity antara embedding kamera live dengan database.
   */
  static cosineSimilarity(vecA, vecB) {
    if (!vecA || !vecB || vecA.length !== vecB.length) return 0;
    let dotProduct = 0.0;
    let normA = 0.0;
    let normB = 0.0;

    for (let i = 0; i < vecA.length; i++) {
      dotProduct += vecA[i] * vecB[i];
      normA += vecA[i] * vecA[i];
      normB += vecB[i] * vecB[i];
    }

    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  /**
   * Mencari pencocokan terbaik dari database.
   */
  async matchFace(queryEmbedding, threshold = 0.50) {
    const knownFaces = await this.getAllFaces();
    if (!knownFaces.length || !queryEmbedding) {
      return { isKnown: false, name: 'Unknown', score: 0.0 };
    }

    let bestScore = -1.0;
    let bestMatch = null;

    for (const item of knownFaces) {
      const sim = FaceDatabase.cosineSimilarity(queryEmbedding, item.embedding);
      if (sim > bestScore) {
        bestScore = sim;
        bestMatch = item;
      }
    }

    const isKnown = bestScore >= threshold;
    return {
      isKnown: isKnown,
      name: isKnown ? bestMatch.name : 'Unknown',
      score: Math.max(0.0, Math.min(1.0, bestScore)),
      matchedFace: bestMatch
    };
  }
}

window.faceDB = new FaceDatabase();
