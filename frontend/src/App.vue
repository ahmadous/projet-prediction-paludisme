<script setup>
import { ref, computed } from 'vue'

// URL du backend Flask. Surchargée par la variable d'env VITE_API_URL si présente.
// Le backend a flask-cors activé, on peut donc l'appeler directement.
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000'

const fileInput = ref(null)
const previewUrl = ref(null)
const selectedFile = ref(null)
const isDragging = ref(false)
const loading = ref(false)
const error = ref(null)
const result = ref(null)

const isInfected = computed(() => result.value?.classe === 'Infecté')

// Pourcentage de probabilité d'infection (0-100)
const infectionPercent = computed(() => {
  if (!result.value) return 0
  return Math.round(result.value.prediction * 100)
})

// Indice de confiance de la classe prédite
const confidence = computed(() => {
  if (!result.value) return 0
  const p = result.value.prediction
  return Math.round((isInfected.value ? p : 1 - p) * 100)
})

function openPicker() {
  fileInput.value?.click()
}

function handleFiles(files) {
  const file = files?.[0]
  if (!file) return
  if (!file.type.startsWith('image/')) {
    error.value = 'Veuillez sélectionner un fichier image (PNG, JPG…).'
    return
  }
  error.value = null
  result.value = null
  selectedFile.value = file
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
  previewUrl.value = URL.createObjectURL(file)
}

function onInputChange(e) {
  handleFiles(e.target.files)
}

function onDrop(e) {
  isDragging.value = false
  handleFiles(e.dataTransfer.files)
}

function reset() {
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
  previewUrl.value = null
  selectedFile.value = null
  result.value = null
  error.value = null
  if (fileInput.value) fileInput.value.value = ''
}

async function predict() {
  if (!selectedFile.value) return
  loading.value = true
  error.value = null
  result.value = null

  const formData = new FormData()
  formData.append('file', selectedFile.value)

  try {
    const res = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      body: formData,
    })
    const data = await res.json()
    if (!res.ok) {
      throw new Error(data.error || `Erreur serveur (${res.status})`)
    }
    result.value = data
  } catch (e) {
    error.value =
      e.message === 'Failed to fetch'
        ? `Impossible de joindre le serveur (${API_URL}). Vérifiez que le backend Flask est démarré.`
        : e.message
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <header class="header">
    <div class="logo">🩸</div>
    <h1>Détection du Paludisme</h1>
    <p class="subtitle">
      Analyse d'images de cellules sanguines par intelligence artificielle
    </p>
  </header>

  <main class="card">
    <!-- Zone de dépôt -->
    <div
      class="dropzone"
      :class="{ dragging: isDragging, 'has-image': previewUrl }"
      @click="openPicker"
      @dragover.prevent="isDragging = true"
      @dragleave.prevent="isDragging = false"
      @drop.prevent="onDrop"
    >
      <input
        ref="fileInput"
        type="file"
        accept="image/*"
        hidden
        @change="onInputChange"
      />

      <template v-if="previewUrl">
        <img :src="previewUrl" alt="Aperçu de la cellule" class="preview" />
        <p class="filename">{{ selectedFile?.name }}</p>
      </template>

      <template v-else>
        <div class="dz-icon">🔬</div>
        <p class="dz-title">Glissez une image de cellule ici</p>
        <p class="dz-sub">ou cliquez pour parcourir vos fichiers</p>
      </template>
    </div>

    <!-- Actions -->
    <div class="actions">
      <button
        class="btn btn-primary"
        :disabled="!selectedFile || loading"
        @click="predict"
      >
        <span v-if="loading" class="spinner"></span>
        {{ loading ? 'Analyse en cours…' : 'Analyser la cellule' }}
      </button>
      <button
        v-if="selectedFile"
        class="btn btn-ghost"
        :disabled="loading"
        @click="reset"
      >
        Réinitialiser
      </button>
    </div>

    <!-- Erreur -->
    <div v-if="error" class="alert alert-error">⚠️ {{ error }}</div>

    <!-- Résultat -->
    <transition name="fade">
      <section
        v-if="result"
        class="result"
        :class="isInfected ? 'result-infected' : 'result-healthy'"
      >
        <div class="result-head">
          <span class="result-emoji">{{ isInfected ? '🦠' : '✅' }}</span>
          <div>
            <p class="result-label">Diagnostic</p>
            <h2 class="result-classe">
              {{ isInfected ? 'Cellule infectée' : 'Cellule saine' }}
            </h2>
          </div>
          <span class="badge">Confiance {{ confidence }}%</span>
        </div>

        <div class="gauge">
          <div class="gauge-labels">
            <span>Probabilité d'infection</span>
            <strong>{{ result.prediction_percent }}</strong>
          </div>
          <div class="gauge-track">
            <div
              class="gauge-fill"
              :class="isInfected ? 'fill-danger' : 'fill-success'"
              :style="{ width: infectionPercent + '%' }"
            ></div>
          </div>
        </div>

        <p class="disclaimer">
          Résultat fourni à titre indicatif par le modèle. Il ne remplace pas un
          diagnostic médical réalisé par un professionnel de santé.
        </p>
      </section>
    </transition>
  </main>

  <footer class="footer">
    Backend&nbsp;: <code>{{ API_URL }}/predict</code>
  </footer>
</template>

<style scoped>
.header {
  text-align: center;
  margin-bottom: 2rem;
}
.logo {
  font-size: 3rem;
  line-height: 1;
}
.header h1 {
  font-size: 2rem;
  margin: 0.5rem 0 0.25rem;
  background: linear-gradient(90deg, #a5b4fc, #f0abfc);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
}
.subtitle {
  color: var(--text-muted);
  font-size: 0.95rem;
}

.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.75rem;
  box-shadow: 0 20px 50px rgba(0, 0, 0, 0.35);
}

.dropzone {
  border: 2px dashed var(--border);
  border-radius: var(--radius);
  padding: 2.5rem 1.5rem;
  text-align: center;
  cursor: pointer;
  transition: all 0.2s ease;
  background: rgba(255, 255, 255, 0.02);
}
.dropzone:hover,
.dropzone.dragging {
  border-color: var(--primary);
  background: rgba(99, 102, 241, 0.08);
}
.dropzone.has-image {
  padding: 1.5rem;
}
.dz-icon {
  font-size: 2.5rem;
  margin-bottom: 0.75rem;
}
.dz-title {
  font-weight: 600;
  margin-bottom: 0.25rem;
}
.dz-sub {
  color: var(--text-muted);
  font-size: 0.9rem;
}
.preview {
  max-width: 220px;
  max-height: 220px;
  width: auto;
  border-radius: 12px;
  image-rendering: pixelated;
  border: 1px solid var(--border);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
}
.filename {
  margin-top: 0.75rem;
  color: var(--text-muted);
  font-size: 0.85rem;
  word-break: break-all;
}

.actions {
  display: flex;
  gap: 0.75rem;
  margin-top: 1.25rem;
}
.btn {
  flex: 1;
  padding: 0.85rem 1rem;
  border-radius: 12px;
  border: none;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.15s ease;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
}
.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.btn-primary {
  background: var(--primary);
  color: white;
}
.btn-primary:not(:disabled):hover {
  background: var(--primary-hover);
}
.btn-ghost {
  flex: 0 0 auto;
  background: transparent;
  color: var(--text-muted);
  border: 1px solid var(--border);
}
.btn-ghost:not(:disabled):hover {
  color: var(--text);
  border-color: var(--text-muted);
}

.spinner {
  width: 16px;
  height: 16px;
  border: 2px solid rgba(255, 255, 255, 0.4);
  border-top-color: white;
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
}
@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.alert {
  margin-top: 1.25rem;
  padding: 0.85rem 1rem;
  border-radius: 12px;
  font-size: 0.9rem;
}
.alert-error {
  background: var(--danger-soft);
  color: #fca5a5;
  border: 1px solid rgba(239, 68, 68, 0.3);
}

.result {
  margin-top: 1.5rem;
  padding: 1.5rem;
  border-radius: var(--radius);
  border: 1px solid var(--border);
}
.result-healthy {
  background: var(--success-soft);
  border-color: rgba(34, 197, 94, 0.3);
}
.result-infected {
  background: var(--danger-soft);
  border-color: rgba(239, 68, 68, 0.3);
}
.result-head {
  display: flex;
  align-items: center;
  gap: 1rem;
}
.result-emoji {
  font-size: 2.25rem;
}
.result-label {
  color: var(--text-muted);
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.result-classe {
  font-size: 1.4rem;
  margin-top: 0.15rem;
}
.badge {
  margin-left: auto;
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid var(--border);
  padding: 0.35rem 0.7rem;
  border-radius: 999px;
  font-size: 0.8rem;
  font-weight: 600;
  white-space: nowrap;
}

.gauge {
  margin-top: 1.5rem;
}
.gauge-labels {
  display: flex;
  justify-content: space-between;
  font-size: 0.9rem;
  margin-bottom: 0.5rem;
}
.gauge-track {
  height: 12px;
  background: rgba(255, 255, 255, 0.08);
  border-radius: 999px;
  overflow: hidden;
}
.gauge-fill {
  height: 100%;
  border-radius: 999px;
  transition: width 0.6s cubic-bezier(0.22, 1, 0.36, 1);
}
.fill-danger {
  background: linear-gradient(90deg, #f87171, var(--danger));
}
.fill-success {
  background: linear-gradient(90deg, #4ade80, var(--success));
}

.disclaimer {
  margin-top: 1.25rem;
  font-size: 0.8rem;
  color: var(--text-muted);
  line-height: 1.5;
}

.footer {
  text-align: center;
  margin-top: 2rem;
  color: var(--text-muted);
  font-size: 0.8rem;
}
.footer code {
  background: rgba(255, 255, 255, 0.06);
  padding: 0.15rem 0.45rem;
  border-radius: 6px;
}

.fade-enter-active {
  transition: all 0.35s ease;
}
.fade-enter-from {
  opacity: 0;
  transform: translateY(10px);
}

@media (max-width: 520px) {
  .result-head {
    flex-wrap: wrap;
  }
  .badge {
    margin-left: 0;
  }
}
</style>
