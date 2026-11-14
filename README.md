# RootCause – Android Plant Disease Detection & Advisory Application

> **Tagline:** *"Healthy Crops, Better Harvests."*  
> **Target Problem:** Offline-first AI plant disease detection and agricultural advisory designed for smallholder farmers with limited connectivity.

---

## 🌾 Project Overview

RootCause is a modern, farmer-friendly Android frontend application engineered to identify crop diseases in seconds without requiring an active internet connection. Designed specifically for agricultural field conditions, the interface features large tactile touch targets, natural earthy green visuals, high-contrast sunlight readability, regional Indian language support, and integrated voice (Text-to-Speech) assistance.

### Supported Crops & Diseases

#### 1. Rice Paddy (*Oryza sativa*)
1. **Bacterial Leaf Blight** (*Xanthomonas oryzae pv. oryzae*)
2. **Brown Spot** (*Bipolaris oryzae*)
3. **Leaf Blast** (*Magnaporthe oryzae*)
4. **Leaf Scald** (*Microdochium oryzae*)
5. **Sheath Blight** (*Rhizoctonia solani*)

#### 2. Wheat (*Triticum aestivum*)
1. **Yellow Rust / Stripe Rust** (*Puccinia striiformis*)
2. **Brown Rust / Leaf Rust** (*Puccinia triticina*)
3. **Stem Rust / Black Rust** (*Puccinia graminis*)
4. **Powdery Mildew** (*Blumeria graminis*)

---

## 📱 Architecture & Required Screens

RootCause follows an offline-first modular architecture implementing all 9 required screens plus crop guides and interactive modals:

| Screen # | Screen Name | Description & Key Features |
|---|---|---|
| **Screen 1** | **Welcome / Splash** | Original RootCause emblem, tagline, agricultural field illustration, feature highlights, and prominent "Get Started" button. |
| **Screen 2** | **Home Dashboard** | Farmer greeting ("Namaste, Kisan Mitra"), 1-tap "Detect Plant Disease" hero card, Rice & Wheat quick diagnosis cards, "Ask a Farming Question" advisory banner, and recent diagnosis records. |
| **Screen 3** | **Crop Selection** | High-definition crop cards for Rice and Wheat detailing scientific names, seasons, vulnerable growth stages, and direct guide links. |
| **Screen 4** | **Image Capture / Upload** | Dual-mode capture (Live Camera stream with framing reticle & Gallery upload), close-up photography tips, Jotform farmer field information form, and disease detection trigger. |
| **Screen 5** | **Disease Detection Results** | Scanned leaf display with diagnostic bounding reticle, confidence score meter (e.g., 94.8%), severity rating, key symptoms, clearly labelled demo/offline mode banner, and graceful handling of low-confidence/uncertain images. |
| **Screen 6** | **Disease Information & Advisory** | Comprehensive pathology dossier: visible signs, causes, organic/bio-control treatments, chemical fungicides with exact farmer dosages, preventive practices, voice reading toolbar, and KVK warning. |
| **Screen 7** | **Language Settings** | 8 languages: English, हिन्दी (Hindi), ಕನ್ನಡ (Kannada), తెలుగు (Telugu), தமிழ் (Tamil), বাংলা (Bengali), मराठी (Marathi), and ਪੰਜਾਬੀ (Punjabi) with instant UI switching. |
| **Screen 8** | **Diagnosis History** | Offline LocalStorage diagnosis records with date, crop filter, confidence match percentage, severity chips, and record inspection. |
| **Screen 9** | **Settings & Support** | Speech rate selector, voice toggle, offline database status, ML backend API URL configuration modal, and one-touch dialer for the Kisan Call Center (`1800-180-1551`). |
| **Guides** | **Knowledge Catalog** | Offline searchable repository of all 9 diseases for Rice and Wheat. |

---

## 📋 Jotform Workflow Integration

The frontend faithfully implements all fields and workflow steps from Jotform [`263172181069457`](https://app.jotform.com/263172181069457) and embedded form [`263172078776063`](https://form.jotform.com/263172078776063):

1. **Farmer Name:** First Name & Last Name
2. **Email / Contact:** Farmer phone number or email address
3. **Crop Selection:** Rice Paddy, Wheat, or Other
4. **Region / Location:** District or State (e.g. Karnal, Mandya, Ludhiana)
5. **Question / Observations:** Field description of symptoms and onset
6. **Leaf Image:** Live camera snapshot or gallery image upload
7. **Offline Queueing:** Local storage queue with visual confirmation receipt

---

## 🚀 How to Run and Preview

### Method 1: Local HTTP Server (Recommended)
From the project root directory, run:
```bash
python -m http.server 8000
```
Then open your web browser at:
```
http://localhost:8000
```

### Method 2: Direct File Launch
Simply double-click [`index.html`](file:///c:/Users/RG%20Revaiah/Desktop/Coursera/School%20Work/BMS_Hacks/index.html) in your file explorer to open it in Chrome, Edge, or Firefox.

### Method 3: Desktop vs. Phone View
When viewing on desktop:
- Click the **"Toggle Phone Frame"** button in the top-right corner to switch between an Android smartphone mockup and responsive fullscreen.
- Click **"Quick Demo"** to run an instant simulation on sample rice leaves.

---

## 🔌 Machine Learning Backend Integration

The frontend is completely decoupled from the machine learning model.

### Default Mode
By default, RootCause operates 100% offline using a built-in heuristic diagnostic engine and comprehensive local pathology database.

### Connecting Your Custom ML Server
1. Navigate to **Settings** (`Screen 9`) → **Machine Learning Backend Hook**.
2. Enter your backend endpoint URL (e.g., `http://localhost:5000/api/diagnose`).
3. The frontend will send a `POST` request with:
```json
{
  "image": "data:image/jpeg;base64,...",
  "crop": "rice"
}
```
4. Expected response schema:
```json
{
  "diseaseId": "rice_bacterial_leaf_blight",
  "diseaseName": "Bacterial Leaf Blight",
  "confidence": 94.6,
  "severity": "High",
  "notes": "Water-soaked lesions along leaf margins."
}
```
If the backend is unreachable or offline, the app automatically falls back to the built-in offline engine.

---

## 📦 Android APK Packaging

RootCause is configured as a Progressive Web App (PWA) with [`manifest.json`](manifest.json) and [`sw.js`](sw.js). To package it into a native Android `.apk`:

### Using Capacitor (Recommended):
```bash
npm install @capacitor/core @capacitor/cli @capacitor/android
npx cap init RootCause com.rootcause.app --web-dir .
npx cap add android
npx cap open android
```
Then click **Build APK** inside Android Studio.

### Using Android WebView:
Load `file:///android_asset/index.html` inside a standard Android `WebView` with `WebSettings.setDomStorageEnabled(true)` and `setJavaScriptEnabled(true)`.

---

## 👥 Team
- **BMS Hacks - RootCause Team**
- Branch: `frontend`
