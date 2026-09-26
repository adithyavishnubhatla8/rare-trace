# RARETRACE Android APK Generation & Launch Guide

This guide explains how to launch and build the **RARETRACE.apk** Android application using **Android Studio** with the new RARETRACE logo icon, full CSV file upload support, and PDF report downloads.

---

## 1. Project Files Created for Android Studio

The Android project is pre-built in the [`android/`](file:///c:/Users/Adithya/Downloads/RARE%20DEASES/android) directory:

- **[`AndroidManifest.xml`](file:///c:/Users/Adithya/Downloads/RARE%20DEASES/android/app/src/main/AndroidManifest.xml):** Configures app permissions (`INTERNET`, `READ_EXTERNAL_STORAGE`, `WRITE_EXTERNAL_STORAGE`, `usesCleartextTraffic="true"`).
- **[`MainActivity.kt`](file:///c:/Users/Adithya/Downloads/RARE%20DEASES/android/app/src/main/java/com/raretrace/app/MainActivity.kt):** Configures Android `WebView`, CSV File Upload Picker (`WebChromeClient`), and PDF/CSV Download Manager (`DownloadListener`).
- **[`activity_main.xml`](file:///c:/Users/Adithya/Downloads/RARE%20DEASES/android/app/src/main/res/layout/activity_main.xml):** WebView layout.
- **Launcher Icons (`ic_launcher.png`):** Located in [`res/mipmap-.../`](file:///c:/Users/Adithya/Downloads/RARE%20DEASES/android/app/src/main/res), using your exact dark RARETRACE logo image!

---

## 2. Step-by-Step Instructions to Build APK in Android Studio

### Step 1: Start your Flask Backend Server
Ensure your Python Flask backend is running on your host machine:
```bash
python app.py
```
*(Runs on `http://0.0.0.0:5000`)*

---

### Step 2: Open Project in Android Studio
1. Open **Android Studio**.
2. Click **Open an Existing Project**.
3. Navigate to `C:\Users\Adithya\Downloads\RARE DEASES\android` and click **OK**.
4. Android Studio will automatically sync Gradle files.

---

### Step 3: Configure Server Connection URL
In [`android/app/src/main/java/com/raretrace/app/MainActivity.kt`](file:///c:/Users/Adithya/Downloads/RARE%20DEASES/android/app/src/main/java/com/raretrace/app/MainActivity.kt#L22):

- **If testing on Android Studio Emulator:**
  Keep the default IP:
  ```kotlin
  private val SERVER_URL = "http://10.0.2.2:5000"
  ```
- **If testing on Physical Android Phone (connected via USB or Wi-Fi):**
  Change to your PC's local Wi-Fi IP address (e.g., `192.168.1.5`):
  ```kotlin
  private val SERVER_URL = "http://192.168.1.5:5000"
  ```
- **If hosted on Cloud (Render / Railway / Heroku):**
  ```kotlin
  private val SERVER_URL = "https://your-raretrace-app.onrender.com"
  ```

---

### Step 4: Build & Generate APK File
1. In Android Studio menu, click **Build** &rarr; **Build Bundle(s) / APK(s)** &rarr; **Build APK(s)**.
2. Android Studio will compile the project and show a notification: **"APK(s) generated successfully"**.
3. Click **locate** to find `app-debug.apk` (or `app-release.apk`) in:
   `android/app/build/outputs/apk/debug/app-debug.apk`

---

### Step 5: Install and Run on Android
1. Transfer `app-debug.apk` to your Android phone via USB or WhatsApp/Drive.
2. Tap the `.apk` file to install (allow "Install from unknown sources" if prompted).
3. Open the **RARETRACE** app icon (featuring your new dark RARETRACE emblem logo).
4. The full RARETRACE ML Dashboard will launch natively with full CSV upload and PDF report download capabilities!
