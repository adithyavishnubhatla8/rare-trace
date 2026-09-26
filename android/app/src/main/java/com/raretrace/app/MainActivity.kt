package com.raretrace.app

import android.app.DownloadManager
import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.view.View
import android.webkit.URLUtil
import android.webkit.ValueCallback
import android.webkit.WebChromeClient
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.swiperefreshlayout.widget.SwipeRefreshLayout

class MainActivity : AppCompatActivity() {

    private lateinit var webView: WebView
    private lateinit var swipeRefreshLayout: SwipeRefreshLayout
    private lateinit var topProgressBar: ProgressBar
    private lateinit var centerSpinner: ProgressBar
    private lateinit var errorContainer: LinearLayout
    private lateinit var errorText: TextView
    private lateinit var currentServerUrlText: TextView
    private lateinit var retryButton: Button
    private lateinit var configureServerButton: Button

    private var filePathCallback: ValueCallback<Array<Uri>>? = null
    private lateinit var fileChooserLauncher: ActivityResultLauncher<Intent>
    private lateinit var prefs: SharedPreferences

    companion object {
        private const val PREFS_NAME = "raretrace_prefs"
        private const val KEY_SERVER_URL = "server_url"
        private const val DEFAULT_SERVER_URL = "http://10.0.2.2:5000"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

        // Initialize UI Elements
        webView = findViewById(R.id.webView)
        swipeRefreshLayout = findViewById(R.id.swipeRefreshLayout)
        topProgressBar = findViewById(R.id.topProgressBar)
        centerSpinner = findViewById(R.id.centerSpinner)
        errorContainer = findViewById(R.id.errorContainer)
        errorText = findViewById(R.id.errorText)
        currentServerUrlText = findViewById(R.id.currentServerUrlText)
        retryButton = findViewById(R.id.retryButton)
        configureServerButton = findViewById(R.id.configureServerButton)

        // Register modern Activity Result launcher for CSV file picker
        fileChooserLauncher = registerForActivityResult(
            ActivityResultContracts.StartActivityForResult()
        ) { result ->
            if (filePathCallback == null) return@registerForActivityResult
            var results: Array<Uri>? = null
            if (result.resultCode == RESULT_OK && result.data != null) {
                val dataString = result.data?.dataString
                val clipData = result.data?.clipData
                if (clipData != null) {
                    val uriList = mutableListOf<Uri>()
                    for (i in 0 until clipData.itemCount) {
                        uriList.add(clipData.getItemAt(i).uri)
                    }
                    results = uriList.toTypedArray()
                } else if (dataString != null) {
                    results = arrayOf(Uri.parse(dataString))
                }
            }
            filePathCallback?.onReceiveValue(results)
            filePathCallback = null
        }

        // Modern OnBackPressedCallback replacing deprecated onBackPressed()
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (webView.canGoBack()) {
                    webView.goBack()
                } else {
                    isEnabled = false
                    onBackPressedDispatcher.onBackPressed()
                }
            }
        })

        setupWebView()
        setupListeners()

        loadServerUrl(getServerUrl())
    }

    private fun getServerUrl(): String {
        return prefs.getString(KEY_SERVER_URL, DEFAULT_SERVER_URL) ?: DEFAULT_SERVER_URL
    }

    private fun saveServerUrl(url: String) {
        val cleanUrl = url.trim().trimEnd('/')
        prefs.edit().putString(KEY_SERVER_URL, cleanUrl).apply()
        currentServerUrlText.text = "Server: $cleanUrl"
        loadServerUrl(cleanUrl)
    }

    private fun loadServerUrl(url: String) {
        errorContainer.visibility = View.GONE
        webView.visibility = View.VISIBLE
        centerSpinner.visibility = View.VISIBLE
        currentServerUrlText.text = "Server: $url"
        webView.loadUrl(url)
    }

    private fun setupWebView() {
        val settings: WebSettings = webView.settings
        settings.javaScriptEnabled = true
        settings.domStorageEnabled = true
        settings.allowFileAccess = true
        settings.allowContentAccess = true
        settings.useWideViewPort = true
        settings.loadWithOverviewMode = true
        settings.databaseEnabled = true
        settings.mixedContentMode = WebSettings.MIXED_CONTENT_COMPATIBILITY_MODE

        // Modern WebViewClient with WebResourceRequest
        webView.webViewClient = object : WebViewClient() {
            override fun shouldOverrideUrlLoading(view: WebView?, request: WebResourceRequest?): Boolean {
                val url = request?.url?.toString()
                if (url != null && (url.startsWith("http://") || url.startsWith("https://"))) {
                    return false // Let WebView handle it
                }
                return false
            }

            override fun onPageStarted(view: WebView?, url: String?, favicon: Bitmap?) {
                super.onPageStarted(view, url, favicon)
                topProgressBar.visibility = View.VISIBLE
            }

            override fun onPageFinished(view: WebView?, url: String?) {
                super.onPageFinished(view, url)
                topProgressBar.visibility = View.GONE
                centerSpinner.visibility = View.GONE
                swipeRefreshLayout.isRefreshing = false
            }

            override fun onReceivedError(view: WebView?, request: WebResourceRequest?, error: WebResourceError?) {
                super.onReceivedError(view, request, error)
                if (request?.isForMainFrame == true) {
                    topProgressBar.visibility = View.GONE
                    centerSpinner.visibility = View.GONE
                    swipeRefreshLayout.isRefreshing = false
                    webView.visibility = View.GONE
                    errorContainer.visibility = View.VISIBLE
                    val desc = error?.description?.toString() ?: "Network error"
                    errorText.text = "Failed to connect to RARETRACE server:\n$desc"
                }
            }
        }

        // Modern WebChromeClient for File Choosing & Progress
        webView.webChromeClient = object : WebChromeClient() {
            override fun onProgressChanged(view: WebView?, newProgress: Int) {
                super.onProgressChanged(view, newProgress)
                topProgressBar.progress = newProgress
                if (newProgress >= 100) {
                    topProgressBar.visibility = View.GONE
                    centerSpinner.visibility = View.GONE
                }
            }

            override fun onShowFileChooser(
                webView: WebView?,
                filePathCallback: ValueCallback<Array<Uri>>?,
                fileChooserParams: FileChooserParams?
            ): Boolean {
                this@MainActivity.filePathCallback?.onReceiveValue(null)
                this@MainActivity.filePathCallback = filePathCallback

                val intent = fileChooserParams?.createIntent() ?: Intent(Intent.ACTION_GET_CONTENT).apply {
                    type = "*/*"
                    addCategory(Intent.CATEGORY_OPENABLE)
                }

                try {
                    fileChooserLauncher.launch(intent)
                } catch (e: Exception) {
                    this@MainActivity.filePathCallback?.onReceiveValue(null)
                    this@MainActivity.filePathCallback = null
                    Toast.makeText(this@MainActivity, "Cannot open file chooser: ${e.message}", Toast.LENGTH_LONG).show()
                    return false
                }
                return true
            }
        }

        // Handle PDF and CSV Report Downloads cleanly for modern Android 10 - 15
        webView.setDownloadListener { url, userAgent, contentDisposition, mimetype, _ ->
            try {
                val fileName = URLUtil.guessFileName(url, contentDisposition, mimetype)
                val request = DownloadManager.Request(Uri.parse(url)).apply {
                    setMimeType(mimetype)
                    addRequestHeader("User-Agent", userAgent)
                    setDescription("Downloading RARETRACE Report: $fileName")
                    setTitle(fileName)
                    setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
                    setDestinationInExternalPublicDir(Environment.DIRECTORY_DOWNLOADS, fileName)
                }

                val dm = getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager
                dm.enqueue(request)
                Toast.makeText(applicationContext, "Downloading $fileName to Downloads...", Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {
                Toast.makeText(applicationContext, "Download failed: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun setupListeners() {
        swipeRefreshLayout.setOnRefreshListener {
            webView.reload()
        }

        retryButton.setOnClickListener {
            loadServerUrl(getServerUrl())
        }

        configureServerButton.setOnClickListener {
            showConfigureServerDialog()
        }
    }

    private fun showConfigureServerDialog() {
        val currentUrl = getServerUrl()
        val input = EditText(this).apply {
            setText(currentUrl)
            hint = "e.g. http://192.168.1.5:5000"
            setSelection(text.length)
        }

        AlertDialog.Builder(this)
            .setTitle("Configure RARETRACE Server")
            .setMessage("Enter your Flask backend URL:\n\n• Android Emulator: http://10.0.2.2:5000\n• Physical Phone: http://<PC_LOCAL_IP>:5000\n• Cloud: https://your-server.com")
            .setView(input)
            .setPositiveButton("Save & Connect") { _, _ ->
                val newUrl = input.text.toString().trim()
                if (newUrl.isNotEmpty() && (newUrl.startsWith("http://") || newUrl.startsWith("https://"))) {
                    saveServerUrl(newUrl)
                } else {
                    Toast.makeText(this, "URL must start with http:// or https://", Toast.LENGTH_LONG).show()
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }
}
