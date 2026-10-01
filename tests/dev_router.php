<?php
// Local Development Router for BirdNET-Pi
$root = dirname(__DIR__);
$uri = parse_url($_SERVER['REQUEST_URI'], PHP_URL_PATH);

// Ensure include_path includes both root, homepage, and scripts
set_include_path(get_include_path() . PATH_SEPARATOR . $root . PATH_SEPARATOR . $root . '/scripts' . PATH_SEPARATOR . $root . '/homepage');

// Serve static assets from homepage if present
$homepage_file = $root . '/homepage' . $uri;
$ext = pathinfo($homepage_file, PATHINFO_EXTENSION);
if ($uri !== '/' && $ext !== 'php' && file_exists($homepage_file) && !is_dir($homepage_file)) {
    if ($ext === 'css') header('Content-Type: text/css');
    elseif ($ext === 'js') header('Content-Type: application/javascript');
    elseif ($ext === 'svg') header('Content-Type: image/svg+xml');
    elseif ($ext === 'png') header('Content-Type: image/png');
    elseif ($ext === 'jpg' || $ext === 'jpeg') header('Content-Type: image/jpeg');
    readfile($homepage_file);
    exit;
}

// Route index
if ($uri === '/' || $uri === '/index.php') {
    chdir($root . '/homepage');
    require $root . '/homepage/index.php';
    exit;
}

// Route views.php
if ($uri === '/views.php') {
    chdir($root . '/homepage');
    require $root . '/homepage/views.php';
    exit;
}

// Route scripts (overview.php, todays_detections.php, etc.)
$script_file = $root . '/scripts' . $uri;
if (file_exists($script_file) && !is_dir($script_file)) {
    chdir($root . '/scripts');
    require $script_file;
    exit;
}

// Route direct php files if inside homepage
if (file_exists($homepage_file) && pathinfo($homepage_file, PATHINFO_EXTENSION) === 'php') {
    chdir($root . '/homepage');
    require $homepage_file;
    exit;
}

// Placeholder for spectrogram image if requested
if ($uri === '/spectrogram.png') {
    header('Content-Type: image/png');
    // Generate a simple 1x1 or transparent PNG
    echo base64_decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==');
    exit;
}

// Default 404
http_response_code(404);
echo "404 Not Found: " . htmlspecialchars($uri);
