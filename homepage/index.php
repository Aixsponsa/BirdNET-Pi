<?php

$requestUri = parse_url($_SERVER['REQUEST_URI'], PHP_URL_PATH);

if (strpos($requestUri, '/api/v1/') === 0) {
  include_once 'scripts/api.php';
  die();
}

/* Prevent XSS input */
// Deprecated FILTER_SANITIZE_STRING removed. Parameterized outputs are escaped in views.
require_once 'scripts/common.php';
$config = get_config();
$site_name = get_sitename();
$color_scheme = get_color_scheme();
set_timezone();

?>
<!DOCTYPE html>
<html lang="en">
<title><?php echo $site_name; ?></title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link id="iconLink" rel="shortcut icon" sizes=85x85 href="images/bird.png" />
<link rel="stylesheet" href="<?php echo $color_scheme . '?v=' . filemtime($color_scheme); ?>">
<link rel="stylesheet" type="text/css" href="static/dialog-polyfill.css" />
<body>
<div class="banner">
  <div class="brand-lockup">
    <div class="logo">
      <?php if(isset($_GET['logo'])) {
        echo "<a href=\"https://github.com/Nachtzuster/BirdNET-Pi.git\" target=\"_blank\"><img style=\"width:60px;height:60px;\" src=\"images/bird.svg\"></a>";
      } else {
        echo "<a href=\"https://github.com/Nachtzuster/BirdNET-Pi.git\" target=\"_blank\"><img src=\"images/bird.svg\"></a>";
      }?>
    </div>
    <div class="brand-text">
      <h1><a href="/"><img class="topimage" src="images/bnp.png"></a></h1>
      <div class="site-coordinates"><h3><?php echo $site_name; ?></h3></div>
    </div>
  </div>

  <div class="stream-container">
    <?php
    if(isset($_GET['stream'])){
      ensure_authenticated('You cannot listen to the live audio stream');
      echo '
      <div class="stream">
        <audio controls autoplay><source src="/stream"></audio>
      </div>';
    } else {
      echo '
      <div class="stream">
        <form action="index.php" method="GET">
          <button type="submit" name="stream" value="play">Live Audio</button>
        </form>
      </div>';
    }
    ?>
  </div>
</div>
if(isset($_GET['filename'])) {
  $filename = $_GET['filename'];
  $filename_esc = htmlspecialchars($filename, ENT_QUOTES, 'UTF-8');
echo "
<iframe src=\"views.php?view=Recordings&filename=$filename_esc\"></iframe>";
} else {
  echo "
<iframe src=\"views.php\"></iframe>";
}
