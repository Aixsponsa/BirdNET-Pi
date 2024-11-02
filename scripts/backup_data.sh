#!/usr/bin/env bash
# Backup and restore BirdNET data

my_dir=$HOME/BirdNET-Pi/scripts
source /etc/birdnet/birdnet.conf

if [ "$EUID" == 0 ]
  then echo "Please run as a non-root user."
  exit
fi

usage() { echo "Usage: $0 -a backup|restore|size -f <backup_file>" 1>&2; exit 1; }

unset -v ACTION
unset -v ARCHIVE
unset -v QUIET
while getopts "a:f:" o; do
  case "${o}" in
    a)
      ACTION=${OPTARG}
      [ $ACTION == "backup" ] || [ $ACTION == "restore" ] || [ $ACTION == "size" ] || usage
      ;;
    f)
      ARCHIVE=${OPTARG}
      ;;
    *)
      usage
      ;;
  esac
done

[ -z "$ACTION" ] && usage && exit 1
if [ $ACTION != "size" ]; then
  [ -z "$ARCHIVE" ] && usage && exit 1
  [ "$ARCHIVE" == '-' ] && QUIET=1
fi

MEG=1048576

PHP_SERVICE=$(systemctl list-unit-files -t service --output json --no-pager php*-fpm.service | jq --raw-output '.[0].unit_file')
[ -z "$PHP_SERVICE" ] || [ "$PHP_SERVICE" == 'null' ] && echo "Could not determine the php service name, this is most likely a bug." && exit 1

log() {
  [ -z "$QUIET" ] && echo "$1"
}

backup_check() {
  if [ "$ARCHIVE" != '-' ]; then
    [ -f "$ARCHIVE" ] && echo "$ARCHIVE already exists" && exit 1
    estimated_backup_size
    available_space_for_backup
    AVL_MB=$(printf "%1.f" $(bc <<< "$AVAILABLE / $MEG"))
    EST_MB=$(printf "%1.f" $(bc <<< "$ESTIMATED / $MEG"))
    log "Estimated space needed: ${EST_MB}M ($ESTIMATED), space available: ${AVL_MB}M ($AVAILABLE)"
    [ $ESTIMATED -gt $AVAILABLE ] && echo "Not enough space available on $(dirname "$ARCHIVE")"  && exit 1
  fi
}

backup() {
  log "Starting backup, this might take a while"
  CMD='tar --create -f "$ARCHIVE"'
  for obj in  "${optional[@]}";do
    [ -f $obj ] && CMD="$CMD -C $(dirname "$obj") $(basename "$obj")"
  done
  for obj in  "${required[@]}";do
    CMD="$CMD -C $(dirname "$obj") $(basename "$obj")"
  done
  eval "$CMD"
  log "Backup done"
}

estimated_backup_size() {
  CMD='du -s -c -b '
  for obj in  "${optional[@]}";do
    [ -f $obj ] && CMD="$CMD $obj"
  done
  for obj in  "${required[@]}";do
    CMD="$CMD $obj"
  done
  ESTIMATED=$(eval "$CMD | grep total | cut -f 1")
}

available_space_for_backup() {
  AVAILABLE=$(df --output=avail --block-size=1 "$(dirname "$ARCHIVE")" | grep [[:digit:]])
}

available_space_for_restore() {
  AVAILABLE=$(df --output=avail --block-size=1 "$(dirname "/home/$BIRDNET_USER/")" | grep [[:digit:]])
}

estimated_restore_size() {
  TMP=$(du -s -c -b "$ARCHIVE" | grep total | cut -f 1)
  # scale the size up a bit
  ESTIMATED=$(printf "%1.f" $(bc <<< "$TMP * 1.005"))
}

restore_check() {
  [ ! -f "$ARCHIVE" ] && echo "$ARCHIVE" not found && exit 1
  available_space_for_restore
  estimated_restore_size
  AVL_MB=$(printf "%1.f" $(bc <<< "$AVAILABLE / $MEG"))
  EST_MB=$(printf "%1.f" $(bc <<< "$ESTIMATED / $MEG"))
  log "Estimated space needed: ${EST_MB}M ($ESTIMATED), space available: ${AVL_MB}M ($AVAILABLE)"
  [ $ESTIMATED -gt $AVAILABLE ] && echo "Not enough space available on /home/$BIRDNET_USER/"  && exit 1
  log "Checking backup file"
  arch_list=$(tar --list --exclude="*/*" -f "$ARCHIVE" | sed 's/\///')
  for obj in  "${required[@]}";do
    part2=$(basename "$obj")
    ! (echo $arch_list | grep -F -q "$part2") && echo corrupted backup file && exit 1
  done
}

restore() {
  log "Starting restore, this might take a while"
  for obj in  "${required[@]}";do
    tar --extract -p -f "$ARCHIVE" -C "$(dirname "$obj")" "$(basename "$obj")"
  done
  log "Trying to restore optional files"
  for obj in  "${optional[@]}";do
    tar --extract --ignore-failed-read -p -f "$ARCHIVE" -C "$(dirname "$obj")" "$(basename "$obj")" || log "$(basename "$obj") not found in backup file, moving on"
  done
  log "Fixing up configuration file"
  CURRENT_BIRDNET_USER="$BIRDNET_USER"
  source /etc/birdnet/birdnet.conf
  sed -i "s/BIRDNET_USER=.*/BIRDNET_USER=$CURRENT_BIRDNET_USER/" "/home/$CURRENT_BIRDNET_USER/BirdNET-Pi/birdnet.conf"
  sed -i "s|/home/$BIRDNET_USER/|/home/$CURRENT_BIRDNET_USER/|g" "/home/$CURRENT_BIRDNET_USER/BirdNET-Pi/birdnet.conf"
  if [ "$MODEL" == "BirdNET_GLOBAL_6K_V2.4_Model_FP16" ]; then
    /home/$CURRENT_BIRDNET_USER/BirdNET-Pi/scripts/install_language_label_nm.sh -l $DATABASE_LANG
  else
    /home/$CURRENT_BIRDNET_USER/BirdNET-Pi/scripts/install_language_label.sh -l $DATABASE_LANG
  fi
  log "Restore done"
}

required=("/home/$BIRDNET_USER/BirdNET-Pi/birdnet.conf"
"/home/$BIRDNET_USER/BirdNET-Pi/scripts/birds.db"
"/home/$BIRDNET_USER/BirdNET-Pi/BirdDB.txt"
"/home/$BIRDNET_USER/BirdSongs/Extracted/Charts"
"/home/$BIRDNET_USER/BirdSongs/Extracted/By_Date")

# these may or may not exist
optional=("/home/$BIRDNET_USER/BirdNET-Pi/apprise.txt"
"/home/$BIRDNET_USER/BirdNET-Pi/scripts/blacklisted_images.txt"
"/home/$BIRDNET_USER/BirdNET-Pi/scripts/disk_check_exclude.txt"
"/home/$BIRDNET_USER/BirdNET-Pi/exclude_species_list.txt"
"/home/$BIRDNET_USER/BirdNET-Pi/include_species_list.txt")

[ $ACTION == "backup" ] && backup_check
[ $ACTION == "restore" ] && restore_check
if [ $ACTION == "size" ]; then
  estimated_backup_size
  echo $ESTIMATED
  exit
fi

log "Stopping services"
"$my_dir/stop_core_services.sh"

[ $ACTION == "backup" ] && backup
[ $ACTION == "restore" ] && restore

log "Restarting services"
"$my_dir/restart_services.sh" &>/dev/null