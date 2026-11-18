#!/usr/bin/env bash
# a9_tune.sh — Strix Halo (gfx1151) inference tuning for the A9 Mega.
#
# Applies docs/16-kennel-engine.md section 4.3 (machine-config rung):
#   1. GTT lift: kernel params amdgpu.gttsize (MiB) + ttm.pages_limit +
#      ttm.page_pool_size (4-KiB pages)  -> GPU-allocatable pool ~ RAM
#      (gufo recipe, Strata pairs pages_limit with page_pool_size).
#   2. memlock unlimited via limits.d (every Strix Halo engine needs it;
#      stock user limit is 8 MiB).
#   3. BIOS UMA carve-out -> minimum: NOT scriptable; detected + reported.
#   4. opt-in: amdgpu.noretry=1 — halogen issues #83/#85 document noretry=0
#      (retry on lost page mapping) converting page-loss bugs into silent
#      GPU hangs; =1 faults loudly instead. Off by default.
#
# Usage:
#   ./a9_tune.sh                 # --check: read-only status report
#   sudo ./a9_tune.sh --apply    # write configs (backups + restore available)
#   sudo ./a9_tune.sh --apply --gtt-gib 112 --noretry
#   sudo ./a9_tune.sh --restore  # undo the latest backup
#
# Reboot required for kernel params; re-login for memlock.
# Post-reboot verify: cat /proc/cmdline; cat /sys/class/drm/card*/device/mem_info_gtt_total;
#   cargo run -p kennel -- devices   (totalGlobalMem should ~ match GTT target)

set -euo pipefail

PROG=${0##*/}
BACKUP_ROOT=/root/a9-tune-backups
GTT_GIB=auto
DO_APPLY=0 DO_RESTORE=0 NORETRY=0

die() { echo "$PROG: $*" >&2; exit 1; }
note() { echo "  $*"; }

for arg in "$@"; do
  case $arg in
    --check|"") ;;
    --apply) DO_APPLY=1 ;;
    --restore) DO_RESTORE=1 ;;
    --noretry) NORETRY=1 ;;
    --gtt-gib) : ;; # handled below
    --gtt-gib=*) GTT_GIB=${arg#--gtt-gib=} ;;
    *) die "unknown arg: $arg (see header for usage)" ;;
  esac
done
# accept space-separated --gtt-gib N
for ((i=1; i<$#; i++)); do
  [ "${@:i:1}" = "--gtt-gib" ] && GTT_GIB=${@:i+1:1}
done

mem_total_kib() { awk '/^MemTotal:/{print $2}' /proc/meminfo; }
GIB_KIB=$((1024*1024))

gtt_sys() { cat /sys/class/drm/card*/device/mem_info_gtt_total 2>/dev/null | head -1; }
vram_sys() { cat /sys/class/drm/card*/device/mem_info_vram_total 2>/dev/null | head -1; }

param_present() { grep -qw "$1" /proc/cmdline 2>/dev/null; }

# ---------------------------------------------------------------- targets
compute_targets() {
  local mem_gib=$(( $(mem_total_kib) / GIB_KIB ))
  if [ "$GTT_GIB" = auto ]; then
    # leave ~4 GiB for the host; 124 is the gufo ceiling for a 128 GB box
    local want=$(( mem_gib - 4 ))
    if [ $want -gt 124 ]; then want=124; fi
    GTT_GIB=$want
  fi
  # NOTE: with the stock 8 GiB BIOS carve-out MemTotal is ~117 GiB, so the
  # auto default lands ~113; after carve-out -> min, --gtt-gib 124 fits.
  GTT_MIB=$(( GTT_GIB * 1024 ))
  GTT_PAGES=$(( GTT_GIB * 262144 ))   # 1 GiB = 262144 x 4-KiB pages
  KERNEL_PARAMS="amdgpu.gttsize=$GTT_MIB ttm.pages_limit=$GTT_PAGES ttm.page_pool_size=$GTT_PAGES"
  if [ $NORETRY -eq 1 ]; then KERNEL_PARAMS="$KERNEL_PARAMS amdgpu.noretry=1"; fi
  if [ $GTT_GIB -gt $(( mem_gib - 3 )) ]; then
    echo "WARNING: GTT ${GTT_GIB} GiB within 3 GiB of MemTotal ${mem_gib} GiB" >&2
  fi
}

LIMITS_FILE=/etc/security/limits.d/90-kennel-tune.conf
LIMITS_BODY="# kennel / Strix Halo inference tuning (docs/16 sec 4.3)
* soft memlock unlimited
* hard memlock unlimited"

# ------------------------------------------------------------- bootloader
detect_bootloader() {
  if [ -d /boot/loader/entries ] || [ -e /boot/loader/loader.conf ]; then
    echo systemd-boot
  elif [ -e /etc/default/grub ] || [ -d /boot/grub ]; then
    echo grub
  elif [ -e /boot/limine.conf ] || compgen -G '/boot/limine*.conf' >/dev/null 2>&1; then
    echo limine
  else
    echo unknown
  fi
}

backup_file() { # $1 = file
  [ -e "$1" ] || return 0
  mkdir -p "$BK"
  [ -e "$BK/$(basename "$1")" ] || cp -a "$1" "$BK/$(basename "$1")"
  echo "$1" >> "$BK/MANIFEST"
}

patch_cmdline_line() { # $1=file $2=regex matching the line to append onto
  local f=$1 re=$2
  if grep -qE "$re" "$f" && ! grep -q 'amdgpu\.gttsize' "$f"; then
    sed -i -E "/$re/s/$/ $KERNEL_PARAMS/" "$f"
  elif grep -q 'amdgpu\.gttsize' "$f"; then
    note "$f: already carries amdgpu.gttsize (idempotent skip)"
  else
    note "$f: no '$re' line found — skipped"
  fi
}

apply_kernel_params() {
  local bl; bl=$(detect_bootloader)
  echo "bootloader: $bl"
  case $bl in
    systemd-boot)
      # persistent source for kernel-install/UKI layouts, if present
      if [ -e /etc/kernel/cmdline ]; then
        backup_file /etc/kernel/cmdline
        if grep -q 'amdgpu\.gttsize' /etc/kernel/cmdline; then
          note "/etc/kernel/cmdline: already set"
        else
          printf '%s %s\n' "$(cat /etc/kernel/cmdline)" "$KERNEL_PARAMS" > /etc/kernel/cmdline.new
          mv /etc/kernel/cmdline.new /etc/kernel/cmdline
          note "/etc/kernel/cmdline: params appended (survives kernel updates)"
        fi
      fi
      # immediate effect: every existing entry's options line
      shopt -s nullglob
      local f found=0
      for f in /boot/loader/entries/*.conf; do
        backup_file "$f"; found=1
        patch_cmdline_line "$f" '^options[[:space:]]'
      done
      if [ $found -eq 0 ]; then note "no type-1 entries found (UKI-only setup? /etc/kernel/cmdline is the source)"; fi
      ;;
    grub)
      backup_file /etc/default/grub
      local key=GRUB_CMDLINE_LINUX_DEFAULT
      grep -q "^$key=" /etc/default/grub || key=GRUB_CMDLINE_LINUX
      if grep -q 'amdgpu\.gttsize' /etc/default/grub; then
        note "/etc/default/grub: already set"
      else
        sed -i "/^$key=/s/\"\$/ $KERNEL_PARAMS\"/" /etc/default/grub
        if command -v grub-mkconfig >/dev/null; then grub-mkconfig -o /boot/grub/grub.cfg; fi
        note "/etc/default/grub: $key updated + grub-mkconfig run"
      fi
      ;;
    limine)
      if [ -e /etc/default/limine ]; then
        backup_file /etc/default/limine
        if grep -q 'amdgpu\.gttsize' /etc/default/limine; then
          note "/etc/default/limine: already set"
        else
          printf '\nKERNEL_CMDLINE[default]+=" %s"\n' "$KERNEL_PARAMS" >> /etc/default/limine
          if command -v limine-mkinitcpio >/dev/null; then
            limine-mkinitcpio
          elif command -v limine-update >/dev/null; then
            limine-update
          fi
          note "/etc/default/limine: params appended + Limine config regenerated"
        fi
      else
        note "/etc/default/limine not found — append manually to your Limine config: $KERNEL_PARAMS"
      fi
      ;;
    unknown)
      note "no supported bootloader detected — kernel params NOT applied."
      note "append manually to your boot cmdline: $KERNEL_PARAMS"
      ;;
  esac
}

apply_limits() {
  if [ -e "$LIMITS_FILE" ] && grep -q memlock "$LIMITS_FILE"; then
    note "$LIMITS_FILE already present"
  else
    backup_file "$LIMITS_FILE"
    printf '%s\n' "$LIMITS_BODY" > "$LIMITS_FILE"
    note "$LIMITS_FILE written (memlock unlimited; re-login to take effect)"
  fi
}

do_apply() {
  [ "$(id -u)" -eq 0 ] || die "--apply needs root (sudo)"
  compute_targets
  BK="$BACKUP_ROOT/$(date +%Y%m%d-%H%M%S)"
  mkdir -p "$BK"
  echo "backup dir: $BK"
  echo "== kernel params =="
  apply_kernel_params
  echo "== memlock =="
  apply_limits
  echo "== sysctl =="
  local mmc; mmc=$(sysctl -n vm.max_map_count 2>/dev/null || echo 0)
  if [ "$mmc" -ge 1048576 ]; then
    note "vm.max_map_count = $mmc (already fine)"
  else
    printf 'vm.max_map_count = 1048576\n' > /etc/sysctl.d/90-kennel-tune.conf
    note "vm.max_map_count raised to 1048576"
  fi
  report
  echo "REBOOT to activate kernel params; re-login for memlock."
}

do_restore() {
  [ "$(id -u)" -eq 0 ] || die "--restore needs root (sudo)"
  local dir=${1:-}
  if [ -z "$dir" ]; then
    dir=$(ls -1d "$BACKUP_ROOT"/* 2>/dev/null | sort | tail -1) || true
    [ -n "$dir" ] || die "no backups under $BACKUP_ROOT"
  fi
  echo "restoring from: $dir"
  local f
  while read -r f; do
    if [ -e "$dir/$(basename "$f")" ]; then
      cp -a "$dir/$(basename "$f")" "$f"
      note "restored $f"
    fi
  done < <(sort -u "$dir/MANIFEST" 2>/dev/null)
  # files we created that had no original to back up
  if [ -e /etc/sysctl.d/90-kennel-tune.conf ] && [ ! -e "$dir/90-kennel-tune.conf" ]; then
    rm -f /etc/sysctl.d/90-kennel-tune.conf; note "removed sysctl override"
  fi
  if [ -e "$LIMITS_FILE" ] && [ ! -e "$dir/$(basename "$LIMITS_FILE")" ]; then
    rm -f "$LIMITS_FILE"; note "removed limits override"
  fi
  if [ -e /etc/default/grub ] && command -v grub-mkconfig >/dev/null; then
    grub-mkconfig -o /boot/grub/grub.cfg && note "grub-mkconfig re-run"
  fi
  echo "reboot to finish restoring."
}

report() {
  echo
  echo "== current state =="
  local g v; g=$(gtt_sys); v=$(vram_sys)
  if [ -n "$g" ]; then note "GTT pool: $(( g / 1073741824 )) GiB -> target ${GTT_GIB} GiB"; fi
  if [ -n "$v" ]; then note "BIOS UMA carve-out (VRAM): $(( v / 1073741824 )) GiB — set to minimum (512 MiB-1 GiB) in BIOS; GTT supersedes it and the RAM comes back to Linux"; fi
  if param_present amdgpu.gttsize; then
    note "cmdline: $(grep -o 'amdgpu[^ ]*' /proc/cmdline | tr '\n' ' ')"
    note "ttm: $(grep -o 'ttm[^ ]*' /proc/cmdline | tr '\n' ' ')"
  else
    note "cmdline: amdgpu.gttsize NOT set (GTT stays at ~half-RAM default)"
  fi
  local ml
  if [ "$(id -u)" -eq 0 ]; then ml=$(su -l rose -c 'ulimit -l' 2>/dev/null || echo '?'); else ml=$(ulimit -l); fi
  note "memlock: ${ml} KiB -> unlimited after --apply + re-login"
}

do_check() {
  GTT_GIB=${GTT_GIB/auto/auto}
  compute_targets
  report
  echo
  echo "planned kernel params: $KERNEL_PARAMS"
  echo "planned memlock file:  $LIMITS_FILE"
  echo "(run: sudo $0 --apply)"
}

if [ $DO_RESTORE -eq 1 ]; then do_restore "${1:-}"; exit 0; fi
if [ $DO_APPLY -eq 1 ]; then do_apply; else do_check; fi
