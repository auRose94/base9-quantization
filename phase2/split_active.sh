#!/usr/bin/env bash
# TRUE (exit 0) iff the 7900 XT is in compute-only duty:
#   - its VRAM usage is small (the compositor has moved to the iGPU or another
#     card), so the full ~19.98 GiB is available for the 14B QLoRA spike;
#   - no foreign compute (ROCm/kfd) user is on the card — the only accepted
#     residents are processes inside this pipeline's/m-chat's own cgroups.
# Used by mix_pipeline_unit.sh as the "hardware split is live" gate for ANY
# route: BIOS-hybrid iGPU compositing, KWIN_DRM_DEVICES render-route, or a
# logged-out window. Polls cheaply; safe to call every 30-60 s.
THRESHOLD_MB=1200
# the 7900 XT = the first rocm device (PCI 03:00.0, before the 12:00.0 iGPU);
# pin to its "Total Used Memory" line — the iGPU now adds a second GPU block
used=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/^GPU\[0\]/ && /Total Used Memory/ {print $NF; exit}')
[ -n "$used" ] || exit 1
[ "$used" -lt $((THRESHOLD_MB * 1000000)) ] || exit 1
# no foreign compute user on /dev/kfd
fuser /dev/kfd > /tmp/split_kfd_users 2>/dev/null
for p in $(cat /tmp/split_kfd_users 2>/dev/null); do
    cg=$(cat /proc/$p/cgroup 2>/dev/null) || continue
    case "$cg" in
        *mixpipeline.service*|*k9chat.service*) : ;;
        *) exit 1 ;;
    esac
done
exit 0