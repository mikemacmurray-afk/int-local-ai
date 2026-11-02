#!/usr/bin/env bash
# ==============================================================================
# configure_vram_limit.sh
# Raises macOS Apple Silicon Metal wired memory limit (iogpu.wired_limit_mb)
# for a 48 GB Mac Mini Pro and optionally installs a LaunchDaemon so it persists
# across reboots.
#
# Default target: 43008 MB (42 GB) — leaves 6 GB for macOS kernel & daemons.
# Headless server option: pass 44032 (43 GB) as $1.
# ==============================================================================

set -euo pipefail

LIMIT_MB="${1:-43008}"
PLIST_PATH="/Library/LaunchDaemons/com.localai.vramlimit.plist"

if [[ $EUID -ne 0 ]]; then
   echo "ERROR: Please run with sudo: sudo $0 [limit_mb]"
   exit 1
fi

echo "==> Current iogpu.wired_limit_mb:"
sysctl iogpu.wired_limit_mb || true

echo "==> Applying iogpu.wired_limit_mb=${LIMIT_MB} immediately..."
sysctl -w iogpu.wired_limit_mb="${LIMIT_MB}"

echo "==> Writing persistent LaunchDaemon to ${PLIST_PATH}..."
cat <<EOF > "${PLIST_PATH}"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.localai.vramlimit</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/sbin/sysctl</string>
        <string>-w</string>
        <string>iogpu.wired_limit_mb=${LIMIT_MB}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>
EOF

chown root:wheel "${PLIST_PATH}"
chmod 644 "${PLIST_PATH}"
launchctl load -w "${PLIST_PATH}" 2>/dev/null || true

echo "==> Verified iogpu.wired_limit_mb is now:"
sysctl iogpu.wired_limit_mb

