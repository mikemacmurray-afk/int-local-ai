#!/usr/bin/env bash
# ==============================================================================
# configure_vram_limit.sh
# Raises macOS Apple Silicon Metal wired memory limit (iogpu.wired_limit_mb)
# for a 64 GB Mac Mini M4 Pro and installs a persistent LaunchDaemon.
#
# Default target for 64 GB Mac Mini: 57344 MB (56 GB) — leaves 8 GB for macOS.
# Headless server option (64 GB):    pass 58368 (57 GB) as $1.
# Legacy 48 GB Mac Mini option:      pass 43008 (42 GB) as $1.
# ==============================================================================

set -euo pipefail

LIMIT_MB="${1:-57344}"
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
