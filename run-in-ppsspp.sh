#!/usr/bin/env bash

# Starts the built EBOOT in PPSSPP without a desktop window, so the demo can be
# run from a script or from psp-devloop.
#
# The emulator writes ms0:/raytracer-result.json into state/ppsspp/, which is
# what the caller waits for. Nothing here decides whether the run passed; that
# is the caller's job.
#
# Exits 77 when no PPSSPP can be found, so an unconfigured machine reports the
# stage as skipped rather than failed.

set -Eeuo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
eboot="$project_dir/target/mipsel-sony-psp/release/EBOOT.PBP"

if [[ ! -f "$eboot" ]]; then
  printf 'Build it first: cargo psp --release\n' >&2
  exit 2
fi

# PPSSPP_BIN wins; otherwise take one from PATH. As a local convenience this
# also finds the copy that ships with a sibling psp-tuxracer checkout, which is
# where the author's build lives.
if [[ -z "${PPSSPP_BIN:-}" ]]; then
  if command -v PPSSPPSDL >/dev/null; then
    PPSSPP_BIN="$(command -v PPSSPPSDL)"
  else
    sibling="$project_dir/../tuxracer-psp/emulator/AppDir/shared/bin/PPSSPPSDL"
    if [[ -x "$sibling" ]]; then
      PPSSPP_BIN="$sibling"
      export LD_LIBRARY_PATH="$project_dir/../tuxracer-psp/emulator/host-libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
  fi
fi

if [[ -z "${PPSSPP_BIN:-}" || ! -x "$PPSSPP_BIN" ]]; then
  printf '[skip] no PPSSPP found; set PPSSPP_BIN to its executable\n'
  exit 77
fi

# PPSSPP puts its memory stick under $XDG_CONFIG_HOME/ppsspp, so ms0:/ ends up
# in state/ppsspp/ and the result file lands next to PSP/.
export XDG_CONFIG_HOME="$project_dir/state"
export XDG_DATA_HOME="$project_dir/state/data"
export XDG_CACHE_HOME="$project_dir/state/cache"
mkdir -p "$XDG_CONFIG_HOME/ppsspp/PSP" "$XDG_DATA_HOME" "$XDG_CACHE_HOME"

# No window, no audio device: this has to run on a machine nobody is sitting at.
export SDL_VIDEODRIVER=x11
export SDL_AUDIODRIVER=dummy
unset WAYLAND_DISPLAY EGL_PLATFORM

exec xvfb-run -a -s '-screen 0 960x544x24 -nolisten tcp' \
  "$PPSSPP_BIN" --windowed --log="$project_dir/state/ppsspp.log" "$eboot"
