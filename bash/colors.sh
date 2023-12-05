# File should only be sourced (thanks https://stackoverflow.com/a/28776166)
sourced=0
if [ -n "$ZSH_VERSION" ]; then
  case $ZSH_EVAL_CONTEXT in *:file) sourced=1;; esac
elif [ -n "$KSH_VERSION" ]; then
  [ "$(cd -- "$(dirname -- "$0")" && pwd -P)/$(basename -- "$0")" != "$(cd -- "$(dirname -- "${.sh.file}")" && pwd -P)/$(basename -- "${.sh.file}")" ] && sourced=1
elif [ -n "$BASH_VERSION" ]; then
  (return 0 2>/dev/null) && sourced=1
else # All other shells: examine $0 for known shell binary filenames.
     # Detects `sh` and `dash`; add additional shell filenames as needed.
  case ${0##*/} in sh|-sh|dash|-dash) sourced=1;; esac
fi

if [[ $sourced = 0 ]]; then
  echo "File should not be run directly, it should only be sourced.";
  exit 1;
fi

# Terminal Colors
FMT_BLACK="$(tput setaf 0)"
FMT_RED="$(tput setaf 1)"
FMT_GREEN="$(tput setaf 2)"
FMT_YELLOW="$(tput setaf 3)"
FMT_BLUE="$(tput setaf 4)"
FMT_MAGENTA="$(tput setaf 5)"
FMT_CYAN="$(tput setaf 6)"
FMT_WHITE="$(tput setaf 7)"

# Terminal Formatting
FMT_BOLD="$(tput bold)"

# Terminal Reset
FMT_RESET="$(tput sgr0)"
