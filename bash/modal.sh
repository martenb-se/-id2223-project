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

# Colors must be sources before
[ -z "${FMT_RESET}" ] && { echo "Source colors.sh first!"; exit 1; }

# Modal help
# 1: run|deploy
modal_execute() {
  modal $1 $PYTHON_SCRIPT > "modal-$1.log" 2>&1 &
  modal_exec_pid=$!
  tail -f "modal-$1.log" &
  tail_pid=$!
  wait "$modal_exec_pid"
  modal_exec_exit_code=$?

  # Kill tail now that program is done
  kill -9 $tail_pid

  if [ "$modal_exec_exit_code" -eq 0 ]; then
    echo "Good"
  else
    if grep -q "No function was specified, and no \`stub\` variable could be found" "modal-$1.log"; then
      echo -e "\n\n\n${FMT_RED}No stub defined! Cannot continue.${FMT_RESET}"
      exit 1
    else
      echo -e "\n\n\n${FMT_RED}Unknown error.${FMT_RESET}"
      exit 1
    fi
  fi
}