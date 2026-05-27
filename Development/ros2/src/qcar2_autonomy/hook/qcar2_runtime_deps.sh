if [ -z "$QCAR2_RUNTIME_DEPS_DONE" ]; then
  _qcar2_runtime_prefix="$COLCON_CURRENT_PREFIX"
  if [ -z "$_qcar2_runtime_prefix" ]; then
    _qcar2_runtime_script="${BASH_SOURCE:-$0}"
    _qcar2_runtime_prefix="$(cd "`dirname "$_qcar2_runtime_script"`/../.." > /dev/null && pwd)"
    unset _qcar2_runtime_script
  fi

  for _qcar2_runtime_root in \
    "$_qcar2_runtime_prefix/.." \
    "$_qcar2_runtime_prefix/../.."
  do
    if [ -f "$_qcar2_runtime_root/scripts/install_runtime_deps.sh" ]; then
      "$_qcar2_runtime_root/scripts/install_runtime_deps.sh"
      export QCAR2_RUNTIME_DEPS_DONE=1
      break
    fi
  done

  unset _qcar2_runtime_root
  unset _qcar2_runtime_prefix
fi
