#!/usr/bin/env bash
# Source this file before launching Fast CP with Python, torchrun, or Slurm.

_fast_cp_project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export FAST_CP_DEPS_ROOT=${FAST_CP_DEPS_ROOT:-$_fast_cp_project_root/.deps/fast-cp}
export FAST_CP_BUILD_ROOT=${FAST_CP_BUILD_ROOT:-$FAST_CP_DEPS_ROOT/src/fast-ulysses}
export FLASH_ATTENTION_ROOT=${FLASH_ATTENTION_ROOT:-$FAST_CP_DEPS_ROOT/src/flash-attention}
export NVSHMEM_HOME=${NVSHMEM_HOME:-$FAST_CP_DEPS_ROOT/python/nvidia/nvshmem}

# Preserve caller paths and avoid duplicates when launchers source this again.
_fast_cp_prepend_path() {
    local variable=$1 entry=$2
    case ":${!variable-}:" in
        *":$entry:"*) ;;
        *) printf -v "$variable" '%s' "$entry${!variable:+:${!variable}}" ;;
    esac
    export "$variable"
}

for _fast_cp_source in "$FLASH_ATTENTION_ROOT" "$FAST_CP_BUILD_ROOT"; do
    _fast_cp_prepend_path CHITU_EXTRA_PYTHONPATH "$_fast_cp_source"
    _fast_cp_prepend_path PYTHONPATH "$_fast_cp_source"
done
_fast_cp_prepend_path PYTHONPATH "$_fast_cp_project_root"
_fast_cp_prepend_path LD_LIBRARY_PATH "$NVSHMEM_HOME/lib"

unset -f _fast_cp_prepend_path
unset _fast_cp_project_root _fast_cp_source
