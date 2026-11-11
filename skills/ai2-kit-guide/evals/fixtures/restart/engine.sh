#!/usr/bin/env bash
# Mock only: this script never invokes a scientific program.
printf 'attempt\n' >> attempts.log
case "${MOCK_ENGINE_BEHAVIOR:-success}" in
  fail) exit 7 ;;
  no_output) exit 0 ;;
  empty_output) : > trajectory.out ;;
  success) printf 'complete\n' > trajectory.out ;;
  *) exit 9 ;;
esac
