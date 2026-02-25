#!/usr/bin/env bash
# Run all vendored Python tests through numpy-rust-wasm.
# Usage: ./tests/python/run_tests.sh [path-to-binary]
set -euo pipefail

BINARY="${1:-target/debug/numpy-python}"

if [ ! -f "$BINARY" ]; then
    echo "Binary not found: $BINARY"
    echo "Build first: cargo build -p numpy-rust-wasm"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
total_passed=0
total_failed=0
any_failed=0

for test_file in "$SCRIPT_DIR"/test_*.py; do
    echo "--- $(basename "$test_file") ---"
    if "$BINARY" "$test_file"; then
        : # output handled by test file
    else
        any_failed=1
    fi
done

if [ "$any_failed" -ne 0 ]; then
    echo ""
    echo "SOME TESTS FAILED"
    exit 1
else
    echo ""
    echo "ALL TEST FILES PASSED"
fi
