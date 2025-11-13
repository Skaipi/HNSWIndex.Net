#!/usr/bin/env bash
set -euo pipefail
set -x

# 1) Install tools we need inside the manylinux_2_28 container
if command -v dnf >/dev/null 2>&1; then
    dnf install -y clang curl
    dnf clean all
elif command -v microdnf >/dev/null 2>&1; then
    microdnf install -y clang curl
    microdnf clean all
elif command -v yum >/dev/null 2>&1; then
    yum install -y clang curl
    yum clean all
else
    echo "No dnf/microdnf/yum found in manylinux container; cannot install clang/curl." >&2
    exit 1
fi

# 2) Install .NET SDK into a local directory
curl -sSL https://dot.net/v1/dotnet-install.sh -o dotnet-install.sh
bash ./dotnet-install.sh --channel 9.0

export DOTNET_ROOT="$HOME/.dotnet"
export PATH="$HOME/.dotnet:$PATH"

# 3) Build your NativeAOT .NET library
bash ./build.sh