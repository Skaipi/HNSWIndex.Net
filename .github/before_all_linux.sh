#!/usr/bin/env bash
set -euo pipefail
set -x

# 1) Install tools we need inside the manylinux_2_28 container
dnf install -y clang curl

# 2) Install .NET SDK into a local directory
curl -sSL https://dot.net/v1/dotnet-install.sh -o dotnet-install.sh
bash ./dotnet-install.sh --channel 9.0

export DOTNET_ROOT="$HOME/.dotnet"
export PATH="$HOME/.dotnet:$PATH"

# 3) Build your NativeAOT .NET library
bash ./build.sh