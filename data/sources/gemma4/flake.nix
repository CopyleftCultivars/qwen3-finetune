{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
      in {
        devShells.default = pkgs.mkShell {
          # Only use Nix for build tools — NOT Python.
          # Nix Python has a different glibc than Ubuntu 22.04 and can't access
          # the system CUDA driver. The system/conda Python already works with both.
          packages = [
            pkgs.cmake
            pkgs.git
            pkgs.just
            pkgs.gcc
          ];

          shellHook = ''
            # Resolve system Python — prefers conda base if active, else python3
            SYS_PYTHON=$(which python3)
            echo "Using Python: $SYS_PYTHON ($(''${SYS_PYTHON} --version))"

            # PIP_NO_USER=1 prevents pip from writing to ~/.local — all packages
            # go into .venv instead, keeping the environment fully isolated.
            export PIP_NO_USER=1
            export PYTHONNOUSERSITE=1

            if [ ! -d .venv ]; then
              echo ">>> Creating .venv with system Python..."
              ''${SYS_PYTHON} -m venv .venv

              echo ">>> Installing unsloth and deps..."
              # Let unsloth pick its own torch version — pinning torch separately
              # causes unsloth 2024.x to be chosen which conflicts with new transformers.
              # --extra-index-url ensures torch CUDA wheels are available when unsloth
              # resolves its torch dependency.
              .venv/bin/pip install --no-user \
                unsloth bitsandbytes \
                "huggingface_hub>=0.22.0" \
                sentencepiece protobuf safetensors \
                --extra-index-url https://download.pytorch.org/whl/cu124 || exit 1

              echo ">>> Setup complete."
            fi

            source .venv/bin/activate
            echo "Activated .venv — $(python --version), torch $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not yet installed')"
          '';
        };
      }
    );
}
