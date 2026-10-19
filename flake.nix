{
  description = "tractor — local System One extraction API development";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
          postal = (pkgs.libpostal.override { withData = true; }).overrideAttrs {
            version = "1.1.4";
            src = pkgs.fetchFromGitHub {
              owner = "openvenues";
              repo = "libpostal";
              rev = "ffafa3fcd687631955453df70906c53d5677ce9a";
              hash = "sha256-L7t/z5mBrV7RxRrkYDRyKhjQRS1p5EOE01f2eEOUxfI=";
            };
          };
        in
        {
          default = pkgs.mkShell {
            name = "tractor";
            packages = with pkgs; [
              python312
              uv
              git
              curl
              jq
              just
              pkg-config
              stdenv.cc
              postal
            ];
            # Binary Python wheels need these shared libraries on NixOS.
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
              pkgs.stdenv.cc.cc.lib
              pkgs.zlib
              postal
            ];
            UV_PYTHON = "${pkgs.python312}/bin/python3.12";
            UV_PYTHON_DOWNLOADS = "never";
            PYTHONNOUSERSITE = "1";
            HF_HUB_DISABLE_TELEMETRY = "1";
            TOKENIZERS_PARALLELISM = "false";
            shellHook = ''
              export HF_HOME="$PWD/.cache/huggingface"
              export UV_CACHE_DIR="$PWD/.cache/uv"
              # CUDA runtime comes from the locked wheels; use the host driver.
              if [ -d /run/opengl-driver/lib ]; then
                export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/run/opengl-driver/lib"
                # Triton otherwise invokes /sbin/ldconfig, absent on NixOS.
                export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib
              fi
              export PATH="$PWD/.venv/bin:$PATH"
            '';
          };
        }
      );
      formatter = forAllSystems (system: nixpkgs.legacyPackages.${system}.nixfmt);
    };
}
