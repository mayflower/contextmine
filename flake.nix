{
  description = "ContextMine development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [ inputs.treefmt-nix.flakeModule ];

      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

      perSystem =
        {
          config,
          pkgs,
          lib,
          system,
          ...
        }:
        let
          # The Python environment is built by Nix from uv.lock.
          workspace = inputs.uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

          # Packages built from source that don't declare their build backend.
          buildSystemOverrides = final: prev: {
            multilspy = prev.multilspy.overrideAttrs (old: {
              nativeBuildInputs = old.nativeBuildInputs ++ final.resolveBuildSystem { flit-core = [ ]; };
            });
          };

          # Workspace members and their directories, installed editable in the dev shell.
          workspaceMembers = {
            contextmine-api = "apps/api";
            contextmine-core = "packages/core";
            contextmine-worker = "apps/worker";
          };

          # hatchling needs `editables` to build editable installs.
          editableOverrides =
            final: prev:
            lib.genAttrs (lib.attrNames workspaceMembers) (
              name:
              prev.${name}.overrideAttrs (old: {
                nativeBuildInputs = old.nativeBuildInputs ++ final.resolveBuildSystem { editables = [ ]; };
              })
            );

          pythonSet =
            (pkgs.callPackage inputs.pyproject-nix.build.packages { python = pkgs.python314; }).overrideScope
              (
                lib.composeManyExtensions [
                  inputs.pyproject-build-systems.overlays.wheel
                  (workspace.mkPyprojectOverlay { sourcePreference = "wheel"; })
                  buildSystemOverrides
                  # Workspace members are installed editable from the checkout.
                  (workspace.mkEditablePyprojectOverlay { root = "$REPO_ROOT"; })
                  editableOverrides
                ]
              );

          virtualenv = pythonSet.mkVirtualEnv "contextmine-dev-env" workspace.deps.all;

          # Same toolchain as rust/spider_md/rust-toolchain.toml and the worker image.
          rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust/spider_md/rust-toolchain.toml;

          # Model API keys etc. live encrypted in .secrets.sops.yaml (flat map of
          # setting name -> value, encrypted to the developer's age key) and are
          # decrypted into one file per setting for the containers' secrets dir.
          contextmine-secrets = pkgs.writeShellApplication {
            name = "contextmine-secrets";
            runtimeInputs = [
              pkgs.sops
              pkgs.age
              pkgs.jq
              pkgs.git
            ];
            text = ''
              file="$(git rev-parse --show-toplevel)/.secrets.sops.yaml"
              out="''${CONTEXTMINE_SECRETS_HOST_DIR:?enter nix develop first}"

              decrypt() {
                # The parent stays private to the user; docker bind-mounts the
                # inner directory, whose files the container user must read.
                mkdir -p "$out"
                chmod 700 "$(dirname "$out")"
                chmod 755 "$out"
                find "$out" -mindepth 1 -maxdepth 1 -type f -delete
                if [ ! -f "$file" ]; then
                  return 0
                fi
                local json key
                json="$(sops decrypt --output-type json "$file")"
                for key in $(jq -r 'keys[]' <<<"$json"); do
                  if [[ ! "$key" =~ ^[a-z0-9_]+$ ]]; then
                    echo "contextmine-secrets: skipping invalid setting name '$key'" >&2
                    continue
                  fi
                  (umask 022 && jq -j --arg key "$key" '.[$key]' <<<"$json" >"$out/$key")
                done
              }

              case "''${1:-}" in
                edit)
                  key_file="''${SOPS_AGE_KEY_FILE:-''${XDG_CONFIG_HOME:-$HOME/.config}/sops/age/keys.txt}"
                  SOPS_AGE_RECIPIENTS="$(age-keygen -y "$key_file" | paste -sd, -)"
                  export SOPS_AGE_RECIPIENTS
                  sops edit "$file"
                  decrypt
                  ;;
                decrypt)
                  decrypt
                  ;;
                *)
                  echo "usage: contextmine-secrets edit|decrypt" >&2
                  echo "  edit     edit .secrets.sops.yaml (e.g. openai_api_key: sk-...)" >&2
                  echo "  decrypt  write one file per setting to \$CONTEXTMINE_SECRETS_HOST_DIR" >&2
                  exit 2
                  ;;
              esac
            '';
          };
        in
        {
          _module.args.pkgs = import inputs.nixpkgs {
            inherit system;
            overlays = [ inputs.rust-overlay.overlays.default ];
          };

          # `nix fmt` / `treefmt`; also exposed as a `nix flake check` check.
          treefmt = {
            projectRootFile = "flake.nix";
            programs.nixfmt.enable = true;
            programs.ruff-format = {
              enable = true;
              # Same ruff as uv.lock / pre-commit, so formatting doesn't drift.
              package = lib.addMetaAttrs { mainProgram = "ruff"; } pythonSet.ruff;
              # Generated protobuf code, also excluded in [tool.ruff].
              excludes = [
                "*_pb2.py"
                "*_pb2.pyi"
              ];
            };
            programs.rustfmt = {
              enable = true;
              package = rustToolchain;
            };
          };

          devShells.default = pkgs.mkShell {
            packages = [
              # Python workspace incl. dev group (ruff, ty, pytest, pre-commit)
              virtualenv
              pkgs.uv

              # Frontend (apps/web)
              pkgs.nodejs_24

              # Crawler (rust/spider_md)
              rustToolchain
              pkgs.pkg-config
              pkgs.openssl

              # Local database / tooling
              pkgs.postgresql_16
              pkgs.git
              config.treefmt.build.wrapper

              # Secrets for docker-compose.local.yml
              contextmine-secrets
              pkgs.sops
              pkgs.age
            ];

            env = {
              # uv must not manage the environment; Nix does. `uv run` uses it as is.
              UV_NO_SYNC = "1";
              UV_PYTHON = "${virtualenv}/bin/python";
              UV_PROJECT_ENVIRONMENT = "${virtualenv}";
              UV_PYTHON_DOWNLOADS = "never";
            };

            shellHook = ''
              export REPO_ROOT=$(git rev-parse --show-toplevel)
              # The editable .pth files expand $REPO_ROOT at runtime, which static tools
              # like ty can't evaluate. PYTHONPATH makes the members visible to them too.
              export PYTHONPATH=${
                lib.concatMapStringsSep ":" (dir: "$REPO_ROOT/${dir}") (lib.attrValues workspaceMembers)
              }

              # Local single-developer stack: AUTH_MODE=local, secrets from sops.
              export COMPOSE_FILE="$REPO_ROOT/docker-compose.yml:$REPO_ROOT/docker-compose.local.yml"
              export CONTEXTMINE_SECRETS_HOST_DIR="''${XDG_RUNTIME_DIR:-''${TMPDIR:-/tmp}}/contextmine/secrets"
              contextmine-secrets decrypt || echo "contextmine-secrets: decryption failed" >&2
            '';
          };
        };
    };
}
