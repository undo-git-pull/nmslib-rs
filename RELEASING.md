# Releasing

nmslib-rs is published to PyPI as `nmslib-rs` (imported as `nmslib_rs`). Two GitHub Actions workflows handle it.

`.github/workflows/ci.yml` runs on every push and pull request. On Ubuntu and macOS, once each for Python 3.11, 3.12, 3.13 and 3.14, it checks formatting with `cargo fmt --check`, runs `cargo test`, builds the extension with `maturin develop --release`, and runs `pytest tests`.

`.github/workflows/release.yml` runs when a tag matching `v*` is pushed. It builds wheels for CPython 3.11, 3.12, 3.13 and 3.14 on these platforms:

| Platform | Runner |
|---|---|
| Linux x86_64 (manylinux) | `ubuntu-latest` |
| Linux aarch64 (manylinux) | `ubuntu-24.04-arm` |
| macOS x86_64 | `macos-15-intel` |
| macOS arm64 | `macos-latest` |
| Windows x64 | `windows-latest` |

Each wheel is installed into a fresh environment for its Python version and the Python tests are run against it before anything is uploaded. The build fails if it produces a wheel for any other Python version. The workflow also builds an sdist, checks that the tag equals `v` plus the `version` in `Cargo.toml`, publishes everything to PyPI, and creates a GitHub release with the same files attached. Publishing uses PyPI trusted publishing, so no API token is stored in the repository.

The x86_64 wheels detect AVX2 and AVX-512 at runtime, so one wheel serves all x86_64 CPUs.

## One-time setup

These steps only need doing before the first release.

1. On PyPI, open <https://pypi.org/manage/account/publishing/> and add a pending trusted publisher with these values:

   | Field | Value |
   |---|---|
   | PyPI project name | `nmslib-rs` |
   | Owner | `undo-git-pull` |
   | Repository name | `nmslib-rs` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

   The pending publisher turns into a normal one when the first upload creates the project.

2. The `pypi` environment exists in the GitHub repository under Settings, Environments. To require a manual approval before each upload, add yourself as a required reviewer there.

## Cutting a release

1. Bump `version` in `Cargo.toml`. `pyproject.toml` reads the version from there, so it needs no edit. Run `cargo check` so `Cargo.lock` picks up the new version.

2. Commit and push to `main`, then wait for CI to pass.

   ```sh
   git commit -am "Release 0.2.0"
   git push origin main
   ```

3. Tag the commit and push the tag.

   ```sh
   git tag v0.2.0
   git push origin v0.2.0
   ```

4. Watch the run with `gh run watch` or on the Actions tab. If the `pypi` environment has a required reviewer, approve the `publish` job when it is waiting.

5. Check the result.

   ```sh
   python -m venv /tmp/nmslib-rs-check
   /tmp/nmslib-rs-check/bin/pip install nmslib-rs==0.2.0
   /tmp/nmslib-rs-check/bin/python -c "import nmslib_rs; print(nmslib_rs.simd_level())"
   ```

## Dry runs

The release workflow can also be started by hand from the Actions tab or with:

```sh
gh workflow run release.yml --ref main
```

A manual run builds and tests every wheel and the sdist and uploads them as workflow artifacts. It skips the version check, the PyPI upload and the GitHub release. Use it after changing dependencies or the workflow itself.

## When a release fails

PyPI never accepts the same version twice, even after a file is deleted. If the workflow fails before the `publish` job, fix the problem, move the tag and push it again:

```sh
git tag -f v0.2.0 <fixed-commit>
git push -f origin v0.2.0
```

If `publish` uploaded some files and then failed, bump to the next patch version and release that instead.
