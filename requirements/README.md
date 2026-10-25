# Dependency locks

These locks target Python 3.11. `functional.txt` is universal across supported
platforms and intentionally excludes real model and external review SDKs.
`runtime-linux-x86_64.txt` covers the default project dependencies on Linux
x86_64; it is not a Jetson/ARM lock or a certificate of CUDA compatibility.
`build.txt` locks wheel-building dependencies. All installs use SHA256 checks.

The input constraints preserve the direct package versions used for ordinary
functional validation. Resolve, review and test dependency updates explicitly.
Do not use the ordinary-function lock to run real detection.

Regenerate from the repository root with uv 0.12.5:

```text
uv pip compile requirements/functional.in --python-version 3.11 --universal --generate-hashes --default-index https://pypi.org/simple -o requirements/functional.txt
uv pip compile pyproject.toml -c requirements/runtime-constraints.txt --python-version 3.11 --python-platform x86_64-manylinux_2_28 --generate-hashes --default-index https://pypi.org/simple -o requirements/runtime-linux-x86_64.txt
uv pip compile requirements/build.in --python-version 3.11 --universal --generate-hashes --default-index https://pypi.org/simple -o requirements/build.txt
python scripts/check_dependency_locks.py
```

Use a separate environment for ordinary functionality:

```text
python -m pip install --require-hashes -r requirements/functional.txt
python scripts/run_local_functional_tests.py --suite all
```

The guarded test entry point adds `src/` to the import path; it does not require
installing the project's model dependencies. For the frontend, commit
`web/package-lock.json` and use `npm --prefix web ci`. CI checks ordinary backend
functionality, frontend tests/build, OpenAPI consistency, isolated browser
navigation and the non-root `functional-check` container target. None of these
checks constitutes real recognition, GPU or production acceptance.

Container OS layers use explicit distribution/major-version tags and can still
receive upstream patch updates. Archive the final image digest for an accepted
deployment; dependency locks alone do not make an OS image bit-for-bit immutable.
