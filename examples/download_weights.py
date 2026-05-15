from __future__ import annotations

import sys

from fourx_mamba import DEFAULT_REPO, download_weights


def main() -> None:
    repo_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REPO
    path = download_weights(repo_id)
    print(path)


if __name__ == "__main__":
    main()
