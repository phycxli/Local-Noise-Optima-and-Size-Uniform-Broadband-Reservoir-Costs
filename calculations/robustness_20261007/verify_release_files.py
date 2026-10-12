"""Read-only verification of the released calculations and archive members."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--available", action="store_true", help="Permit missing large result archives and explicitly report partial coverage")
    parser.add_argument("--extracted-results", action="store_true", help="Also verify the complete extracted results directory")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "FILE_MANIFEST_SHA256.json").read_text(encoding="utf-8"))
    source = json.loads((ROOT / "SOURCE_SHA256.json").read_text(encoding="utf-8"))["files"]
    checked, members, missing = 0, 0, []
    for name, expected in manifest["release_files"].items():
        path = ROOT / name
        if not path.is_file() and args.available and name in manifest["result_archives"]:
            missing.append(name)
            continue
        assert path.is_file(), f"Missing released file: {name}"
        assert path.stat().st_size == expected["bytes"], name
        assert digest(path) == expected["sha256"], name
        checked += 1
    expected_results = {name for name in source if name.startswith("results/")}
    seen = set()
    for archive_name, names in manifest["result_archives"].items():
        path = ROOT / archive_name
        if not path.is_file() and args.available:
            continue
        with zipfile.ZipFile(path) as archive:
            assert len(archive.namelist()) == len(set(archive.namelist()))
            assert set(archive.namelist()) == set(names), archive_name
            for name in archive.namelist():
                safe = PurePosixPath(name)
                assert not safe.is_absolute() and ".." not in safe.parts
                assert name in expected_results and name not in seen, name
                data = archive.read(name)
                assert len(data) == source[name]["bytes"], name
                assert hashlib.sha256(data).hexdigest() == source[name]["sha256"], name
                seen.add(name)
                members += 1
    if not missing:
        assert seen == expected_results
    if args.extracted_results:
        for name in sorted(expected_results):
            path = ROOT / name
            assert path.is_file(), f"Missing extracted result: {name}"
            assert path.stat().st_size == source[name]["bytes"] and digest(path) == source[name]["sha256"], name
    print(json.dumps({"all_checks_passed": True, "full_inventory_verified": not missing,
                      "released_files_checked": checked, "result_members_checked": members,
                      "missing_result_archives": missing, "extracted_results_checked": args.extracted_results}, indent=2))


if __name__ == "__main__":
    main()
