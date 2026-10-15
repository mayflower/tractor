"""Store accepted batch widths; SQLite TEXT preserves Decimal without float conversion."""

import argparse
import sqlite3
from pathlib import Path

from tractor.jsonio import loads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results")
    parser.add_argument("database")
    args = parser.parse_args()
    written = skipped = 0
    with sqlite3.connect(args.database) as database:
        database.execute("CREATE TABLE IF NOT EXISTS products "
                         "(id TEXT PRIMARY KEY, width_mm TEXT NOT NULL)")
        for line in Path(args.results).read_text().splitlines():
            if not line.strip():
                continue
            result = loads(line)
            if result.get("status") != "extracted" or result.get("issues"):
                skipped += 1
                continue
            value = result["data"]["width_mm"]
            if value is None:
                skipped += 1
                continue
            database.execute(
                "INSERT INTO products(id, width_mm) VALUES (?, ?) "
                "ON CONFLICT(id) DO UPDATE SET width_mm=excluded.width_mm",
                (str(result["id"]), str(value)),
            )
            written += 1
    print(f"Stored {written} extracted item(s); skipped {skipped} review/error item(s).")


if __name__ == "__main__":
    main()
