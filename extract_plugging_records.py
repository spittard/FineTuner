import sys
import os
import json
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from finetuner.data.dataset import CreateDataSet

SERVER = "TLG-DATA3\\TLG_DEV"
DATABASE = "SQLWebRefTable"
TABLE = "AcctRef.Master"
OUTPUT_FILE = "plugging_records.json"


def main():
    parser = argparse.ArgumentParser(description='Extract plugging records from DB')
    parser.add_argument('--max-rows', type=int, default=None,
                        help='Limit extraction to N rows (default: all)')
    parser.add_argument('--output', default=OUTPUT_FILE,
                        help=f'Output file (default: {OUTPUT_FILE})')
    args = parser.parse_args()

    print("=" * 60)
    print("Plugging Records Extraction")
    print(f"  Server:   {SERVER}")
    print(f"  Database: {DATABASE}")
    print(f"  Table:    {TABLE}")
    print(f"  Output:   {args.output}")
    if args.max_rows:
        print(f"  Limit:    {args.max_rows} rows")
    print("=" * 60)

    with CreateDataSet("sqlserver") as dataset_creator:
        print(f"\nConnecting to {SERVER}...")
        success = dataset_creator.connect(
            DATABASE,
            server=SERVER,
            trusted_connection=True
        )

        if not success:
            print("FAILED: Could not connect to database.")
            sys.exit(1)

        print("\nExtracting PluggingStatus='P' records...")
        records = dataset_creator.extract_plugging_records(
            TABLE,
            company_column="Original",
            city_column="City",
            state_column="State",
            row_column="Row",
            plugging_column="PluggingStatus",
            max_rows=args.max_rows
        )

        if not records:
            print("WARNING: No plugging records found (or extraction failed).")
            sys.exit(1)

        print(f"\nSaving {len(records):,} plugging records to {args.output}...")
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        file_mb = os.path.getsize(args.output) / (1024 * 1024)
        print(f"\nDone.")
        print(f"  Records extracted: {len(records):,}")
        print(f"  Output file:       {args.output} ({file_mb:.1f} MB)")

        # Show a sample
        print("\nSample records (first 5):")
        for r in records[:5]:
            loc = f"{r.get('City', '')}, {r.get('State', '')}".strip(', ')
            print(f"  Row {r.get('ID')}: {r.get('Company Name')} — {loc}")


if __name__ == "__main__":
    main()
