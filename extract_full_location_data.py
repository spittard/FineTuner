import sys
import os

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from finetuner.data.dataset import CreateDataSet

def main():
    server = "TLG-DATA3\\TLG_DEV"
    database = "SQLWebRefTable"
    table = "AcctRef.Master"
    company_col = "Original"
    output_file = "companies_with_location.json"

    print(f"Initializing CreateDataSet...")
    with CreateDataSet("sqlserver") as dataset_creator:
        print(f"Connecting to {server}...")
        success = dataset_creator.connect(
            database,
            server=server,
            trusted_connection=True
        )

        if not success:
            print("Failed to connect!")
            return

        print(f"Extraction started (this may take several minutes)...")
        success = dataset_creator.create_dataset_with_location(
            table,
            output_file,
            company_column=company_col,
            city_column="City",
            state_column="State",
            row_column="Row",
            exclude_plugging=True,
            plugging_column="PluggingStatus"
        )

        if success:
            print(f"SUCCESS: Data extracted to {output_file}")
            print("NOTE: Plugging records (PluggingStatus='P') were excluded.")
            print("      The FAISS cache must be rebuilt before the new index takes effect.")
            print("      Run: python run_cache_server.py  (it will rebuild on first load)")
        else:
            print("FAILED: Extraction failed")

if __name__ == "__main__":
    main()
