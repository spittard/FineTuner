import json
import os
import pyodbc
from typing import List, Dict, Any, Optional

class CreateDataSet:
    """Class to create training datasets from database tables"""
    
    def __init__(self, db_type: str = "sqlserver"):
        """
        Initialize the dataset creator
        
        Args:
            db_type: Type of database (currently supports "sqlserver")
        """
        self.db_type = db_type.lower()
        self.connection = None
        
    def connect_sqlserver(self, server: str, database_name: str, user: str = None, password: str = None, port: int = 1433, driver: str = "ODBC Driver 17 for SQL Server", trusted_connection: bool = False) -> bool:
        """Connect to SQL Server database"""
        try:
            # Build connection string
            # TrustServerCertificate=yes;Encrypt=no avoids ODBC Driver 17 SSL handshake
            # failures when the client TLS stack doesn't match the server's cipher suite.
            ssl_opts = "TrustServerCertificate=yes;Encrypt=no"
            if trusted_connection:
                # Windows Authentication
                if port != 1433:
                    connection_string = f"DRIVER={{{driver}}};SERVER={server},{port};DATABASE={database_name};Trusted_Connection=yes;{ssl_opts}"
                else:
                    connection_string = f"DRIVER={{{driver}}};SERVER={server};DATABASE={database_name};Trusted_Connection=yes;{ssl_opts}"
            else:
                # SQL Authentication
                if not user or not password:
                    print("ERROR: SQL Authentication requires both username and password")
                    return False
                
                if port != 1433:
                    connection_string = f"DRIVER={{{driver}}};SERVER={server},{port};DATABASE={database_name};UID={user};PWD={password};{ssl_opts}"
                else:
                    connection_string = f"DRIVER={{{driver}}};SERVER={server};DATABASE={database_name};UID={user};PWD={password};{ssl_opts}"
            
            self.connection = pyodbc.connect(connection_string)
            auth_type = "Windows Authentication" if trusted_connection else "SQL Authentication"
            print(f"OK: Connected to SQL Server database: {database_name} using {auth_type}")
            return True
        except Exception as e:
            print(f"ERROR: Failed to connect to SQL Server database: {e}")
            print("TIP: Make sure you have the appropriate ODBC driver installed")
            print("TIP: Common drivers: 'ODBC Driver 17 for SQL Server', 'ODBC Driver 18 for SQL Server'")
            if trusted_connection:
                print("TIP: For Windows Authentication, ensure your Windows account has access to the database")
            return False
    
    def connect(self, database_name: str, **kwargs) -> bool:
        """Connect to database based on type"""
        if self.db_type == "sqlserver":
            return self.connect_sqlserver(
                server=kwargs.get('server'),
                database_name=database_name,
                user=kwargs.get('user'),
                password=kwargs.get('password'),
                port=kwargs.get('port', 1433),
                driver=kwargs.get('driver', "ODBC Driver 17 for SQL Server"),
                trusted_connection=kwargs.get('trusted_connection', False)
            )
        else:
            print(f"ERROR: Unsupported database type: {self.db_type}")
            print("TIP: Currently only supports 'sqlserver'")
            return False
    
    def extract_data(self, table_name: str, column_name: str, max_rows: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Extract data from database table and format for FineTuner
        
        Args:
            table_name: Name of the table to extract from (can include schema)
            column_name: Name of the column containing company names
            max_rows: Maximum number of rows to extract (None for all rows)
            
        Returns:
            List of dictionaries in the format [{"Company Name": "value"}, ...]
        """
        if not self.connection:
            print("ERROR: No database connection. Call connect() first.")
            return []
        
        try:
            cursor = self.connection.cursor()
            
            # Parse table name for schema and table
            table_parts = table_name.split('.')
            if len(table_parts) == 2:
                schema_name = table_parts[0]
                actual_table_name = table_parts[1]
                # SQL Server query with proper bracket notation for schema-qualified tables
                if max_rows:
                    query = f"SELECT TOP {max_rows} [{column_name}] FROM (SELECT DISTINCT [{column_name}] FROM [{schema_name}].[{actual_table_name}] WHERE [{column_name}] IS NOT NULL AND [{column_name}] != '') AS distinct_data"
                else:
                    query = f"SELECT DISTINCT [{column_name}] FROM [{schema_name}].[{actual_table_name}] WHERE [{column_name}] IS NOT NULL AND [{column_name}] != ''"
            else:
                # SQL Server query with proper bracket notation for default schema tables
                if max_rows:
                    query = f"SELECT TOP {max_rows} [{column_name}] FROM (SELECT DISTINCT [{column_name}] FROM [{table_name}] WHERE [{column_name}] IS NOT NULL AND [{column_name}] != '') AS distinct_data"
                else:
                    query = f"SELECT DISTINCT [{column_name}] FROM [{table_name}] WHERE [{column_name}] IS NOT NULL AND [{column_name}] != ''"
            
            print(f"   Executing query: {query[:100]}...")
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            
            print(f"   Query returned {len(results):,} rows")
            
            # Format data for FineTuner
            print("   Formatting data for FineTuner...")
            formatted_data = []
            for i, row in enumerate(results):
                company_name = row[0].strip() if row[0] else ""
                if company_name:  # Only add non-empty names
                    formatted_data.append({"Company Name": company_name})
                
                # Show progress every 100,000 rows
                if (i + 1) % 100000 == 0:
                    print(f"   Processed {i + 1:,} rows...")
            
            limit_info = f" (limited to {max_rows:,} rows)" if max_rows else ""
            print(f"   Extracted {len(formatted_data):,} unique company names from {table_name}.{column_name}{limit_info}")
            return formatted_data
            
        except Exception as e:
            print(f"ERROR: Error extracting data: {e}")
            return []
    
    def extract_data_with_location(self, table_name: str, company_column: str = "Original",
                                     city_column: str = "City", state_column: str = "State",
                                     row_column: str = "Row",
                                     max_rows: Optional[int] = None,
                                     exclude_plugging: bool = False,
                                     plugging_column: str = "PluggingStatus") -> List[Dict[str, Any]]:
        """
        Extract company data with location (city, state), record counts, and row ID.

        Args:
            table_name: Name of the table to extract from (can include schema, e.g., 'AcctRef.Master')
            company_column: Name of the column containing company names (default: 'Original')
            city_column: Name of the column containing city (default: 'City')
            state_column: Name of the column containing state (default: 'State')
            row_column: Name of the unique row identifier column (default: 'Row')
            max_rows: Maximum number of rows to extract (None for all rows)
            exclude_plugging: When True, excludes rows where plugging_column = 'P' (default: False)
            plugging_column: Column used to filter plugging records (default: 'PluggingStatus')

        Returns:
            List of dictionaries in the format:
            [{"ID": N, "Company Name": "value", "City": "value", "State": "value", "Count": N}, ...]
        """
        if not self.connection:
            print("ERROR: No database connection. Call connect() first.")
            return []

        try:
            cursor = self.connection.cursor()

            # Parse table name for schema and table
            table_parts = table_name.split('.')
            if len(table_parts) == 2:
                schema_name = table_parts[0]
                actual_table_name = table_parts[1]
                table_ref = f"[{schema_name}].[{actual_table_name}]"
            else:
                table_ref = f"[{table_name}]"

            # Build WHERE clause
            where_parts = [f"[{company_column}] IS NOT NULL", f"[{company_column}] != ''"]
            if exclude_plugging:
                where_parts.append(f"([{plugging_column}] IS NULL OR [{plugging_column}] != 'P')")
            where_clause = " AND ".join(where_parts)

            # MIN([Row]) picks a stable representative row ID per group
            select_cols = f"MIN([{row_column}]) as RowID, [{company_column}], [{city_column}], [{state_column}], COUNT(*) as RecordCount"
            group_by = f"[{company_column}], [{city_column}], [{state_column}]"

            if max_rows:
                query = f"""
                    SELECT TOP {max_rows} {select_cols}
                    FROM {table_ref}
                    WHERE {where_clause}
                    GROUP BY {group_by}
                    ORDER BY [{company_column}]
                """
            else:
                query = f"""
                    SELECT {select_cols}
                    FROM {table_ref}
                    WHERE {where_clause}
                    GROUP BY {group_by}
                    ORDER BY [{company_column}]
                """

            plug_info = " (excluding PluggingStatus='P')" if exclude_plugging else ""
            print(f"   Executing query{plug_info}: {query[:120].strip()}...")
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()

            print(f"   Query returned {len(results):,} rows")

            # Format data with location info
            print("   Formatting data with location info...")
            formatted_data = []
            for i, row in enumerate(results):
                row_id = row[0]
                company_name = row[1].strip() if row[1] else ""
                city = row[2].strip() if row[2] else ""
                state = row[3].strip() if row[3] else ""
                count = row[4] if row[4] else 0

                if company_name:
                    formatted_data.append({
                        "ID": row_id,
                        "Company Name": company_name,
                        "City": city,
                        "State": state,
                        "Count": count
                    })

                if (i + 1) % 100000 == 0:
                    print(f"   Processed {i + 1:,} rows...")

            limit_info = f" (limited to {max_rows:,} rows)" if max_rows else ""
            print(f"   Extracted {len(formatted_data):,} company records with location from {table_name}{limit_info}")
            return formatted_data

        except Exception as e:
            print(f"ERROR: Error extracting data with location: {e}")
            import traceback
            traceback.print_exc()
            return []

    def extract_data_with_location_industry(self, table_name: str, company_column: str = "Original",
                                              city_column: str = "City", state_column: str = "State",
                                              row_column: str = "Row", sic_column: str = "SIC",
                                              max_rows: Optional[int] = None,
                                              exclude_plugging: bool = False,
                                              plugging_column: str = "PluggingStatus") -> List[Dict[str, Any]]:
        """
        Extract company data with location, record counts, row ID, and SIC industry code.

        Same as extract_data_with_location but adds MIN(SIC) per group.

        Returns:
            List of dicts: [{"ID": N, "Company Name": ..., "City": ..., "State": ..., "Count": N, "SIC": ...}, ...]
        """
        if not self.connection:
            print("ERROR: No database connection. Call connect() first.")
            return []

        try:
            cursor = self.connection.cursor()

            table_parts = table_name.split('.')
            if len(table_parts) == 2:
                schema_name = table_parts[0]
                actual_table_name = table_parts[1]
                table_ref = f"[{schema_name}].[{actual_table_name}]"
            else:
                table_ref = f"[{table_name}]"

            where_parts = [f"[{company_column}] IS NOT NULL", f"[{company_column}] != ''"]
            if exclude_plugging:
                where_parts.append(f"([{plugging_column}] IS NULL OR [{plugging_column}] != 'P')")
            where_clause = " AND ".join(where_parts)

            select_cols = (
                f"MIN([{row_column}]) as RowID, [{company_column}], [{city_column}], [{state_column}], "
                f"COUNT(*) as RecordCount, "
                f"LTRIM(RTRIM(MIN(CAST([{sic_column}] AS NVARCHAR(100))))) as SICCode"
            )
            group_by = f"[{company_column}], [{city_column}], [{state_column}]"

            if max_rows:
                query = f"SELECT TOP {max_rows} {select_cols} FROM {table_ref} WHERE {where_clause} GROUP BY {group_by} ORDER BY [{company_column}]"
            else:
                query = f"SELECT {select_cols} FROM {table_ref} WHERE {where_clause} GROUP BY {group_by} ORDER BY [{company_column}]"

            plug_info = " (excluding PluggingStatus='P')" if exclude_plugging else ""
            print(f"   Executing query{plug_info}: {query[:120].strip()}...")
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()

            print(f"   Query returned {len(results):,} rows")
            print("   Formatting data with location + SIC...")
            formatted_data = []
            empty_labels = {"", "TBD", "tbd", "N/A", "NONE", "UNKNOWN"}
            for i, row in enumerate(results):
                company_name = row[1].strip() if row[1] else ""
                if not company_name:
                    continue
                sic = (row[5] or "").strip()
                if sic in empty_labels:
                    sic = ""
                formatted_data.append({
                    "ID": row[0],
                    "Company Name": company_name,
                    "City": row[2].strip() if row[2] else "",
                    "State": row[3].strip() if row[3] else "",
                    "Count": row[4] if row[4] else 0,
                    "SIC": sic,
                })
                if (i + 1) % 100000 == 0:
                    print(f"   Processed {i + 1:,} rows...")

            limit_info = f" (limited to {max_rows:,} rows)" if max_rows else ""
            n_sic = sum(1 for r in formatted_data if r["SIC"])
            print(f"   Extracted {len(formatted_data):,} records{limit_info}; {n_sic:,} have a SIC value")
            return formatted_data

        except Exception as e:
            print(f"ERROR: Error extracting data with location+industry: {e}")
            import traceback
            traceback.print_exc()
            return []

    def create_dataset_with_location_industry(self, table_name: str, output_file: str,
                                               company_column: str = "Original",
                                               city_column: str = "City",
                                               state_column: str = "State",
                                               row_column: str = "Row",
                                               sic_column: str = "SIC",
                                               max_rows: Optional[int] = None,
                                               exclude_plugging: bool = False,
                                               plugging_column: str = "PluggingStatus") -> bool:
        """Complete workflow: extract data with location + SIC and save to JSON."""
        print(f"Starting dataset creation with location + industry data...")
        print(f"   Table: {table_name}  |  Output: {output_file}")
        print(f"   Exclude Plugging: {exclude_plugging}")
        if max_rows:
            print(f"   Max rows: {max_rows:,}")

        data = self.extract_data_with_location_industry(
            table_name, company_column, city_column, state_column,
            row_column=row_column, sic_column=sic_column,
            max_rows=max_rows, exclude_plugging=exclude_plugging,
            plugging_column=plugging_column,
        )
        if not data:
            return False

        success = self.save_to_json(data, output_file)
        if success:
            print(f"Dataset creation completed successfully!")
            print(f"   Total entries: {len(data):,}")
            print(f"   File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
        return success

    def extract_plugging_records(self, table_name: str, company_column: str = "Original",
                                  city_column: str = "City", state_column: str = "State",
                                  row_column: str = "Row",
                                  plugging_column: str = "PluggingStatus",
                                  max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Extract individual Plugging Records (PluggingStatus = 'P'), one entry per source row.

        Unlike extract_data_with_location (which groups), this returns every individual row
        so each plugging record can be matched independently.

        Args:
            table_name: Name of the table (e.g., 'AcctRef.Master')
            company_column: Column containing company names (default: 'Original')
            city_column: Column containing city (default: 'City')
            state_column: Column containing state (default: 'State')
            row_column: Unique row identifier column (default: 'Row')
            plugging_column: Column used to identify plugging records (default: 'PluggingStatus')
            max_rows: Maximum rows to extract (None for all)

        Returns:
            List of dicts: [{"ID": row_val, "Company Name": ..., "City": ..., "State": ...}, ...]
        """
        if not self.connection:
            print("ERROR: No database connection. Call connect() first.")
            return []

        try:
            cursor = self.connection.cursor()

            table_parts = table_name.split('.')
            if len(table_parts) == 2:
                schema_name = table_parts[0]
                actual_table_name = table_parts[1]
                table_ref = f"[{schema_name}].[{actual_table_name}]"
            else:
                table_ref = f"[{table_name}]"

            top_clause = f"TOP {max_rows} " if max_rows else ""
            query = f"""
                SELECT {top_clause}[{row_column}], [{company_column}], [{city_column}], [{state_column}]
                FROM {table_ref}
                WHERE [{plugging_column}] = 'P'
                  AND [{company_column}] IS NOT NULL AND [{company_column}] != ''
                ORDER BY [{row_column}]
            """

            print(f"   Executing plugging records query: {query[:120].strip()}...")
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()

            print(f"   Query returned {len(results):,} plugging records")

            formatted_data = []
            for i, row in enumerate(results):
                row_id = row[0]
                company_name = row[1].strip() if row[1] else ""
                city = row[2].strip() if row[2] else ""
                state = row[3].strip() if row[3] else ""

                if company_name:
                    formatted_data.append({
                        "ID": row_id,
                        "Company Name": company_name,
                        "City": city,
                        "State": state
                    })

                if (i + 1) % 100000 == 0:
                    print(f"   Processed {i + 1:,} rows...")

            limit_info = f" (limited to {max_rows:,} rows)" if max_rows else ""
            print(f"   Extracted {len(formatted_data):,} plugging records from {table_name}{limit_info}")
            return formatted_data

        except Exception as e:
            print(f"ERROR: Error extracting plugging records: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def create_dataset_with_location(self, table_name: str, output_file: str,
                                      company_column: str = "Original",
                                      city_column: str = "City",
                                      state_column: str = "State",
                                      row_column: str = "Row",
                                      max_rows: Optional[int] = None,
                                      exclude_plugging: bool = False,
                                      plugging_column: str = "PluggingStatus") -> bool:
        """
        Complete workflow: extract data with location and save to JSON.

        Args:
            table_name: Name of the table to extract from (e.g., 'AcctRef.Master')
            output_file: Path to output JSON file
            company_column: Column name for company names (default: 'Original')
            city_column: Column name for city (default: 'City')
            state_column: Column name for state (default: 'State')
            row_column: Unique row identifier column (default: 'Row')
            max_rows: Maximum number of rows to extract (None for all rows)
            exclude_plugging: Exclude rows where PluggingStatus = 'P' (default: False)
            plugging_column: Column name for plugging status (default: 'PluggingStatus')

        Returns:
            True if successful, False otherwise
        """
        print(f"Starting dataset creation with location data...")
        print(f"   Table: {table_name}")
        print(f"   Company Column: {company_column}")
        print(f"   City Column: {city_column}")
        print(f"   State Column: {state_column}")
        print(f"   Row ID Column: {row_column}")
        print(f"   Exclude Plugging (PluggingStatus='P'): {exclude_plugging}")
        print(f"   Output: {output_file}")
        if max_rows:
            print(f"   Max rows: {max_rows:,}")
        else:
            print(f"   Max rows: No limit (all rows)")
        print()

        data = self.extract_data_with_location(
            table_name, company_column, city_column, state_column,
            row_column=row_column, max_rows=max_rows,
            exclude_plugging=exclude_plugging, plugging_column=plugging_column
        )
        if not data:
            return False
        
        success = self.save_to_json(data, output_file)
        if success:
            print(f"Dataset creation with location completed successfully!")
            print(f"   Total entries: {len(data):,}")
            print(f"   File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
        
        return success

    def save_to_json(self, data: List[Dict[str, str]], output_file: str) -> bool:
        """
        Save extracted data to JSON file
        
        Args:
            data: List of dictionaries from extract_data()
            output_file: Path to output JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Saving {len(data):,} entries to {output_file}...")
            print("   This may take a while for large datasets...")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"   Successfully saved {len(data):,} entries to {output_file}")
            return True
            
        except Exception as e:
            print(f"ERROR: Error saving to JSON: {e}")
            return False
    
    def create_dataset(self, table_name: str, column_name: str, output_file: str, max_rows: Optional[int] = None) -> bool:
        """
        Complete workflow: extract data and save to JSON
        
        Args:
            table_name: Name of the table to extract from
            column_name: Name of the column containing company names
            output_file: Path to output JSON file
            max_rows: Maximum number of rows to extract (None for all rows)
            
        Returns:
            True if successful, False otherwise
        """
        print(f"Starting dataset creation process...")
        print(f"   Table: {table_name}")
        print(f"   Column: {column_name}")
        print(f"   Output: {output_file}")
        if max_rows:
            print(f"   Max rows: {max_rows:,}")
        else:
            print(f"   Max rows: No limit (all rows)")
        print()
        
        data = self.extract_data(table_name, column_name, max_rows)
        if not data:
            return False
        
        success = self.save_to_json(data, output_file)
        if success:
            print(f"Dataset creation completed successfully!")
            print(f"   Total entries: {len(data):,}")
            print(f"   File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
        
        return success
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            print("OK: Database connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

if __name__ == "__main__":
    # Example usage when run directly
    print("CreateDataSet - SQL Server to JSON Dataset Converter")
    print("Use this class to extract company names from SQL Server databases")
    print("\nExample usage:")
    print("from CreateDataSet import CreateDataSet")
    print("dataset_creator = CreateDataSet('sqlserver')")
    print("dataset_creator.connect('mydb', server='localhost', trusted_connection=True)")
    print("dataset_creator.create_dataset('companies', 'company_name', 'output.json')")
    print("dataset_creator.create_dataset('companies', 'company_name', 'output.json', max_rows=1000)")
    print("\nSQL Server with Windows Authentication:")
    print("dataset_creator = CreateDataSet('sqlserver')")
    print("dataset_creator.connect('mydb', server='localhost', trusted_connection=True)")
    print("dataset_creator.create_dataset('companies', 'company_name', 'output.json', max_rows=500)")
    print("\nSQL Server with SQL Authentication:")
    print("dataset_creator = CreateDataSet('sqlserver')")
    print("dataset_creator.connect('mydb', server='localhost', user='sa', password='password')")
    print("dataset_creator.create_dataset('companies', 'company_name', 'output.json', max_rows=2000)")
    print("\n--- NEW: Extract with Location Data ---")
    print("dataset_creator = CreateDataSet('sqlserver')")
    print("dataset_creator.connect('SQLWebRefTable', server='localhost', trusted_connection=True)")
    print("dataset_creator.create_dataset_with_location(")
    print("    table_name='AcctRef.Master',")
    print("    output_file='companies_with_location.json',")
    print("    company_column='Original',")
    print("    city_column='City',")
    print("    state_column='State'")
    print(")")
