import json
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
            if trusted_connection:
                # Windows Authentication
                if port != 1433:
                    connection_string = f"DRIVER={{{driver}}};SERVER={server},{port};DATABASE={database_name};Trusted_Connection=yes"
                else:
                    connection_string = f"DRIVER={{{driver}}};SERVER={server};DATABASE={database_name};Trusted_Connection=yes"
            else:
                # SQL Authentication
                if not user or not password:
                    print("ERROR: SQL Authentication requires both username and password")
                    return False
                
                if port != 1433:
                    connection_string = f"DRIVER={{{driver}}};SERVER={server},{port};DATABASE={database_name};UID={user};PWD={password}"
                else:
                    connection_string = f"DRIVER={{{driver}}};SERVER={server};DATABASE={database_name};UID={user};PWD={password}"
            
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
            print(f"   ✓ Extracted {len(formatted_data):,} unique company names from {table_name}.{column_name}{limit_info}")
            return formatted_data
            
        except Exception as e:
            print(f"ERROR: Error extracting data: {e}")
            return []
    
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
            print(f"💾 Saving {len(data):,} entries to {output_file}...")
            print("   This may take a while for large datasets...")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"   ✓ Successfully saved {len(data):,} entries to {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ ERROR: Error saving to JSON: {e}")
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
        print(f"🚀 Starting dataset creation process...")
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
            print(f"🎉 Dataset creation completed successfully!")
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
