#!/usr/bin/env python3
"""
Update PluggingStatus to 'C' for control set companies in SQL Server
Updates [SQLWebRefTable].[AcctRef].[Master].[PluggingStatus] = 'C' for companies in the control set
"""

import json
import os
import sys
import pyodbc
import argparse
from typing import List, Dict, Any

def load_control_set(filepath: str) -> List[str]:
    """Load company names from control set JSON file"""
    if not os.path.exists(filepath):
        print(f"ERROR: File '{filepath}' not found")
        return None
    
    print(f"Loading control set from: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        company_names = []
        for item in data:
            if isinstance(item, dict) and "Company Name" in item:
                company_names.append(item["Company Name"])
        
        if not company_names:
            print("ERROR: No company names found in control set")
            return None
        
        print(f"Loaded {len(company_names):,} company names from control set")
        return company_names
    
    except Exception as e:
        print(f"ERROR: Failed to load control set: {e}")
        import traceback
        traceback.print_exc()
        return None

def connect_sqlserver(server: str, database_name: str, user: str = None, password: str = None, 
                      port: int = 1433, driver: str = "ODBC Driver 17 for SQL Server", 
                      trusted_connection: bool = False):
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
                return None
            
            if port != 1433:
                connection_string = f"DRIVER={{{driver}}};SERVER={server},{port};DATABASE={database_name};UID={user};PWD={password}"
            else:
                connection_string = f"DRIVER={{{driver}}};SERVER={server};DATABASE={database_name};UID={user};PWD={password}"
        
        connection = pyodbc.connect(connection_string)
        auth_type = "Windows Authentication" if trusted_connection else "SQL Authentication"
        print(f"OK: Connected to SQL Server database: {database_name} using {auth_type}")
        return connection
    except Exception as e:
        print(f"ERROR: Failed to connect to SQL Server database: {e}")
        print("TIP: Make sure you have the appropriate ODBC driver installed")
        print("TIP: Common drivers: 'ODBC Driver 17 for SQL Server', 'ODBC Driver 18 for SQL Server'")
        if trusted_connection:
            print("TIP: For Windows Authentication, ensure your Windows account has access to the database")
        return None

def update_plugging_status(connection, company_names: List[str], table_schema: str = "AcctRef", 
                          table_name: str = "Master", company_column: str = "Original", 
                          dry_run: bool = False):
    """
    Update PluggingStatus to 'C' for companies in the control set
    
    Args:
        connection: SQL Server connection object
        company_names: List of company names to update
        table_schema: Schema name (default: AcctRef)
        table_name: Table name (default: Master)
        company_column: Column name containing company names (default: Original)
        dry_run: If True, only show what would be updated without making changes
    
    Returns:
        Tuple of (total_matched, total_updated)
    """
    if not connection:
        print("ERROR: No database connection")
        return (0, 0)
    
    try:
        cursor = connection.cursor()
        
        print(f"\n{'DRY RUN: ' if dry_run else ''}Updating PluggingStatus to 'C' for control set companies...")
        print(f"   Table: [{table_schema}].[{table_name}]")
        print(f"   Company Column: [{company_column}]")
        print(f"   Companies to match: {len(company_names):,}")
        
        # Use a temporary table approach for better performance with many companies
        # This avoids SQL Server's parameter limit and is more efficient
        
        # Create temporary table
        temp_table_name = "#ControlSetCompanies"
        print("\nCreating temporary table for company matching...")
        cursor.execute(f"""
            CREATE TABLE {temp_table_name} (
                CompanyName NVARCHAR(MAX) NOT NULL
            )
        """)
        
        # Insert company names into temporary table
        print(f"   Inserting {len(company_names):,} company names into temporary table...")
        insert_query = f"INSERT INTO {temp_table_name} (CompanyName) VALUES (?)"
        for company_name in company_names:
            cursor.execute(insert_query, company_name)
        
        # Count matches
        print("\nChecking how many records match...")
        count_query = f"""
            SELECT COUNT(*) 
            FROM [{table_schema}].[{table_name}] m
            INNER JOIN {temp_table_name} t ON m.[{company_column}] = t.CompanyName
        """
        cursor.execute(count_query)
        total_matched = cursor.fetchone()[0]
        print(f"   Found {total_matched:,} matching records")
        
        if total_matched == 0:
            print("   No matching records found. Nothing to update.")
            cursor.execute(f"DROP TABLE {temp_table_name}")
            cursor.close()
            return (0, 0)
        
        # Show sample of what will be updated
        sample_query = f"""
            SELECT TOP 10 m.[{company_column}], m.[PluggingStatus]
            FROM [{table_schema}].[{table_name}] m
            INNER JOIN {temp_table_name} t ON m.[{company_column}] = t.CompanyName
        """
        cursor.execute(sample_query)
        samples = cursor.fetchall()
        print(f"\n   Sample of records that will be updated:")
        for i, (company, status) in enumerate(samples[:5], 1):
            status_str = status if status else "(NULL)"
            print(f"      {i}. {company} (current status: {status_str})")
        
        if dry_run:
            print(f"\nDRY RUN: Would update {total_matched:,} records")
            print("   Run without --dry-run to perform the actual update")
            cursor.execute(f"DROP TABLE {temp_table_name}")
            cursor.close()
            return (total_matched, 0)
        
        # Perform the update
        print(f"\nUpdating {total_matched:,} records...")
        update_query = f"""
            UPDATE m
            SET m.[PluggingStatus] = 'C'
            FROM [{table_schema}].[{table_name}] m
            INNER JOIN {temp_table_name} t ON m.[{company_column}] = t.CompanyName
        """
        
        cursor.execute(update_query)
        rows_updated = cursor.rowcount
        
        # Clean up temporary table
        cursor.execute(f"DROP TABLE {temp_table_name}")
        
        # Commit the transaction
        connection.commit()
        cursor.close()
        
        print(f"OK: Successfully updated {rows_updated:,} records")
        return (total_matched, rows_updated)
        
    except Exception as e:
        print(f"ERROR: Failed to update records: {e}")
        import traceback
        traceback.print_exc()
        try:
            # Try to clean up temporary table if it exists
            cursor.execute(f"DROP TABLE {temp_table_name}")
        except:
            pass
        if connection:
            connection.rollback()
        return (0, 0)

def main():
    parser = argparse.ArgumentParser(
        description='Update PluggingStatus to C for control set companies in SQL Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Windows Authentication (default)
  python update_control_set_plugging_status.py companies_control_set.json --server localhost --database SQLWebRefTable
  
  # SQL Authentication
  python update_control_set_plugging_status.py companies_control_set.json --server localhost --database SQLWebRefTable --user sa --password password
  
  # Dry run (preview changes without updating)
  python update_control_set_plugging_status.py companies_control_set.json --server localhost --database SQLWebRefTable --dry-run
  
  # Custom table/column names
  python update_control_set_plugging_status.py companies_control_set.json --server localhost --database SQLWebRefTable --table-schema AcctRef --table-name Master --company-column Original
        """
    )
    
    parser.add_argument('control_set', help='Path to control set JSON file (e.g., companies_control_set.json)')
    parser.add_argument('--server', required=True, help='SQL Server instance name')
    parser.add_argument('--database', required=True, help='Database name')
    parser.add_argument('--user', help='SQL Server username (required for SQL Authentication)')
    parser.add_argument('--password', help='SQL Server password (required for SQL Authentication)')
    parser.add_argument('--port', type=int, default=1433, help='SQL Server port (default: 1433)')
    parser.add_argument('--driver', default='ODBC Driver 17 for SQL Server', 
                       help='ODBC driver (default: ODBC Driver 17 for SQL Server)')
    parser.add_argument('--trusted-connection', action='store_true', default=True,
                       help='Use Windows Authentication (default: True)')
    parser.add_argument('--no-trusted-connection', dest='trusted_connection', action='store_false',
                       help='Use SQL Authentication instead of Windows Authentication')
    parser.add_argument('--table-schema', default='AcctRef', help='Table schema name (default: AcctRef)')
    parser.add_argument('--table-name', default='Master', help='Table name (default: Master)')
    parser.add_argument('--company-column', default='Original', help='Column name containing company names (default: Original)')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without updating (dry run mode)')
    
    args = parser.parse_args()
    
    # Validate authentication
    if not args.trusted_connection and (not args.user or not args.password):
        print("ERROR: --user and --password are required when not using Windows Authentication")
        print("TIP: Use --trusted-connection (default) for Windows Authentication")
        sys.exit(1)
    
    # Load control set
    print("=" * 60)
    print("Step 1: Loading control set")
    print("=" * 60)
    company_names = load_control_set(args.control_set)
    if not company_names:
        print(f"ERROR: Failed to load control set from {args.control_set}")
        sys.exit(1)
    
    # Connect to database
    print("\n" + "=" * 60)
    print("Step 2: Connecting to SQL Server")
    print("=" * 60)
    connection = connect_sqlserver(
        server=args.server,
        database_name=args.database,
        user=args.user,
        password=args.password,
        port=args.port,
        driver=args.driver,
        trusted_connection=args.trusted_connection
    )
    
    if not connection:
        print("ERROR: Failed to connect to database")
        sys.exit(1)
    
    # Update PluggingStatus
    print("\n" + "=" * 60)
    print("Step 3: Updating PluggingStatus")
    print("=" * 60)
    total_matched, total_updated = update_plugging_status(
        connection=connection,
        company_names=company_names,
        table_schema=args.table_schema,
        table_name=args.table_name,
        company_column=args.company_column,
        dry_run=args.dry_run
    )
    
    # Close connection
    connection.close()
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"   Control set companies: {len(company_names):,}")
    print(f"   Records matched: {total_matched:,}")
    if args.dry_run:
        print(f"   Records that would be updated: {total_matched:,} (DRY RUN)")
    else:
        print(f"   Records updated: {total_updated:,}")
    print("=" * 60)
    
    if args.dry_run:
        print("\nNOTE: This was a dry run. No changes were made.")
        print("Run without --dry-run to perform the actual update.")

if __name__ == "__main__":
    main()

