import pyodbc
import json
import argparse
import os

def extract_sample_data(output_file, sample_size=100000):
    print(f"Connecting to SQL Server to extract random {sample_size:,} records...")
    
    # Matches CreateDataSet logic
    conn_str = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=TLG-DATA3\\TLG_DEV;DATABASE=SQLWebRefTable;Trusted_Connection=yes"
    
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        # Use simple TOP queries on AcctRef.Master
        query = f"""
        SELECT TOP {sample_size} 
            [Original] as 'Company Name',
            [City],
            [State]
        FROM [SQLWebRefTable].[AcctRef].[Master]
        WHERE [Original] IS NOT NULL AND LEN([Original]) > 0
        ORDER BY NEWID()
        """
        
        print("Executing query (this might take a moment)...")
        cursor.execute(query)
        
        columns = [column[0] for column in cursor.description]
        results = []
        
        row_count = 0
        while True:
            rows = cursor.fetchmany(10000)
            if not rows:
                break
                
            for row in rows:
                item = dict(zip(columns, row))
                # Clean strings
                if item.get('Company Name'):
                    item['Company Name'] = str(item['Company Name']).strip()
                if item.get('City'):
                    item['City'] = str(item['City']).strip()
                if item.get('State'):
                    item['State'] = str(item['State']).strip()
                
                results.append(item)
                row_count += 1
                
            print(f"   Extracted {row_count:,} rows...", end="\r")
            
        print(f"\nExtracted {len(results):,} total records.")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
            
        print(f"Saved sample dataset to {output_file}")
        return True
        
    except Exception as e:
        print(f"Error extracting data: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract random sample from Company Master')
    parser.add_argument('--output', default='companies_sample_100k.json', help='Output JSON file')
    parser.add_argument('--size', type=int, default=100000, help='Sample size')
    
    args = parser.parse_args()
    
    extract_sample_data(args.output, args.size)
