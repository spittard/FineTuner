import pyodbc
import sys

def test_conn(server):
    print(f"Testing connection to: {server}")
    conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE=SQLWebRefTable;Trusted_Connection=yes"
    try:
        conn = pyodbc.connect(conn_str)
        print("Success!")
        conn.close()
        return True
    except Exception as e:
        print(f"Failed: {e}")
        return False

servers = ["TLG-DATA3\\TLG_DEV", "TLG-DATA3\TLG_DEV", "192.168.133.111\\TLG_DEV"] # IP from SSMS title bar?
for s in servers:
    if test_conn(s):
        print(f"\nWinning server string: {s}")
        break
