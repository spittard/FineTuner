# CreateDataset.ps1
# PowerShell script for creating training datasets from SQL Server
# Uses FineTuner.py with create-dataset mode

param(
    [string]$Server = "localhost",
    [string]$Database = "SQLWebRefTable",
    [string]$Table = "AcctRef.Master",
    [string]$Column = "Original",
    [string]$OutputFile = "training_data.json",
    [string]$Driver = "ODBC Driver 17 for SQL Server",
    [int]$Port = 1433,
    [switch]$TrustedConnection = $true,
    [string]$User = "",
    [string]$Password = "",
    [int]$MaxRows = 0,
    [switch]$Help
)

function Show-Help {
    Write-Host @"
CreateDataset.ps1 - PowerShell script for creating training datasets from SQL Server

USAGE:
    .\CreateDataset.ps1 [OPTIONS]

PARAMETERS:
    -Server <string>           SQL Server instance name (default: prompts for input)
    -Database <string>         Database name (default: prompts for input)
    -Table <string>            Table name to extract from (can include schema, e.g., "AcctRef.Master")
    -Column <string>           Column name containing company names (default: prompts for input)
    -OutputFile <string>       Output JSON file path (default: prompts for input)
    -Driver <string>           ODBC driver (default: "ODBC Driver 17 for SQL Server")
    -Port <int>                SQL Server port (default: 1433)
    -TrustedConnection         Use Windows Authentication (Trusted_Connection=yes) [DEFAULT]
    -User <string>             SQL Server username (for SQL Authentication)
    -Password <string>         SQL Server password (for SQL Authentication)
    -MaxRows <int>             Maximum number of rows to extract (default: 0 for all)
    -Help                      Show this help message

EXAMPLES:
    # Interactive mode with Windows Authentication (default)
    .\CreateDataset.ps1

    # Command line mode with Windows Authentication and schema-qualified table
    .\CreateDataset.ps1 -Server "localhost" -Database "SQLWebRefTable" -Table "AcctRef.Master" -Column "Original" -OutputFile "training_data.json" -TrustedConnection

    # Command line mode with Windows Authentication (default schema)
    .\CreateDataset.ps1 -Server "localhost" -Database "CompanyDB" -Table "Companies" -Column "CompanyName" -OutputFile "companies.json" -TrustedConnection

    # Command line mode with SQL Authentication
    .\CreateDataset.ps1 -Server "localhost" -Database "CompanyDB" -Table "Companies" -Column "CompanyName" -OutputFile "companies.json" -User "sa" -Password "password"

    # Use custom ODBC driver with Windows Authentication
    .\CreateDataset.ps1 -Server "localhost" -Database "CompanyDB" -Table "Companies" -Column "CompanyName" -OutputFile "companies.json" -Driver "ODBC Driver 18 for SQL Server" -TrustedConnection

    # Limit to 1000 rows with Windows Authentication
    .\CreateDataset.ps1 -Server "localhost" -Database "CompanyDB" -Table "Companies" -Column "CompanyName" -OutputFile "companies.json" -TrustedConnection -MaxRows 1000

    # Limit to 500 rows with SQL Authentication
    .\CreateDataset.ps1 -Server "localhost" -Database "CompanyDB" -Table "Companies" -Column "CompanyName" -OutputFile "companies.json" -User "sa" -Password "password" -MaxRows 500

NOTES:
    - By default, Windows Authentication (TrustedConnection) is used
    - Use -User and -Password for SQL Authentication
    - Table names can include schemas (e.g., "SchemaName.TableName")
    - Use -MaxRows to limit the number of rows extracted (0 = all rows)
    - The script will prompt for missing required parameters
    - Connection testing is available to verify database access before extraction

"@ -ForegroundColor Cyan
}

function Test-PythonInstallation {
    try {
        $pythonVersion = python --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] Python found: $pythonVersion" -ForegroundColor Green
            return $true
        }
    }
    catch {
        Write-Host "[ERROR] Python not found in PATH" -ForegroundColor Red
        return $false
    }
    return $false
}

function Test-FineTunerFiles {
    $fineTunerExists = Test-Path "FineTuner.py"
    $createDataSetExists = Test-Path "CreateDataSet.py"
    
    if (-not $fineTunerExists) {
        Write-Host "[ERROR] FineTuner.py not found in current directory" -ForegroundColor Red
        return $false
    }
    
    if (-not $createDataSetExists) {
        Write-Host "[ERROR] CreateDataSet.py not found in current directory" -ForegroundColor Red
        return $false
    }
    
    Write-Host "[OK] Required files found" -ForegroundColor Green
    return $true
}

function Get-UserInput {
    param(
        [string]$Prompt,
        [string]$DefaultValue = "",
        [bool]$Required = $true
    )
    
    do {
        if ($DefaultValue) {
            $input = Read-Host "$Prompt [$DefaultValue]"
            if ($input -eq "") {
                $input = $DefaultValue
            }
        } else {
            $input = Read-Host $Prompt
        }
        
        if ($Required -and $input -eq "") {
            Write-Host "[ERROR] This field is required. Please enter a value." -ForegroundColor Red
            continue
        }
        
        return $input
    } while ($Required -and $input -eq "")
}

function Get-SQLCredentials {
    param(
        [string]$Prompt = "Do you want to use SQL Authentication instead of Windows Authentication?"
    )
    
    Write-Host "`n[QUESTION] $Prompt (Y/N)" -ForegroundColor Yellow
    $useSQLAuth = Read-Host
    
    if ($useSQLAuth -match "^[Yy]") {
        $user = Get-UserInput "Enter SQL Server username" "" $true
        $password = Get-UserInput "Enter SQL Server password" "" $true
        
        return @{
            User = $user
            Password = $password
            TrustedConnection = $false
        }
    } else {
        return @{
            User = ""
            Password = ""
            TrustedConnection = $true
        }
    }
}

function Test-SQLServerConnection {
    param(
        [string]$Server,
        [string]$Database,
        [bool]$TrustedConnection,
        [string]$User,
        [string]$Password
    )
    
    Write-Host "[INFO] Testing SQL Server connection..." -ForegroundColor Yellow
    
    try {
        if ($TrustedConnection) {
            $connectionString = "Server=$Server;Database=$Database;Trusted_Connection=true;"
        } else {
            $connectionString = "Server=$Server;Database=$Database;User Id=$User;Password=$Password;"
        }
        
        Write-Host "[INFO] Connection string: $connectionString" -ForegroundColor Gray
        
        $connection = New-Object System.Data.SqlClient.SqlConnection($connectionString)
        $connection.Open()
        
        # Test a simple query
        $testQuery = "SELECT 1 as test"
        $command = New-Object System.Data.SqlClient.SqlCommand($testQuery, $connection)
        $result = $command.ExecuteScalar()
        
        $connection.Close()
        
        Write-Host "[OK] SQL Server connection successful!" -ForegroundColor Green
        Write-Host "[INFO] Test query result: $result" -ForegroundColor Gray
        return $true
    }
    catch [System.Data.SqlClient.SqlException] {
        Write-Host "[ERROR] SQL Server connection failed with SQL Exception:" -ForegroundColor Red
        Write-Host "   Error Number: $($_.Exception.Number)" -ForegroundColor Red
        Write-Host "   Error Message: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "   Server: $($_.Exception.Server)" -ForegroundColor Red
        Write-Host "   State: $($_.Exception.State)" -ForegroundColor Red
        Write-Host "   Class: $($_.Exception.Class)" -ForegroundColor Red
        Write-Host "   Line Number: $($_.Exception.LineNumber)" -ForegroundColor Red
        
        # Provide specific guidance based on common error numbers
        switch ($_.Exception.Number) {
            18456 { Write-Host "[TIP] Authentication failed. Check your Windows account permissions." -ForegroundColor Yellow }
            4060 { Write-Host "[TIP] Database not accessible. Check database name and permissions." -ForegroundColor Yellow }
            40615 { Write-Host "[TIP] Cannot connect to server. Check server name and if SQL Server is running." -ForegroundColor Yellow }
            40631 { Write-Host "[TIP] Server not found. Check server name and network connectivity." -ForegroundColor Yellow }
            40632 { Write-Host "[TIP] Connection timeout. Check network and firewall settings." -ForegroundColor Yellow }
            40635 { Write-Host "[TIP] Connection failed. Check if SQL Server is running and accessible." -ForegroundColor Yellow }
            default { Write-Host "[TIP] Check SQL Server documentation for error $($_.Exception.Number)" -ForegroundColor Yellow }
        }
        
        return $false
    }
    catch [System.Exception] {
        Write-Host "[ERROR] General connection error: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "[INFO] Exception type: $($_.Exception.GetType().Name)" -ForegroundColor Gray
        
        if ($_.Exception.InnerException) {
            Write-Host "[INFO] Inner exception: $($_.Exception.InnerException.Message)" -ForegroundColor Gray
        }
        
        return $false
    }
}

function Test-DatabaseObjects {
    param(
        [string]$Server,
        [string]$Database,
        [string]$Table,
        [string]$Column,
        [bool]$TrustedConnection,
        [string]$User,
        [string]$Password
    )
    
    Write-Host "[INFO] Testing database objects..." -ForegroundColor Yellow
    
    try {
        if ($TrustedConnection) {
            $connectionString = "Server=$Server;Database=$Database;Trusted_Connection=true;"
        } else {
            $connectionString = "Server=$Server;Database=$Database;User Id=$User;Password=$Password;"
        }
        
        $connection = New-Object System.Data.SqlClient.SqlConnection($connectionString)
        $connection.Open()
        
        # Parse table name for schema and table
        $tableParts = $Table.Split('.')
        $schemaName = ""
        $tableName = $Table
        
        if ($tableParts.Length -eq 2) {
            $schemaName = $tableParts[0]
            $tableName = $tableParts[1]
            Write-Host "[INFO] Parsed table name: Schema='$schemaName', Table='$tableName'" -ForegroundColor Gray
        }
        
        # Test if table exists (with schema if specified)
        if ($schemaName) {
            $tableQuery = "SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = '$schemaName' AND TABLE_NAME = '$tableName'"
        } else {
            $tableQuery = "SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = '$Table'"
        }
        
        $command = New-Object System.Data.SqlClient.SqlCommand($tableQuery, $connection)
        $tableExists = $command.ExecuteScalar()
        
        if ($tableExists -eq 0) {
            Write-Host "[ERROR] Table '$Table' not found in database '$Database'" -ForegroundColor Red
            Write-Host "[TIP] Check the table name and ensure it exists in the specified database" -ForegroundColor Yellow
            
            # Show available tables with schemas
            $tablesQuery = "SELECT TABLE_SCHEMA, TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE' ORDER BY TABLE_SCHEMA, TABLE_NAME"
            $command = New-Object System.Data.SqlClient.SqlCommand($tablesQuery, $connection)
            $tables = $command.ExecuteReader()
            
            $tableList = @()
            while ($tables.Read()) {
                $schema = $tables["TABLE_SCHEMA"]
                $name = $tables["TABLE_NAME"]
                if ($schema -eq "dbo") {
                    $tableList += $name
                } else {
                    $tableList += "$schema.$name"
                }
            }
            $tables.Close()
            
            if ($tableList.Count -gt 0) {
                Write-Host "[INFO] Available tables in database '$Database':" -ForegroundColor Cyan
                $tableList | ForEach-Object { Write-Host "   - $_" -ForegroundColor White }
            }
            
            $connection.Close()
            return $false
        }
        
        Write-Host "[OK] Table '$Table' found" -ForegroundColor Green
        
        # Test if column exists (with schema if specified)
        if ($schemaName) {
            $columnQuery = "SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = '$schemaName' AND TABLE_NAME = '$tableName' AND COLUMN_NAME = '$Column'"
        } else {
            $columnQuery = "SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '$Table' AND COLUMN_NAME = '$Column'"
        }
        
        $command = New-Object System.Data.SqlClient.SqlCommand($columnQuery, $connection)
        $columnExists = $command.ExecuteScalar()
        
        if ($columnExists -eq 0) {
            Write-Host "[ERROR] Column '$Column' not found in table '$Table'" -ForegroundColor Red
            Write-Host "[TIP] Check the column name and ensure it exists in the specified table" -ForegroundColor Yellow
            
            # Show available columns (with schema if specified)
            if ($schemaName) {
                $columnsQuery = "SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = '$schemaName' AND TABLE_NAME = '$tableName' ORDER BY COLUMN_NAME"
            } else {
                $columnsQuery = "SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '$Table' ORDER BY COLUMN_NAME"
            }
            
            $command = New-Object System.Data.SqlClient.SqlCommand($columnsQuery, $connection)
            $columns = $command.ExecuteReader()
            
            $columnList = @()
            while ($columns.Read()) {
                $columnList += "$($columns['COLUMN_NAME']) ($($columns['DATA_TYPE']))"
            }
            $columns.Close()
            
            if ($columnList.Count -gt 0) {
                Write-Host "[INFO] Available columns in table '$Table':" -ForegroundColor Cyan
                $columnList | ForEach-Object { Write-Host "   - $_" -ForegroundColor White }
            }
            
            $connection.Close()
            return $false
        }
        
        Write-Host "[OK] Column '$Column' found" -ForegroundColor Green
        
        # Test if column has data (with schema if specified)
        if ($schemaName) {
            $dataQuery = "SELECT COUNT(*) FROM [$schemaName].[$tableName] WHERE [$Column] IS NOT NULL AND [$Column] != ''"
        } else {
            $dataQuery = "SELECT COUNT(*) FROM [$Table] WHERE [$Column] IS NOT NULL AND [$Column] != ''"
        }
        
        Write-Host "[INFO] Executing data count query: $dataQuery" -ForegroundColor Gray
        
        try {
            $command = New-Object System.Data.SqlClient.SqlCommand($dataQuery, $connection)
            $dataCount = $command.ExecuteScalar()
            
            Write-Host "[INFO] Column '$Column' contains $dataCount non-empty values" -ForegroundColor Cyan
            
            if ($dataCount -eq 0) {
                Write-Host "[WARNING] Column '$Column' contains no data. The generated dataset will be empty." -ForegroundColor Yellow
            }
        }
        catch [System.Data.SqlClient.SqlException] {
            Write-Host "[WARNING] Could not count data in column '$Column': $($_.Exception.Message)" -ForegroundColor Yellow
            Write-Host "[INFO] This might be due to permissions or column data type. Proceeding anyway..." -ForegroundColor Gray
            $dataCount = -1
        }
        
        $connection.Close()
        return $true
        
    }
    catch [System.Data.SqlClient.SqlException] {
        Write-Host "[ERROR] Database object test failed with SQL Exception:" -ForegroundColor Red
        Write-Host "   Error Number: $($_.Exception.Number)" -ForegroundColor Red
        Write-Host "   Error Message: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "   Server: $($_.Exception.Server)" -ForegroundColor Red
        Write-Host "   State: $($_.Exception.State)" -ForegroundColor Red
        
        return $false
    }
    catch [System.Exception] {
        Write-Host "[ERROR] General error testing database objects: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

function Build-FineTunerCommand {
    param(
        [string]$Server,
        [string]$Database,
        [string]$Table,
        [string]$Column,
        [string]$OutputFile,
        [string]$Driver,
        [int]$Port,
        [bool]$TrustedConnection,
        [string]$User,
        [string]$Password,
        [int]$MaxRows
    )
    
    $command = "python FineTuner.py --mode create-dataset --db-type sqlserver"
    
    if ($Server) { $command += " --db-server `"$Server`"" }
    if ($Database) { $command += " --db-name `"$Database`"" }
    if ($Table) { $command += " --table-name `"$Table`"" }
    if ($Column) { $command += " --column-name `"$Column`"" }
    if ($OutputFile) { $command += " --output-file `"$OutputFile`"" }
    if ($Driver) { $command += " --db-driver `"$Driver`"" }
    if ($Port -and $Port -ne 1433) { $command += " --db-port $Port" }
    if ($TrustedConnection) { $command += " --trusted-connection" }
    if ($User) { $command += " --db-user `"$User`"" }
    if ($Password) { $command += " --db-password `"$Password`"" }
    if ($MaxRows -gt 0) { $command += " --max-rows $MaxRows" }
    
    return $command
}

function Show-Summary {
    param(
        [string]$Server,
        [string]$Database,
        [string]$Table,
        [string]$Column,
        [string]$OutputFile,
        [string]$Driver,
        [int]$Port,
        [bool]$TrustedConnection,
        [string]$User,
        [string]$Password,
        [int]$MaxRows
    )
    
    Write-Host "`n[INFO] Configuration Summary:" -ForegroundColor Cyan
    Write-Host "   Server: $Server" -ForegroundColor White
    Write-Host "   Database: $Database" -ForegroundColor White
    Write-Host "   Table: $Table" -ForegroundColor White
    Write-Host "   Column: $Column" -ForegroundColor White
    Write-Host "   Output File: $OutputFile" -ForegroundColor White
    Write-Host "   Driver: $Driver" -ForegroundColor White
    Write-Host "   Port: $Port" -ForegroundColor White
    Write-Host "   Max Rows: $MaxRows" -ForegroundColor White
    
    if ($TrustedConnection) {
        Write-Host "   Authentication: Windows (Trusted Connection)" -ForegroundColor White
    } else {
        Write-Host "   Authentication: SQL Authentication" -ForegroundColor White
        Write-Host "   User: $User" -ForegroundColor White
        Write-Host "   Password: $(if ($Password) { '***' } else { 'Not Set' })" -ForegroundColor White
    }
}

function Invoke-FineTunerWithErrorHandling {
    param(
        [string]$Command
    )
    
    Write-Host "[INFO] Executing FineTuner command:" -ForegroundColor Green
    Write-Host $Command -ForegroundColor White
    
    Write-Host "`n[INFO] Starting dataset creation..." -ForegroundColor Yellow
    
    try {
        # Capture both output and errors
        $output = & cmd /c "$Command 2>&1"
        $exitCode = $LASTEXITCODE
        
        # Display the output
        if ($output) {
            Write-Host "`n[INFO] FineTuner output:" -ForegroundColor Cyan
            foreach ($line in $output) {
                if ($line -match "ERROR|Error|error") {
                    Write-Host $line -ForegroundColor Red
                } elseif ($line -match "WARNING|Warning|warning") {
                    Write-Host $line -ForegroundColor Yellow
                } elseif ($line -match "SUCCESS|Success|success|OK|Ok|ok") {
                    Write-Host $line -ForegroundColor Green
                } else {
                    Write-Host $line -ForegroundColor White
                }
            }
        }
        
        if ($exitCode -eq 0) {
            Write-Host "`n[OK] Dataset creation completed successfully!" -ForegroundColor Green
            Write-Host "[TIP] You can now use the generated file for training:" -ForegroundColor Cyan
            Write-Host "   python FineTuner.py --mode train --dataset `"$OutputFile`"" -ForegroundColor White
            return $true
        } else {
            Write-Host "`n[ERROR] Dataset creation failed with exit code: $exitCode" -ForegroundColor Red
            
            # Provide specific guidance based on common exit codes
            switch ($exitCode) {
                1 { Write-Host "[TIP] General error occurred. Check the output above for details." -ForegroundColor Yellow }
                2 { Write-Host "[TIP] Invalid arguments or missing parameters." -ForegroundColor Yellow }
                3 { Write-Host "[TIP] Database connection failed. Check your SQL Server settings." -ForegroundColor Yellow }
                4 { Write-Host "[TIP] Data extraction failed. Check table/column names and permissions." -ForegroundColor Yellow }
                5 { Write-Host "[TIP] File I/O error. Check file paths and permissions." -ForegroundColor Yellow }
                default { Write-Host "[TIP] Unknown error. Check the output above for details." -ForegroundColor Yellow }
            }
            
            return $false
        }
    }
    catch {
        Write-Host "`n[ERROR] Error executing FineTuner: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "[INFO] Exception type: $($_.Exception.GetType().Name)" -ForegroundColor Gray
        
        if ($_.Exception.InnerException) {
            Write-Host "[INFO] Inner exception: $($_.Exception.InnerException.Message)" -ForegroundColor Gray
        }
        
        return $false
    }
}

# Main script execution
if ($Help) {
    Show-Help
    exit 0
}

Write-Host "CreateDataset.ps1 - SQL Server to Training Dataset Converter" -ForegroundColor Green
Write-Host "================================================================`n" -ForegroundColor Green

# Check prerequisites
if (-not (Test-PythonInstallation)) {
    Write-Host "[ERROR] Please install Python and add it to your PATH" -ForegroundColor Red
    exit 1
}

if (-not (Test-FineTunerFiles)) {
    Write-Host "[ERROR] Please ensure FineTuner.py and CreateDataSet.py are in the current directory" -ForegroundColor Red
    exit 1
}

# Get configuration from parameters or user input
if (-not $Server) { $Server = Get-UserInput "Enter SQL Server instance name" "localhost" }
if (-not $Database) { $Database = Get-UserInput "Enter database name" }
if (-not $Table) { $Table = Get-UserInput "Enter table name" }
if (-not $Column) { $Column = Get-UserInput "Enter column name containing company names" }
if (-not $OutputFile) { $OutputFile = Get-UserInput "Enter output JSON file path" "training_data.json" }
if ($MaxRows -eq 0) { $MaxRows = Get-UserInput "Enter maximum number of rows to extract (0 for all)" 0 }

# Handle authentication if not specified
if (-not $TrustedConnection -and -not $User) {
    $authConfig = Get-SQLCredentials
    $TrustedConnection = $authConfig.TrustedConnection
    $User = $authConfig.User
    $Password = $authConfig.Password
}

# Show configuration summary
Show-Summary -Server $Server -Database $Database -Table $Table -Column $Column -OutputFile $OutputFile -Driver $Driver -Port $Port -TrustedConnection $TrustedConnection -User $User -Password $Password -MaxRows $MaxRows

# Ask for confirmation
Write-Host "`n[QUESTION] Do you want to proceed with this configuration? (Y/N)" -ForegroundColor Yellow
$confirm = Read-Host
if ($confirm -notmatch "^[Yy]") {
    Write-Host "[INFO] Operation cancelled by user" -ForegroundColor Red
    exit 0
}

# Test SQL Server connection (optional)
Write-Host "`n[QUESTION] Do you want to test the SQL Server connection first? (Y/N)" -ForegroundColor Yellow
$testConnection = Read-Host
if ($testConnection -match "^[Yy]") {
    if (-not (Test-SQLServerConnection -Server $Server -Database $Database -TrustedConnection $TrustedConnection -User $User -Password $Password)) {
        Write-Host "[ERROR] Connection test failed. Please check your configuration." -ForegroundColor Red
        exit 1
    }
}

# Test database objects
if (-not (Test-DatabaseObjects -Server $Server -Database $Database -Table $Table -Column $Column -TrustedConnection $TrustedConnection -User $User -Password $Password)) {
    Write-Host "[ERROR] Database object test failed. Please check your configuration." -ForegroundColor Red
    exit 1
}

# Build and execute the FineTuner command
$fineTunerCommand = Build-FineTunerCommand -Server $Server -Database $Database -Table $Table -Column $Column -OutputFile $OutputFile -Driver $Driver -Port $Port -TrustedConnection $TrustedConnection -User $User -Password $Password -MaxRows $MaxRows

Invoke-FineTunerWithErrorHandling -Command $fineTunerCommand

Write-Host "`n[INFO] Script completed!" -ForegroundColor Green
