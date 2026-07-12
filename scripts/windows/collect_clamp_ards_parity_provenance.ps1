[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("run_1", "run_2")]
    [string]$RunLabel,

    [Parameter(Mandatory = $true)]
    [string]$ClampVersion,

    [Parameter(Mandatory = $true)]
    [string]$ClampBuild,

    [Parameter(Mandatory = $true)]
    [string]$ProjectCommit,

    [Parameter(Mandatory = $true)]
    [string]$ProjectDir,

    [Parameter(Mandatory = $true)]
    [string]$OutputDir,

    [Parameter(Mandatory = $true)]
    [string]$ProvenanceOutput,

    [Parameter(Mandatory = $true)]
    [string]$ExportSettings,

    [Parameter(Mandatory = $true)]
    [string]$OffsetConvention,

    [Parameter(Mandatory = $true)]
    [string]$NullConvention,

    [Parameter(Mandatory = $true)]
    [string]$ManualCommand,

    [Parameter(Mandatory = $true)]
    [string]$StartedAtUtc,

    [Parameter(Mandatory = $true)]
    [string]$FinishedAtUtc
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Get-RelativePath {
    param(
        [Parameter(Mandatory = $true)][string]$BasePath,
        [Parameter(Mandatory = $true)][string]$TargetPath
    )
    $base = [System.IO.Path]::GetFullPath($BasePath).TrimEnd("\") + "\"
    $target = [System.IO.Path]::GetFullPath($TargetPath)
    $baseUri = New-Object System.Uri($base)
    $targetUri = New-Object System.Uri($target)
    return [System.Uri]::UnescapeDataString($baseUri.MakeRelativeUri($targetUri).ToString())
}

function Get-HashedFileRecord {
    param(
        [Parameter(Mandatory = $true)][System.IO.FileInfo]$File,
        [Parameter(Mandatory = $true)][string]$Root
    )
    return [ordered]@{
        relative_path = (Get-RelativePath -BasePath $Root -TargetPath $File.FullName)
        bytes = $File.Length
        sha256 = (Get-FileHash -LiteralPath $File.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

$resolvedProject = (Resolve-Path -LiteralPath $ProjectDir).Path
$resolvedOutput = (Resolve-Path -LiteralPath $OutputDir).Path
$outputFiles = @(
    Get-ChildItem -LiteralPath $resolvedOutput -Recurse -File |
        Where-Object { $_.Extension.ToLowerInvariant() -in @(".txt", ".xmi") } |
        Sort-Object FullName
)
if ($outputFiles.Count -eq 0) {
    throw "No TXT/XMI files were found in the returned CLAMP output directory."
}

$projectFiles = @(
    Get-ChildItem -LiteralPath $resolvedProject -Recurse -File |
        Where-Object {
            $relative = Get-RelativePath -BasePath $resolvedProject -TargetPath $_.FullName
            $normalized = $relative.Replace("\", "/").ToLowerInvariant()
            -not ($normalized.StartsWith("data/input/") -or $normalized.StartsWith("data/output/"))
        } |
        Sort-Object FullName
)
$resourceFiles = @(
    Get-ChildItem -LiteralPath (Join-Path $resolvedProject "Components") -Recurse -File |
        Sort-Object FullName
)

$operatingSystem = Get-CimInstance Win32_OperatingSystem
$javaOutput = @(& java -version 2>&1 | ForEach-Object { $_.ToString() })
if ($LASTEXITCODE -ne 0 -or $javaOutput.Count -eq 0) {
    throw "The Java runtime version could not be collected."
}
$projectHashes = [ordered]@{}
foreach ($file in $projectFiles) {
    $relative = Get-RelativePath -BasePath $resolvedProject -TargetPath $file.FullName
    $projectHashes[$relative] = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
}
$resourceHashes = [ordered]@{}
foreach ($file in $resourceFiles) {
    $relative = Get-RelativePath -BasePath $resolvedProject -TargetPath $file.FullName
    $resourceHashes[$relative] = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
}
$outputRecords = @(
    foreach ($file in $outputFiles) {
        Get-HashedFileRecord -File $file -Root $resolvedOutput
    }
)

$payload = [ordered]@{
    schema_version = 1
    run_label = $RunLabel
    recorded_at_utc = [DateTime]::UtcNow.ToString("o")
    started_at_utc = $StartedAtUtc
    finished_at_utc = $FinishedAtUtc
    clamp = [ordered]@{
        version = $ClampVersion
        build = $ClampBuild
    }
    windows = [ordered]@{
        caption = $operatingSystem.Caption
        version = $operatingSystem.Version
        build_number = $operatingSystem.BuildNumber
        architecture = $operatingSystem.OSArchitecture
        locale = [System.Globalization.CultureInfo]::CurrentCulture.Name
        timezone = [System.TimeZoneInfo]::Local.Id
        timezone_utc_offset = [System.TimeZoneInfo]::Local.GetUtcOffset([DateTime]::Now).ToString()
    }
    java = [ordered]@{
        version_output = ($javaOutput -join "`n")
    }
    project = [ordered]@{
        commit = $ProjectCommit
        files_sha256 = $projectHashes
    }
    resources_sha256 = $resourceHashes
    export_settings = $ExportSettings
    offset_convention = $OffsetConvention
    null_convention = $NullConvention
    manual_commands = @($ManualCommand)
    output_files = $outputRecords
}

$provenancePath = [System.IO.Path]::GetFullPath($ProvenanceOutput)
$provenanceParent = Split-Path -Parent $provenancePath
if (-not (Test-Path -LiteralPath $provenanceParent)) {
    New-Item -ItemType Directory -Path $provenanceParent | Out-Null
}
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText(
    $provenancePath,
    (($payload | ConvertTo-Json -Depth 20) + "`n"),
    $utf8NoBom
)

$manifestPath = [System.IO.Path]::ChangeExtension($provenancePath, "SHA256SUMS")
$manifestLines = @(
    foreach ($record in $outputRecords) {
        "{0}  {1}" -f $record.sha256, $record.relative_path
    }
)
[System.IO.File]::WriteAllText(
    $manifestPath,
    (($manifestLines -join "`n") + "`n"),
    $utf8NoBom
)

Write-Host "Recorded $($outputRecords.Count) output files in $provenancePath"
Write-Host "Wrote the returned-run checksum manifest to $manifestPath"
