$ErrorActionPreference = "Stop"
$thesisDirectory = Join-Path $PSScriptRoot "PLANTILLA_PDF_FINAL"
$mainSource = Join-Path $thesisDirectory "main.tex"
$tectonic = Get-Command "tectonic" -CommandType Application -ErrorAction SilentlyContinue

if (-not (Test-Path -LiteralPath $mainSource -PathType Leaf)) {
    Write-Error "No se encuentra la fuente oficial: $mainSource"
    exit 1
}
if (-not $tectonic) {
    Write-Error "No se encuentra tectonic en PATH. Instala Tectonic y vuelve a ejecutar este script."
    exit 1
}

Push-Location $thesisDirectory
try {
    Write-Host "Compilando la tesis oficial desde $thesisDirectory..." -ForegroundColor Cyan
    & $tectonic.Source -X compile "main.tex" --keep-logs --keep-intermediates
    if ($LASTEXITCODE -ne 0) {
        throw "Tectonic terminó con código $LASTEXITCODE."
    }

    $pdf = Join-Path $thesisDirectory "main.pdf"
    if (-not (Test-Path -LiteralPath $pdf -PathType Leaf)) {
        throw "La compilación terminó sin generar $pdf."
    }
    Write-Host "Compilación exitosa: $pdf" -ForegroundColor Green
    Get-Item -LiteralPath $pdf | Format-List Name, Length, LastWriteTime
}
catch {
    Write-Error "No se pudo compilar la tesis oficial. $_"
    exit 1
}
finally {
    Pop-Location
}
