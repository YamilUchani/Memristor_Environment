# Script de compilación LaTeX con tectonic
Write-Host "Compilando tesis..." -ForegroundColor Cyan
$tectonic = "..\..\scratch\bin\tectonic.exe"
if (-not (Test-Path $tectonic)) {
    Write-Host "No se encuentra tectonic en $tectonic" -ForegroundColor Red
    exit 1
}
& $tectonic -X compile main.tex --keep-logs --keep-intermediates
if ($LASTEXITCODE -ne 0) {
    Write-Host "Compilación falló" -ForegroundColor Red
    exit 1
}
Write-Host "Compilación exitosa: main.pdf" -ForegroundColor Green
Get-Item main.pdf | Format-List Name, Length, LastWriteTime
