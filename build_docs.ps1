$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$docsRoot = Join-Path $projectRoot "docs"
$englishDocs = Join-Path $docsRoot "en"
$polishDocs = Join-Path $docsRoot "pl"

$pandoc = Get-Command pandoc -ErrorAction SilentlyContinue
if (-not $pandoc) {
    Write-Error "Nie znaleziono Pandoca. Zainstaluj Pandoc i uruchom skrypt ponownie."
    exit 1
}

$requiredFiles = @(
    (Join-Path $projectRoot "README.md"),
    (Join-Path $projectRoot "README.pl.md")
)

foreach ($file in $requiredFiles) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        Write-Error "Brakuje pliku źródłowego: $file"
        exit 1
    }
}

New-Item -ItemType Directory -Force -Path $englishDocs, $polishDocs | Out-Null

Push-Location $projectRoot
try {
    & $pandoc.Source README.md -o README.rst --from=gfm --to=rst
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & $pandoc.Source README.md -o docs/en/index.html `
        --from=gfm --to=html --standalone `
        --metadata title="Means - aggregation functions for Python" `
        --metadata lang=en --css ../assets/site.css
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & $pandoc.Source README.pl.md -o docs/pl/index.html `
        --from=gfm --to=html --standalone `
        --metadata title="Means - funkcje agregacji dla Pythona" `
        --metadata lang=pl --css ../assets/site.css
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}

Write-Host "Wygenerowano: README.rst, docs/en/index.html oraz docs/pl/index.html"
