$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$docsRoot = Join-Path $projectRoot "docs"
$englishDocs = Join-Path $docsRoot "en"
$polishDocs = Join-Path $docsRoot "pl"
$englishShinyDocs = Join-Path $englishDocs "shiny"
$polishShinyDocs = Join-Path $polishDocs "shiny"

$pandoc = Get-Command pandoc -ErrorAction SilentlyContinue
if (-not $pandoc) {
    Write-Error "Nie znaleziono Pandoca. Zainstaluj Pandoc i uruchom skrypt ponownie."
    exit 1
}

$requiredFiles = @(
    (Join-Path $projectRoot "README.md"),
    (Join-Path $projectRoot "README.pl.md"),
    (Join-Path $projectRoot "apps\shiny\README.md"),
    (Join-Path $projectRoot "apps\shiny\README.pl.md")
)

foreach ($file in $requiredFiles) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        Write-Error "Brakuje pliku źródłowego: $file"
        exit 1
    }
}

New-Item -ItemType Directory -Force -Path $englishDocs, $polishDocs, $englishShinyDocs, $polishShinyDocs | Out-Null

function Write-JekyllPage {
    param(
        [string]$SourcePath,
        [string]$TargetPath,
        [string]$Title,
        [string]$Lang
    )
    $body = Get-Content -LiteralPath $SourcePath -Raw -Encoding UTF8
    $frontMatter = @"
---
layout: default
title: "$Title"
lang: $Lang
---

"@
    [System.IO.File]::WriteAllText($TargetPath, $frontMatter + $body, [System.Text.UTF8Encoding]::new($false))
}

Push-Location $projectRoot
try {
    & $pandoc.Source README.md -o README.rst --from=gfm --to=rst
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $rstBody = Get-Content -LiteralPath "README.rst" -Raw -Encoding UTF8
    $rstToc = ".. contents:: Table of contents`r`n   :depth: 4`r`n`r`n"
    [System.IO.File]::WriteAllText((Join-Path $projectRoot "README.rst"), $rstToc + $rstBody, [System.Text.UTF8Encoding]::new($false))

    Write-JekyllPage -SourcePath "README.md" -TargetPath "docs/en/index.md" `
        -Title "Means - aggregation functions for Python" -Lang "en"

    Write-JekyllPage -SourcePath "README.pl.md" -TargetPath "docs/pl/index.md" `
        -Title "Means - Funkcje agregacji dla Python" -Lang "pl"

    Write-JekyllPage -SourcePath "apps/shiny/README.md" -TargetPath "docs/en/shiny/index.md" `
        -Title "Means - Shiny App" -Lang "en"

    Write-JekyllPage -SourcePath "apps/shiny/README.pl.md" -TargetPath "docs/pl/shiny/index.md" `
        -Title "Means - Aplikacja Shiny" -Lang "pl"
}
finally {
    Pop-Location
}

Write-Host "Wygenerowano: README.rst, docs/en/index.md, docs/pl/index.md, docs/en/shiny/index.md oraz docs/pl/shiny/index.md"
