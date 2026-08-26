$inputFile = "mmpose.zip"
$chunkSize = 95MB  # 1ファイルあたりの上限

$stream = [System.IO.File]::OpenRead($inputFile)
$buffer = New-Object byte[] $chunkSize
$partNumber = 1

try {
    while (($bytesRead = $stream.Read($buffer, 0, $chunkSize)) -gt 0) {
        $outputFile = "mmpose.zip.part_$( "{0:D2}" -f $partNumber )"
        $outputStream = [System.IO.File]::Create($outputFile)
        $outputStream.Write($buffer, 0, $bytesRead)
        $outputStream.Close()
        Write-Host "作成完了: $outputFile"
        $partNumber++
    }
} finally {
    $stream.Close()
}
