while IFS= read -r line; do
    unzip $line.SRTMGL1.hgt.zip
    mv $line.SRTMGL1.hgt $line.hgt
    zip $line.SRTMGL1.repacked.hgt.zip $line.hgt
done < "files.txt"