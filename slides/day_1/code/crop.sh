rm -rf cropped
mkdir cropped
FILES=`ls ./tmp -I "*-crop.pdf" `
for f in $FILES
do
  filename=$(basename -- "$f")
  extension="${filename##*.}"
  filename="${filename%.*}"
  #echo $extension

  if [ $extension == 'pdf' ]; then
    echo "Processing $f file..."
    pdfcrop ./tmp/$f
  fi
done
mv ./tmp/*-crop.pdf cropped