#rm -rf cropped
#mkdir cropped
FILES=`ls  ./cropped`
for f in $FILES
do
  filename=$(basename -- "$f")
  extension="${filename##*.}"
  filename="${filename%.*}"
  #echo $extension

  if [ $extension == 'pdf' ]; then
    echo "converting $f file..."
    convert -density 300 ./cropped/$filename.pdf ./cropped/$filename.jpg
  fi
done
#mv *-crop.pdf cropped