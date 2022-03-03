##To be used on Mac
cd /Users/4470246/Projects/Tools/ImageMagick-7.0.8
export MAGICK_HOME="/Users/4470246/Projects/Tools/ImageMagick-7.0.8"
export PATH="$MAGICK_HOME/bin:$PATH"
export DYLD_LIBRARY_PATH="$MAGICK_HOME/lib/"

cd /Users/4470246/Projects/PMO/MeasuringFitnessPerClone/data/LiveCellImaging/
  
ind="A01_190513_rawData/SNU-668/NA190509_Plate_13295/bywell"
oud_02="A02_190517_StichByWell"
oud_03="A03_190516_PixelClassification_ilastik"

## identify  $oud_02/A01/A01_s17_TimePoint_1_full.TIF ##check the size of an image -- is it indeed 1392x1040?

declare -a arr=( '01' '02' '03' '04' '05' '06' '07' '08' '09' '10' '11' '12')
wells=`ls $ind`
for w in $wells; do
  mkdir $oud_03/$w; ##For later
  mkdir $oud_02/$w; ##For now 
  ##loop through timepoints
  for i in "${arr[@]}"; do
    montage $ind/$w/$w"_s"*"_TimePoint_"$i"_"* -tile 3x3 -geometry 1392x1040+0+0 $oud_02/$w/$w"_TimePoint_"$i".TIF"
  done
done

