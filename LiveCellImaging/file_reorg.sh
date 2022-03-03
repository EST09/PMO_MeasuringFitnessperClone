cd /Users/4470246/Projects/PMO/MeasuringFitnessPerClone/data/LiveCellImaging/A01_190513_rawData/SNU-668/NA190509_Plate_13295_sub/byframe
# cd /mnt/ix1/Projects/M027_160330_PMO_cisplatin/data/GastricCancerCLs/LiveCellImaging/SNU-668/NA190509_Plate_13295

##By Frame
tpts=`ls -d TimePoint_*`
for tpt in $tpts; do
  echo $tpt
  # tpt=="TimePoint_1"
  f=`ls $tpt/*Thumb*`
  for x in $f; do
    y=`basename $x`; 
    z=`echo $y | sed 's/_Thumb.TIF//g' | sed 's/NA190509_//g'`; 
    # mkdir byframe/$z; ##Only for 1st timepoint
    mv $x byframe/$z/$tpt"_Thumb.TIF"; 
  done
done

##rename pics such that they are opened in the corrcet order
declare -a arr=("1" "2" "3" "4" "5" "6" "7" "8" "9")
f=`ls`
for x in $f; do
	for i in "${arr[@]}"; do 
		 # echo TimePoint_"$i"_full.TIF; 
  		mv $x/TimePoint_"$i"_full.TIF $x/TimePoint_0"$i"_full.TIF; 
	done
done


