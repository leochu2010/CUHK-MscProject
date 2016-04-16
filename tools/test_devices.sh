# export into 3 files, x: samples, y: features, z: processing time
# combine into 1 at last


echo "#test for $1 times"

CALCULATE=$1
TEST_TIMES=$2
MIN_DEVICE=$3
MAX_DEVICE=$4

#generate matlab array
MATLAB_DEVICE_ARRAY="devices=["

for (( d=$MIN_DEVICE; d <= $MAX_DEVICE; d+=1))
do	
	MATLAB_DEVICE_ARRAY="$MATLAB_DEVICE_ARRAY $d"
done

MATLAB_DEVICE_ARRAY=$MATLAB_DEVICE_ARRAY"];"

TOKEN="tmp_Device_$(date +%s)"

#create device number matlab array


for (( d=$MIN_DEVICE; d <= $MAX_DEVICE; d+=1))
do	
	> ./tmp/$TOKEN-$d
done


#put result into separated tmp files
for i in `seq 1 $TEST_TIMES`
do	
	for (( d=$MIN_DEVICE; d <= $MAX_DEVICE; d+=1))
	do		
		COMMAND="$CALCULATE --stdout --display_processing_time --device $d --test 1"		
		#echo $COMMAND
		RESULT=$($COMMAND)
		processing_time=$(echo $RESULT | cut -d ' ' -f 2)
		echo "test $i, $d device processing time: $processing_time ms"
		echo $processing_time >> ./tmp/$TOKEN-$d
	done
done

#calculate avg
re='^[0-9]+$'
MATLAB_PROCESSING_TIME_ARRAY="processing_time_ms=["
for (( d=$MIN_DEVICE; d <= $MAX_DEVICE; d+=1))
do	
	total=0	
	#read file line by line
	while read -r line 
	do	
		processing_time=$line
		if [[ $processing_time =~ $re ]] ; then
			total=$(($total + $processing_time))
		fi
	done < ./tmp/$TOKEN-$d
	#calculate avg	
	MATLAB_PROCESSING_TIME_ARRAY="$MATLAB_PROCESSING_TIME_ARRAY $(($total/$TEST_TIMES))"		
done
MATLAB_PROCESSING_TIME_ARRAY="$MATLAB_PROCESSING_TIME_ARRAY ];"

echo "%=========================================================================================" >> ./result/$TOKEN
echo "%Command: $CALCULATE">> ./result/$TOKEN
echo "%Number of test for each device number: $TEST_TIMES">> ./result/$TOKEN
echo "%Min Device $MIN_DEVICE -> Max Device MAX_DEVICE" >> ./result/$TOKEN
echo "%=========================================================================================" >> ./result/$TOKEN
echo "$MATLAB_DEVICE_ARRAY">> ./result/$TOKEN
echo "$MATLAB_PROCESSING_TIME_ARRAY">> ./result/$TOKEN
echo "plot(devices, processing_time_ms)">> ./result/$TOKEN

echo "$MATLAB_DEVICE_ARRAY"
echo "$MATLAB_PROCESSING_TIME_ARRAY"
echo "plot(devices, processing_time_ms)"
