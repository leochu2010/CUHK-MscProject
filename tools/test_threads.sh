# export into 3 files, x: samples, y: features, z: processing time
# combine into 1 at last


echo "#test for $1 times"

CALCULATE=$1
TEST_TIMES=$2
MIN_THREAD=$3
MAX_THREAD=$4
STEP=$5

#generate matlab array
MATLAB_THREAD_ARRAY="threads=["

for (( d=$MIN_THREAD; d <= $MAX_THREAD; d+=$STEP))
do	
	MATLAB_THREAD_ARRAY="$MATLAB_THREAD_ARRAY $d"
done

MATLAB_THREAD_ARRAY=$MATLAB_THREAD_ARRAY"];"

TOKEN="tmp_Thread_$(date +%s)"

#create thread number matlab array


for (( d=$MIN_THREAD; d <= $MAX_THREAD; d+=$STEP))
do	
	> ./tmp/$TOKEN-$d
done


#put result into separated tmp files
for i in `seq 1 $TEST_TIMES`
do	
	for (( d=$MIN_THREAD; d <= $MAX_THREAD; d+=$STEP))
	do		
		COMMAND="$CALCULATE --stdout --display_processing_time --thread $d --test 1"		
		#echo $COMMAND
		RESULT=$($COMMAND)
		processing_time=$(echo $RESULT | cut -d ' ' -f 2)
		echo "test $i, $d thread processing time: $processing_time ms"
		echo $processing_time >> ./tmp/$TOKEN-$d
	done
done

#calculate avg
re='^[0-9]+$'
MATLAB_PROCESSING_TIME_ARRAY="processing_time_ms=["
for (( d=$MIN_THREAD; d <= $MAX_THREAD; d+=$STEP))
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
echo "%Number of test for each thread number: $TEST_TIMES">> ./result/$TOKEN
echo "%Min Thread $MIN_THREAD -> Max Thread MAX_THREAD" >> ./result/$TOKEN
echo "%=========================================================================================" >> ./result/$TOKEN
echo "$MATLAB_THREAD_ARRAY">> ./result/$TOKEN
echo "$MATLAB_PROCESSING_TIME_ARRAY">> ./result/$TOKEN
echo "plot(threads, processing_time_ms)">> ./result/$TOKEN

echo "$MATLAB_THREAD_ARRAY"
echo "$MATLAB_PROCESSING_TIME_ARRAY"
echo "plot(threads, processing_time_ms)"
