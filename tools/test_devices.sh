
echo "#test for $1 times"

TEST_TIMES=$1
DEVICE_FROM=$2
DEVICE_TO=$3
DEVICE_STEP=$4
INPUT_FILE=$5

TOKEN="tmp_device_"

#create device number matlab array
echo -e "device=[\c"
for (( t=$DEVICE_FROM; t <= $DEVICE_TO; t+=$DEVICE_STEP ))
do
	echo -e "$t \c"
	> ./tmp/$TOKEN$t
done
echo "];"

#put result into separated tmp files
for i in `seq 1 $TEST_TIMES`
do	
	#total=0	
	for (( t=$DEVICE_FROM; t <= $DEVICE_TO; t+=$DEVICE_STEP ))
	do   
		RESULT=$(./../cal -algorithm pvalue -processor gpu -test 1 -device $t -thread 256 -file $INPUT_FILE -stdout processing_time)
		#echo -e "$(echo $RESULT | cut -d ' ' -f 2) \c"
		processing_time=$(echo $RESULT | cut -d ' ' -f 2)
		echo $processing_time >> ./tmp/$TOKEN$t
		#total=$(($total + $processing_time))
	done		
done

#calculate avg
re='^[0-9]+$'
echo -e "processing_time_ms=[\c"
for (( t=$DEVICE_FROM; t <= $DEVICE_TO; t+=$DEVICE_STEP ))
do
	total=0	
	#read file line by line
	while read -r line 
	do	
		processing_time=$line
		if [[ $processing_time =~ $re ]] ; then
			total=$(($total + $processing_time))
		fi
	done < ./tmp/$TOKEN$t
	#calculate avg
	echo -e "$(($total/$TEST_TIMES)) \c"
done
echo "];"
