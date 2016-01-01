
echo "#test for $1 times"

TEST_TIMES=$1
THREAD_FROM=$2
THREAD_TO=$3
THREAD_STEP=$4
INPUT_FILE=$5

TOKEN="tmp_thread"

#create thread number matlab array
echo -e "threads=[\c"
for (( t=$THREAD_FROM; t <= $THREAD_TO; t+=$THREAD_STEP ))
do
	echo -e "$t \c"
	> ./tmp/$TOKEN$t
done
echo "];"

#put result into separated tmp files
for i in `seq 1 $TEST_TIMES`
do	
	#total=0	
	for (( t=$THREAD_FROM; t <= $THREAD_TO; t+=$THREAD_STEP ))
	do   
		RESULT=$(./../cal -algorithm pvalue -processor gpu -test 1 -device 1 -thread $t -file $INPUT_FILE -stdout processing_time)
		#echo -e "$(echo $RESULT | cut -d ' ' -f 2) \c"
		processing_time=$(echo $RESULT | cut -d ' ' -f 2)
		echo $processing_time >> ./tmp/$TOKEN$t
		#total=$(($total + $processing_time))
	done		
done

#calculate avg
re='^[0-9]+$'
echo -e "processing_time_ms=[\c"
for (( t=$THREAD_FROM; t <= $THREAD_TO; t+=$THREAD_STEP ))
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
