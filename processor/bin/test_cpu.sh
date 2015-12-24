
echo "#test for $1 times"

TEST_TIMES=$1
CPU_FROM=$2
CPU_TO=$3
CPU_STEP=$4
INPUT_FILE=$5

TOKEN="tmp_CPU_"

#create device number matlab array
echo -e "cpu=[\c"
for (( t=$CPU_FROM; t <= $CPU_TO; t+=$CPU_STEP ))
do
	echo -e "$t \c"
	> ./tmp/$TOKEN$t
done
echo "];"

#put result into separated tmp files
for i in `seq 1 $TEST_TIMES`
do	
	#total=0	
	for (( t=$CPU_FROM; t <= $CPU_TO; t+=$CPU_STEP ))
	do   
		RESULT=$(./../cal -algorithm pvalue -processor cpu -test 1 -file $INPUT_FILE -stdout processing_time)
		#echo -e "$(echo $RESULT | cut -d ' ' -f 2) \c"
		processing_time=$(echo $RESULT | cut -d ' ' -f 2)
		echo $processing_time >> ./tmp/$TOKEN$t
		#total=$(($total + $processing_time))
	done		
done

#calculate avg
re='^[0-9]+$'
echo -e "processing_time_ms=[\c"
for (( t=$CPU_FROM; t <= $CPU_TO; t+=$CPU_STEP ))
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
