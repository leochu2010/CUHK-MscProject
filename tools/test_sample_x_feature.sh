# export into 3 files, x: samples, y: features, z: processing time
# combine into 1 at last


echo "#test for $1 times"

CALCULATE=$1
TEST_TIMES=$2
MAX_S_MUL=$3
MAX_F_MUL=$4
MIN_S_F=$5

#generate sample and feature permutation
SAMPLES=$MIN_S_F
FEATURES=$MIN_S_F
SAMPLE_ARRAY[0]=$SAMPLES
FEATURE_ARRAY[0]=$FEATURES
for (( f=1; f <= $(($MAX_F_MUL/2)); f+=1))
do
	FEATURES=$(($MIN_S_F * $f * 2))	
	SAMPLE_ARRAY[${#SAMPLE_ARRAY[@]}]=$SAMPLES
	FEATURE_ARRAY[${#FEATURE_ARRAY[@]}]=$FEATURES
done

for (( s=1; s <= $(($MAX_S_MUL/2)); s+=1))
do

	SAMPLES=$(($MIN_S_F * $s * 2))
	FEATURES=$MIN_S_F
	SAMPLE_ARRAY[${#SAMPLE_ARRAY[@]}]=$SAMPLES
	FEATURE_ARRAY[${#FEATURE_ARRAY[@]}]=$FEATURES		
	for (( f=1; f <= $(($MAX_F_MUL/2)); f+=1))
	do		
		FEATURES=$(($MIN_S_F * $f * 2))
		
		SAMPLE_ARRAY[${#SAMPLE_ARRAY[@]}]=$SAMPLES
		FEATURE_ARRAY[${#FEATURE_ARRAY[@]}]=$FEATURES		
	done
done

#generate matlab array
MATLAB_SAMPLE_ARRAY="samples=[$MIN_S_F"
MATLAB_FEATURE_ARRAY="features=[$MIN_S_F"

for (( s=1; s <= $(($MAX_S_MUL/2)); s+=1))
do
	SAMPLES=$(($MIN_S_F * $s * 2))
	MATLAB_SAMPLE_ARRAY="$MATLAB_SAMPLE_ARRAY $SAMPLES"
done

for (( f=1; f <= $(($MAX_F_MUL/2)); f+=1))
do
	FEATURES=$(($MIN_S_F * $f * 2))		
	MATLAB_FEATURE_ARRAY="$MATLAB_FEATURE_ARRAY $FEATURES"
done

MATLAB_SAMPLE_ARRAY=$MATLAB_SAMPLE_ARRAY"];"
MATLAB_FEATURE_ARRAY=$MATLAB_FEATURE_ARRAY"];"

TOKEN="tmp_S_F_$(date +%s)"
ARRAY_LENGTH=${#SAMPLE_ARRAY[@]}

#create sample number matlab array
#create feature number matlab array


for (( p=0; p < $ARRAY_LENGTH; p+=1))
do	
	> ./tmp/$TOKEN-${SAMPLE_ARRAY[$p]}-${FEATURE_ARRAY[$p]}
done





#put result into separated tmp files
for i in `seq 1 $TEST_TIMES`
do	
	for (( p=0; p < $ARRAY_LENGTH; p+=1))
	do
		SAMPLES=${SAMPLE_ARRAY[$p]}
		FEATURES=${FEATURE_ARRAY[$p]}
		SAMPLES_MUL=$(($SAMPLES/$MIN_S_F))
		FEATURES_MUL=$(($FEATURES/$MIN_S_F))
		COMMAND="$CALCULATE --stdout --display_processing_time --multiply_samples $SAMPLES_MUL --multiply_features $FEATURES_MUL --test 1"		
		#echo $COMMAND
		RESULT=$($COMMAND)		
		processing_time=$(echo $RESULT | cut -d ' ' -f 2)
		echo "test $i, $SAMPLES x $FEATURES processing time: $processing_time ms"
		echo $processing_time >> ./tmp/$TOKEN-$SAMPLES-$FEATURES
	done
done

#calculate avg
re='^[0-9]+$'
MATLAB_PROCESSING_TIME_ARRAY="processing_time_ms=["
PREVIOUS_SAMPLE_SIZE=$MIN_S_F
for (( p=0; p < $ARRAY_LENGTH; p+=1))
do
	SAMPLES=${SAMPLE_ARRAY[$p]}
	FEATURES=${FEATURE_ARRAY[$p]}
	total=0	
	#read file line by line
	while read -r line 
	do	
		processing_time=$line
		if [[ $processing_time =~ $re ]] ; then
			total=$(($total + $processing_time))
		fi
	done < ./tmp/$TOKEN-$SAMPLES-$FEATURES
	#calculate avg
	if [[ $PREVIOUS_SAMPLE_SIZE != $SAMPLES ]] ; then
		MATLAB_PROCESSING_TIME_ARRAY="$MATLAB_PROCESSING_TIME_ARRAY;"	
	fi
	MATLAB_PROCESSING_TIME_ARRAY="$MATLAB_PROCESSING_TIME_ARRAY $(($total/$TEST_TIMES))"	
	PREVIOUS_SAMPLE_SIZE=$SAMPLES
done
MATLAB_PROCESSING_TIME_ARRAY="$MATLAB_PROCESSING_TIME_ARRAY ];"

echo "%=========================================================================================" >> ./result/$TOKEN
echo "%Command: $CALCULATE">> ./result/$TOKEN
echo "%Number of test for each permutation: $TEST_TIMES">> ./result/$TOKEN
echo "%Permutation: [$MIN_S_F - $SAMPLES samples] X [$MIN_S_F - $FEATURES features] " >> ./result/$TOKEN
echo "%=========================================================================================" >> ./result/$TOKEN
echo "$MATLAB_SAMPLE_ARRAY">> ./result/$TOKEN
echo "$MATLAB_FEATURE_ARRAY">> ./result/$TOKEN
echo "$MATLAB_PROCESSING_TIME_ARRAY">> ./result/$TOKEN
echo "surf(samples, features, processing_time_ms)">> ./result/$TOKEN

echo "$MATLAB_SAMPLE_ARRAY"
echo "$MATLAB_FEATURE_ARRAY"
echo "$MATLAB_PROCESSING_TIME_ARRAY"
echo "surf(samples, features, processing_time_ms);"
