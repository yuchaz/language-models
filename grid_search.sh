for VAR in ./*.ini
do
	cat $VAR > ./output/${VAR}.out
	echo "======" >> ./output/${VAR}.out
	nohup python main.py $VAR >> ./output/${VAR}.out &
done
