for VAR in ./*.ini
do
	cat $VAR > ${VAR}.out
	echo "======" >> ${VAR}.out
	nohup python main.py $VAR >> ${VAR}.out &
done
