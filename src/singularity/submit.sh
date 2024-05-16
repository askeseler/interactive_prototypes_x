printf -v date '%(%Y_%m_%d_%H_%M_%S)T\n' -1 
echo $date
sbatch -p gpu1 --output=$date "run.sh" "${@}"
