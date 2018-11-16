num_runs=100;
declare -a bin_sizes=(5 10 15 20 25);

rm "result_*.dat"

for bin_size in ${bin_sizes[@]}; do
    for (( index=1; index <= $num_runs; ++index )); do
        python analisys.py $bin_size >> result_$bin_size.dat
    done
done

# ...
tmp_file="result-tmp.dat"
result_final="result-all.dat"
for result_file in "result_*.dat"; do
    paste -d "," <(grep -oP "'Accuracy:', '\K[^']*" $result_file) $result_final
    > $result_tmp
    mv $result_tmp $result_final
done
