# ...
result_tmp="result-tmp.dat"
result_final="result-all.dat"

rm -f $result_final
touch $result_final

declare -a result_files=()

bin_sizes=(5 10 15 20 25)
for bin_size in ${bin_sizes[@]}; do

    result_file="result_${bin_size}.dat";
    result_files+=("result_${bin_size}.dat.tmp")
    grep -oP "'Accuracy:', '\K[^']*" $result_file > "$result_file.tmp"

done

echo ${bin_sizes[*]} | tr ' ' ',' > $result_final
paste -d "," ${result_files[@]} >> $result_final
