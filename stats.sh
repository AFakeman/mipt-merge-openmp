(echo size threads merge qsort
for size in 2000 20000 200000 2000000; do
    for threads in 1 2 4 8 16; do
        for i in `seq 1 10`; do
            ./main $size $(($size / $threads)) $threads | 
            awk '/.*: ([0-9\.]*)/{print $2}' | 
            paste - -;
        done | echo  $size $threads \
                     $(awk '{sum_merge+=$1; sum_qsort+=$2;} \
                           END {print sum_merge/10,sum_qsort/10}') \
        
    done
done) | python3 render.py
