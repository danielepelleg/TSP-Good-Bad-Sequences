for i in $(seq 1 10)
    do 
        file_name="problem_$i.par"
        ./LKH $file_name 
done