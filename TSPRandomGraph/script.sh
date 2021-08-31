for i in $(seq 1 100)
    do 
        file_name="problem_$i.par"
        ./LKH $file_name 
done