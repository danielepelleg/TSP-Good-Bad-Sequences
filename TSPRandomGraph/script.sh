for i in $(seq 101 200)
    do 
        file_name="problem_$i.par"
        ./LKH $file_name 
done