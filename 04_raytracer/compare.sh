cd v1 
make
#sleep 1s
#echo "----------------- v1 1 -----------------"
time ./main
/opt/cuda/bin/nvprof ./main
#sleep 1s
#echo "----------------- v1 2 -----------------"
#time ./main
#nvprof ./main
#sleep 1s

cd ..

cd v2 
make
#sleep 1s
#echo "----------------- v2 1 -----------------"
time ./main
/opt/cuda/bin/nvprof ./main
#sleep 1s
#echo "----------------- v2 2 -----------------"
#time ./main
#nvprof ./main
#sleep 1s

cd ..
