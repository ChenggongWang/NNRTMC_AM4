# some test on speeed

# gfortran: avoid inlining of matmul is important
source: https://stackoverflow.com/questions/66682180/why-is-matmul-slower-with-gfortran-compiler-optimization-turned-on

-O3 is slower
```
$ gfortran -ffree-form -O0 NN_speed2.f90 -o NN_speed2.out; ./NN_speed2.out
Time =  2.624 seconds.
$ gfortran -ffree-form -O3 NN_speed2.f90 -o NN_speed2.out; ./NN_speed2.out
Time = 31.695 seconds.
```
-finline-matmul-limit=0
```
$ gfortran -ffree-form -O3 -finline-matmul-limit=0 NN_speed2.f90 -o NN_speed2.out; ./NN_speed2.out
Time =  1.996 seconds.
```
# ifort: sgemm come with mkl improves the speed
```
$ ifort -mkl=sequential -O3 NN_speed3_sgemm.F90 ; ./a.out  
total run times:  10000
matmul avg time per run =  0.246 ms
SGEMM  avg time per run =  0.040 ms
```

manual loop is also better
```
$ ifort -O3 NN_speed4_loop.F90 ; ./a.out          
total run times:  10000
matmul avg time per run =  0.261 ms
loop   avg time per run =  0.128 ms
```
