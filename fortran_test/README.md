# benchmark the performance of matrix multiplication with interl compiler

# ifort: ?gemm come with mkl improves the speed

run NN_speed_sgemv.F90 on stellar (intel cpu).
-mkl=sequential forced the code to run in non-threaded mode.
it is faster in this case (on a muticore system).

``` 
$ ifort -mkl=sequential -O3 NN_speed_sgemv.F90 ; ./a.out
total run times:  10000
matmul avg time per run =  0.251 ms
loop  avg time per run =  0.168 ms
SGEMM  avg time per run =  0.041 ms
SGEMV  avg time per run =  0.041 ms
```

on stellar (amd cpu)
``` 
$ ifort -mkl=sequential -O3 NN_speed_sgemv.F90 ; ./a.out
total run times:  10000
matmul avg time per run =  0.634 ms
loop  avg time per run =  0.127 ms
SGEMM  avg time per run =  0.039 ms
SGEMV  avg time per run =  0.038 ms
```


# gfortran: avoid inlining of matmul is important
source: https://stackoverflow.com/questions/66682180/why-is-matmul-slower-with-gfortran-compiler-optimization-turned-on

-O3 is slower?
```
$ gfortran -O0 NN_speed2.f90 -o NN_speed2.out; ./NN_speed2.out
total run times:  10000
matmul avg time per run =  0.047 ms
$ gfortran -O3 NN_speed2.f90 -o NN_speed2.out; ./NN_speed2.out
total run times:  10000
matmul avg time per run =  0.270 ms
```
-finline-matmul-limit=0
```
$ gfortran -O3 -finline-matmul-limit=0 NN_speed2.f90 -o NN_speed2.out; ./NN_speed2.out
total run times:  10000
matmul avg time per run =  0.040 ms
```
