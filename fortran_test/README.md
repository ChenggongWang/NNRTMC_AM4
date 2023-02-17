# benchmark the performance of matrix multiplication with interl compiler

# ifort: ?gemm come with mkl improves the speed

run NN_speed_sgemv.F90 on stellar (intel cpu).
-mkl=sequential forced the code to run in non-threaded mode (or is will use all cores which will slow down the speed).
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

-O3 is slower and -lblas is not realy working
```
$ gfortran -O0 NN_speed_sgemv.F90 -lblas; ./a.out
total run times:  10000
matmul avg time per run =  0.060 ms
loop  avg time per run =  0.640 ms
SGEMM  avg time per run =  0.775 ms
SGEMV  avg time per run =  0.328 ms
$ gfortran -O3 NN_speed_sgemv.F90 -lblas; ./a.out
total run times:  10000
matmul avg time per run =  0.324 ms
loop  avg time per run =  0.319 ms
SGEMM  avg time per run =  0.799 ms
SGEMV  avg time per run =  0.323 ms
```
-finline-matmul-limit=0 helps a lot!
```
$ gfortran -O3 -finline-matmul-limit=0 NN_speed_sgemv.F90 -lblas; ./a.out
total run times:  10000
matmul avg time per run =  0.045 ms
loop  avg time per run =  0.326 ms
SGEMM  avg time per run =  0.800 ms
SGEMV  avg time per run =  0.337 ms
```
