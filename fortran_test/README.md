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
