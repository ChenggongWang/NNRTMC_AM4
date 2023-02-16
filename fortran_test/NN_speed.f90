Program test
implicit none
real :: a(256,100000)
! real :: a(100000,256) 
real :: b(256,256,5)
real, allocatable, dimension(:)::c
integer:: i,j, ilayer
real :: start, finish 
a = 2.431452
b = 1.12345543

call cpu_time(start)
do i = 1,int(1e5) 
    allocate(c(256))
    c = matmul(a(:,i),b(:,:,1)) 
    ! c = matmul(a(i,:),b(:,:,1)) 
    do ilayer = 2,5
        c = matmul(c,b(:,:,ilayer))     
    end do
    deallocate(c)
end do
call cpu_time(finish)
print '("Time = ",f6.3," seconds.")',finish-start  
End Program test

