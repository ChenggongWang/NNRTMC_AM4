module test_mod
implicit none
public NN_Linear_layer_type
type :: NN_Linear_layer_type 
    real, dimension(:,:), pointer :: weight=>NULL()
    real, dimension(:),   pointer :: bias=>NULL()
end type NN_Linear_layer_type
public NN_FC_type
type :: NN_FC_type
    integer :: num_hid_nodes
    integer :: num_layers
    type(NN_Linear_layer_type), dimension(:), pointer:: Layers
end type NN_FC_type

type(NN_FC_type)   :: Rad_NN_FC

contains 
  
subroutine nn_init() 
    integer::ilayer
    Rad_NN_FC%num_layers = 5
    Rad_NN_FC%num_hid_nodes = 256
    allocate(Rad_NN_FC%Layers(Rad_NN_FC%num_layers))
    do ilayer = 1, Rad_NN_FC%num_layers
        allocate(Rad_NN_FC%Layers(ilayer)%weight(256,256)) 
        allocate(Rad_NN_FC%Layers(ilayer)%bias(256))  
    end do
end subroutine nn_init

subroutine nn_pred_1d(x,y) 
    real, dimension(:), intent(in) :: x
    real, dimension(:), intent(inout) :: y
    integer::ilayer 
     
    do ilayer = 1, Rad_NN_FC%num_layers
        y = matmul(x,Rad_NN_FC%Layers(ilayer)%weight)+ Rad_NN_FC%Layers(ilayer)%bias
        y = NN_activ(y)
    end do
end subroutine nn_pred_1d

real elemental function NN_activ(x)
    real, intent(in) :: x
    ! ReLU:
    if (x>0) then
        NN_activ = x
    else
        NN_activ = 0
    end if
    ! tanh
    ! NN_activ = tanh(x)
end function NN_activ
 

end module test_mod
Program test

use test_mod
implicit none
real :: a(256,10000)  
real, allocatable, dimension(:)::c
integer:: i,j, ilayer
real :: start, finish 
a = 2.431452 

call nn_init()
! allocate(c(size(a,1)))
! call nn_pred_1d(a(:,1),c)
call cpu_time(start)
do j = 1, int(1e4) 
    allocate(c(size(a,1)))
    call nn_pred_1d(a(:,j),c)
    deallocate(c) 
end do
call cpu_time(finish)
print '("Time = ",f6.3," seconds.")',finish-start   
End Program test

