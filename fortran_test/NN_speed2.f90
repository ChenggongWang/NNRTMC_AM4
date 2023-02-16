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

subroutine nn_init(num_layers, num_hid_nodes)
    integer, intent(in) :: num_layers, num_hid_nodes
    integer :: ilayer
    Rad_NN_FC%num_layers = num_layers
    Rad_NN_FC%num_hid_nodes = num_hid_nodes
    allocate(Rad_NN_FC%Layers(num_layers))
    do ilayer = 1, num_layers
        allocate(Rad_NN_FC%Layers(ilayer)%weight(num_hid_nodes,num_hid_nodes))
        allocate(Rad_NN_FC%Layers(ilayer)%bias(num_hid_nodes))
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
integer :: num_layers, num_hid_nodes
real, allocatable, dimension(:,:) :: a
real, allocatable, dimension(:)::c
integer:: i,j, ilayer
real :: start, finish
num_layers = 5
num_hid_nodes = 256
!num_layers = 10
!num_hid_nodes = 128
!num_hid_nodes = 512
!num_hid_nodes = 1024
allocate(a(num_hid_nodes, int(1e5)))
a = 2.431452


call nn_init(num_layers, num_hid_nodes)
! allocate(c(size(a,1)))
! call nn_pred_1d(a(:,1),c)
call cpu_time(start)
do j = 1, size(a,2)
    allocate(c(size(a,1)))
    call nn_pred_1d(a(:,j),c)
    deallocate(c)
end do
call cpu_time(finish)
print '("Time = ",f6.3," seconds.")',finish-start
End Program test
