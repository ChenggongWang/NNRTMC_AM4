module test_mod
implicit none
! type for NN layer
public NN_Linear_layer_type
type :: NN_Linear_layer_type
    real, dimension(:,:), pointer :: weight=>NULL()
    real, dimension(:),   pointer :: bias=>NULL()
end type NN_Linear_layer_type
! type for NN structure
public NN_FC_type
type :: NN_FC_type
    integer :: num_hid_nodes
    integer :: num_layers
    type(NN_Linear_layer_type), dimension(:), pointer:: Layers
end type NN_FC_type

type(NN_FC_type)   :: Rad_NN_FC

contains
! initialize NN 
subroutine nn_init(num_layers, num_hid_nodes)
    integer, intent(in) :: num_layers, num_hid_nodes
    integer :: ilayer, i, j 
    Rad_NN_FC%num_layers = num_layers
    Rad_NN_FC%num_hid_nodes = num_hid_nodes
    allocate(Rad_NN_FC%Layers(num_layers))
    ! init each lay
    call random_seed()
    do ilayer = 1, num_layers
        allocate(Rad_NN_FC%Layers(ilayer)%weight(num_hid_nodes,num_hid_nodes))
        allocate(Rad_NN_FC%Layers(ilayer)%bias(num_hid_nodes))
        call random_number(Rad_NN_FC%Layers(ilayer)%weight)
        call random_number(Rad_NN_FC%Layers(ilayer)%bias)
        Rad_NN_FC%Layers(ilayer)%weight = Rad_NN_FC%Layers(ilayer)%weight - 0.5
        Rad_NN_FC%Layers(ilayer)%bias = Rad_NN_FC%Layers(ilayer)%bias - 0.5
    end do
end subroutine nn_init

! run NN with one input (one column)
subroutine nn_pred_1d_matmul(x,y)
    real, dimension(:), intent(in) :: x
    real, dimension(:), intent(inout) :: y
    real, dimension(:), allocatable :: interm
    integer::ilayer
    allocate(interm(Rad_NN_FC%num_hid_nodes))
    interm = x
    ! num_layers matmul called
    do ilayer = 1, Rad_NN_FC%num_layers
        y = matmul(interm,Rad_NN_FC%Layers(ilayer)%weight)+ Rad_NN_FC%Layers(ilayer)%bias
        y = NN_activ(y)
        interm = y
    end do
    deallocate(interm)
end subroutine nn_pred_1d_matmul
subroutine nn_pred_1d_loop(x,y)
    real, dimension(:), intent(in) :: x
    real, dimension(:), intent(inout) :: y
    real, dimension(:), allocatable :: interm
    integer:: ilayer, k
    allocate(interm(Rad_NN_FC%num_hid_nodes))
    interm = x
    ! num_layers matmul called
    do ilayer = 1, Rad_NN_FC%num_layers
        do k = 1, size(rad_nn_fc%layers(ilayer)%weight, 2)
            y(k) = sum(interm(:)*rad_nn_fc%layers(ilayer)%weight(:,k)) + rad_nn_fc%layers(ilayer)%bias(k)
        enddo
        y = NN_activ(y)
        interm = y
    end do
end subroutine nn_pred_1d_loop
! run NN with one input (one column)
subroutine nn_pred_1d_sgemm(x,y)
    real, dimension(:), intent(in) :: x
    real, dimension(:), intent(inout) :: y
    ! local
    integer :: ilayer
    real, dimension(:), allocatable :: interm
    ! for sgemm
    integer :: m, k, n, i, j
    ! num_layers matmul called
    m = 1
    n = Rad_NN_FC%num_hid_nodes
    k = Rad_NN_FC%num_hid_nodes
    allocate(interm(Rad_NN_FC%num_hid_nodes))
    interm = x
    do ilayer = 1, Rad_NN_FC%num_layers
        y =  Rad_NN_FC%Layers(ilayer)%bias
        call SGEMM('N','N',m,n,k,1.0,interm,m,Rad_NN_FC%Layers(ilayer)%weight,k,1.0,y,m)
        y = NN_activ(y)
        interm = y
    end do
    deallocate(interm)
end subroutine nn_pred_1d_sgemm
! run NN with one input (one column)
subroutine nn_pred_1d_sgemv(x,y)
    real, dimension(:), intent(in) :: x
    real, dimension(:), intent(inout) :: y
    ! local
    real, dimension(:), allocatable :: interm
    ! for sgemm
    integer :: m, n
    integer :: ilayer
    m = Rad_NN_FC%num_hid_nodes
    n = Rad_NN_FC%num_hid_nodes
    allocate(interm(Rad_NN_FC%num_hid_nodes))
    interm = x
    ! num_layers matmul called
    do ilayer = 1, Rad_NN_FC%num_layers
        y = Rad_NN_FC%Layers(ilayer)%bias
        call SGEMV('T',m,n,1.0,Rad_NN_FC%Layers(ilayer)%weight,m,interm,1,1.0,y,1)
        y = NN_activ(y)
        interm = y
    end do
    deallocate(interm)
end subroutine nn_pred_1d_sgemv

real elemental function NN_activ(x)
    real, intent(in) :: x
    ! ReLU:
    !if (x>0) then
    !    NN_activ = x
    !else
    !    NN_activ = 0
    !end if
    NN_activ = max(0.0,x)
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
integer:: i,j, ilayer, total_run
real :: start, finish

num_layers = 5
num_hid_nodes = 256
call nn_init(num_layers, num_hid_nodes)

total_run = int(1e4)
!total_run = int(1)
! dummpy input data with total_run columns
allocate(a(num_hid_nodes, total_run))
call random_seed()
call random_number(a)
a = a/10.0

print '("total run times:", I7)', total_run
! test matmul
call cpu_time(start)
do j = 1, total_run
    allocate(c(size(a,1)))
    call nn_pred_1d_matmul(a(:,j),c)
    if (j==1) then ; print *,c(:2); end if
    deallocate(c)
end do
call cpu_time(finish)
print '("matmul avg time per run = ",f6.3," ms")', (finish-start)/total_run*1e3
! test loop
call cpu_time(start)
do j = 1, total_run
    allocate(c(size(a,1)))
    call nn_pred_1d_loop(a(:,j),c)
    if (j==1) then ; print *,c(:2); end if
    deallocate(c)
end do
call cpu_time(finish)
print '("loop  avg time per run = ",f6.3," ms")', (finish-start)/total_run*1e3

! test sgemm
call cpu_time(start)
do j = 1, total_run
    allocate(c(size(a,1)))
    call nn_pred_1d_sgemm(a(:,j),c)
    if (j==1) then ; print *,c(:2); end if
    deallocate(c)
end do
call cpu_time(finish)
print '("SGEMM  avg time per run = ",f6.3," ms")', (finish-start)/total_run*1e3

! test sgemv
call cpu_time(start)
do j = 1, total_run
    allocate(c(size(a,1)))
    call nn_pred_1d_sgemv(a(:,j),c)
    if (j==1) then ; print *,c(:2); end if
    deallocate(c)
end do
call cpu_time(finish)
print '("SGEMV  avg time per run = ",f6.3," ms")', (finish-start)/total_run*1e3

End Program test
