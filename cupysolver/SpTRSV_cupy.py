import cupy as cp
import numpy as np
import cupyx as cpx
import math
from scipy.sparse import csr_matrix, tril
import scipy.linalg
import cupyx.scipy.linalg
import cupyx.scipy.sparse.linalg
import time
import sys


loaded_from_source = r'''
extern "C"{


__global__
void spts_syncfree_cuda_executor_csr(const int* __restrict__            d_csrRowPtr,
                                 const int* __restrict__        	d_csrColIdx,
                                 const double* __restrict__ 		d_csrVal,
                                 int*                          		d_get_value,
                                 const int                      	m,
                                 const int                      	nnz,
                                 const double* __restrict__ 		d_b,
                                 double*                    		d_x,
                                 const int                      	begin,
                                 int*                           	d_id_extractor
                                     )

{
    const int global_id =atomicAdd(d_id_extractor, 1);
//    const int global_id = (begin + blockIdx.x) * blockDim.x + threadIdx.x;
//    if(blockIdx.x==0)
//        printf("%d\n",global_id);
    if(global_id>=m)
        return;

    int col,j,i;
    double xi;
    double left_sum=0;
    i=global_id;
    j=d_csrRowPtr[i];
    
    while(j<d_csrRowPtr[i+1])
    {
        col=d_csrColIdx[j];
        while(d_get_value[col]==1)
            //if(d_get_value[col]==1)
        {
            left_sum+=d_csrVal[j]*d_x[col];
            j++;
            col=d_csrColIdx[j];
        }
        if(i==col)
        {
            xi = (d_b[i] - left_sum) / d_csrVal[d_csrRowPtr[i+1]-1];
            d_x[i] = xi;
            __threadfence();
            d_get_value[i]=1;
            j++;
        }
    }

}


}'''


module = cp.RawModule(code=loaded_from_source)
ker_sptrsv = module.get_function('spts_syncfree_cuda_executor_csr')
#ker_sptrsv_syncfree_analyser = module.get_function('sptrsv_syncfree_cuda_analyser')
#ker_sptrsv_syncfree_executor = module.get_function('sptrsv_syncfree_cuda_executor_update')

def my_sptrsv(A, b, x):
	Acsr = csr_matrix(A)
	m, n=csr_matrix.get_shape(Acsr)
	rows, cols = tril(Acsr).tocsr().nonzero()
	data = tril(Acsr).tocsr()[rows, cols]
	nnzL = np.size(data)

	tic = time.perf_counter()
	csrRowPtrL=np.zeros(m+1, dtype=np.int32)
	csrColIdxL=np.array(cols)
	csrValL= data.reshape(1,nnzL)
	for i in rows:
		for j in range(i+1,m+1):
			csrRowPtrL[j]=csrRowPtrL[j]+1

	dev_csrRowPtrL = cp.asarray(csrRowPtrL)
	dev_csrColIdxL = cp.asarray(csrColIdxL)
	dev_csrValL = cp.asarray(csrValL)
	dev_b = cp.asarray(b)
	dev_x = cp.zeros(m, dtype=cp.float64)
	num_threads = 1024
	num_blocks = math.ceil(m/num_threads)
	d_id_extractor = cp.array([0])
	d_get_value = cp.zeros(m, dtype=cp.int32)
	toc = time.perf_counter()
	print(f"prepare array in {toc - tic:0.8f} seconds")

	tic = time.perf_counter()
	ker_sptrsv((num_blocks,), (num_threads,), (dev_csrRowPtrL, dev_csrColIdxL, dev_csrValL, d_get_value, m, nnzL, dev_b, dev_x, 0, d_id_extractor))
	toc = time.perf_counter()
	print(f"GPU complete L solve in {toc - tic:0.8f} seconds")
	x = cp.asnumpy(dev_x)
	x=x.reshape(m,1)
	print(x)



def CPU_LUfactorsolver_dense(A, b, x):
	tic = time.perf_counter()
	lu, piv = scipy.linalg.lu_factor(A)
	toc = time.perf_counter()
	print(f"CPU complete factor in {toc - tic:0.8f} seconds")

	tic = time.perf_counter()
	x = scipy.linalg.lu_solve((lu, piv), b)
	toc = time.perf_counter()
	print(f"CPU complete solve in {toc - tic:0.8f} seconds")
	print(x)


def GPU_LUfactorsolver_dense(A, b, x):
	x_cp = cp.asarray(x)
	b_cp = cp.asarray(b)
	A_cp = cp.asarray(A)

	tic = time.perf_counter()
	lu_cp, piv_cp = cupyx.scipy.linalg.lu_factor(A_cp)
	toc = time.perf_counter()
	print(f"GPU complete factor in {toc - tic:0.8f} seconds")

	tic = time.perf_counter()
	x_cp = cupyx.scipy.linalg.lu_solve((lu_cp, piv_cp), b_cp)
	toc = time.perf_counter()
	print(f"GPU complete solve in {toc - tic:0.8f} seconds")
	x = cp.asnumpy(x_cp)
	print(x)


def GPU_QRfactorsolver_sparse(A, b, x):
	b_cp = cp.asarray(b)
	x_cp = cp.asarray(x)
	A_cp = cp.asarray(A)
	tic = time.perf_counter()
	x_cp, istop, itn, normr = cupyx.scipy.sparse.linalg.lsqr(A_cp, b_cp)[:4]
	toc = time.perf_counter()
	print(f"GPU complete QR solve in {toc - tic:0.8f} seconds")
	x = x_cp.get()
	x=x.reshape(m,1)
	print(x)

def GPU_solver_triangular(A, b, x):
	Acsr = csr_matrix(A)
	Atri = tril(Acsr).toarray()
	b_cp = cp.asarray(b)
	x_cp = cp.asarray(x)
	Atri_cp = cp.asarray(Atri)
	tic = time.perf_counter()
	x_cp = cupyx.scipy.linalg.solve_triangular(Atri_cp, b_cp)
	toc = time.perf_counter()
	print(f"GPU complete triangular solve in {toc - tic:0.8f} seconds")
	x = cp.asnumpy(x_cp)
	print(x)


m = int(sys.argv[1])

A = np.random.rand(m,m)
b = np.random.rand(m,1)
x = np.zeros(m, dtype=np.float64)

#CPU solver
CPU_LUfactorsolver_dense(A, b, x)

#GPU solver
GPU_LUfactorsolver_dense(A, b, x)

#GPU QR solver
b2 = b.reshape(m)
GPU_QRfactorsolver_sparse(A, b2, x)

#CPU sptrsv

GPU_solver_triangular(A, b, x)

#GPU sptrsv


my_sptrsv(A, b, x)


	



