#!/bin/bash
# overhead_p1.sh - start-up/build/JIT cost on P1, cold vs warm persistent kernel caches (15 fresh processes each)
cd "$(dirname "$0")/../build"
export PATH=${ACPP_HOME:-/opt/acpp}/bin:$PATH OMP_NUM_THREADS=1 POCL_CPU_MAX_CU_NUM=1
OUT=p1_overhead.csv; rm -f $OUT $OUT.meta.csv
run() { env $3 taskset -c 0 $1 -k gemm -n 256 -i 5 -w 0 -p P1 -s $4 -o $OUT -d $2 > /dev/null 2>&1; }
for r in $(seq 1 15); do
  run ./bench_cpp serial "" warm
  rm -rf ~/.cache/pocl; run ./bench_ocl Portable:cpu "" cold
  run ./bench_ocl Portable:cpu "" warm
  run ./bench_ocl Intel:cpu "" warm
  rm -rf ~/.acpp/apps; run ./bench_sycl_generic usm ACPP_VISIBILITY_MASK=omp cold
  run ./bench_sycl_generic usm ACPP_VISIBILITY_MASK=omp warm
  rm -rf ~/.acpp/apps; run ./bench_sycl_generic usm ACPP_VISIBILITY_MASK=ocl cold
  run ./bench_sycl_generic usm ACPP_VISIBILITY_MASK=ocl warm
  run ./bench_sycl_omp usm "" warm
done
echo overhead done
