#!/bin/bash
# run_p1_campaign.sh - measurement campaign on platform P1 (Intel Xeon Cascade Lake, 2 vCPU, KVM guest)
# Critical partition: benchmark pinned to CPU0, single worker thread.
# Interference: enemy (1 thread, 64 MiB streaming RMW) pinned to CPU1.
cd "$(dirname "$0")/../build"
export PATH=${ACPP_HOME:-/opt/acpp}/bin:$PATH OMP_NUM_THREADS=1 POCL_CPU_MAX_CU_NUM=1 OMP_PROC_BIND=true
OUT=${OUT:-p1_results.csv}
PLAN="gemm:256:1000 gemm:512:300 conv:512:1000 conv:1024:500 kalman:4096:1000 kalman:32768:500 null:1:5000"
CONFIGS=(
  "./bench_cpp|serial|"
  "./bench_ocl|Portable:cpu|"
  "./bench_ocl|Intel:cpu|"
  "./bench_sycl_generic|usm|ACPP_VISIBILITY_MASK=omp"
  "./bench_sycl_generic|buf|ACPP_VISIBILITY_MASK=omp"
  "./bench_sycl_omp|usm|"
  "./bench_sycl_generic|usm|ACPP_VISIBILITY_MASK=ocl"
  "./bench_sycl_generic|buf|ACPP_VISIBILITY_MASK=ocl"
)
echo "=== P1 campaign start $(date -Is)"
for sc in iso intf; do
  EPID=""
  if [ $sc = intf ]; then taskset -c 1 ./enemy 64 100000 1 > /dev/null & EPID=$!; sleep 2; fi
  for p in $PLAN; do
    IFS=: read k n it <<< "$p"
    for c in "${CONFIGS[@]}"; do
      IFS='|' read bin dev envs <<< "$c"
      [ $k = null ] && [ $bin = ./bench_cpp ] && continue
      env $envs taskset -c 0 timeout 900 $bin -k $k -n $n -i $it -w 20 -p P1 -s $sc -o $OUT -d $dev 2>&1 | grep '^\[' | grep -v Warning
    done
  done
  [ -n "$EPID" ] && kill $EPID
done
echo "=== P1 campaign end $(date -Is)"
