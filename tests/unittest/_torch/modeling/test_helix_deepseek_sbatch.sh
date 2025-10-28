#!/bin/bash

if [ -z "$WORKDIR" ] || \
   [ -z "$CONTAINER_NAME" ] || \
   [ -z "$CONTAINER_IMAGE" ] || \
   [ -z "$CONTAINER_MOUNT" ] || \
   [ -z "$REPO_DIR" ] || \
   [ -z "$LLM_MODELS_ROOT" ] || \
   [ -z "$SLURM_ACCOUNT" ] || \
   [ -z "$SLURM_JOB_NAME" ] || \
   [ -z "$SLURM_PARTITION" ]; then
  echo "Environment not set, please source test_helix_deepseek_sbatch_env.sh first"
  exit 1
fi

TP=${1:-8}
KVP=${2:-1}
EP=${3:-2}
DENSE=${4:-0}
SINGLE_CTX_LEN=${5:-0}
BATCH=${6:-1}
QUANT=${7:-"nvfp4"}
PROF=${8:-0}

dense_arg=""
if (( DENSE == 1 )); then
  dense_arg="--dense"
fi
ctx_len_arg=""
if (( SINGLE_CTX_LEN != 0 )); then
  ctx_len_arg="--ctx_len_start ${SINGLE_CTX_LEN} --ctx_len_end ${SINGLE_CTX_LEN}"
fi
world_size=$((TP * KVP))
if (( world_size % EP != 0 )); then
  echo "World size $world_size must be a multiple of EP $EP"
  exit 1
fi
gpus_per_node=4
NODES=$(((world_size + gpus_per_node - 1) / gpus_per_node))
gpus=$((NODES * gpus_per_node))
sbatch <<EOF
#!/bin/bash
#SBATCH --nodes=${NODES}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --time=01:00:00
#SBATCH --job-name=${SLURM_JOB_NAME}
#SBATCH --comment=fact_off
#SBATCH --gres=gpu:${gpus_per_node}
#SBATCH --segment=${NODES}
#SBATCH --ntasks=${gpus}
#SBATCH --ntasks-per-node=${gpus_per_node}

cleanup_on_failure() {
    echo "Error: \$1"
    scancel \${SLURM_JOB_ID}
    exit 1
}

set -x

logdir=\${WORKDIR}/slurm-\${SLURM_JOB_ID}
mkdir -p \${logdir}
full_logdir=\${logdir}

nsys_prefix=""
if (( ${PROF} == 1 )); then
  nsys_file=\${full_logdir}/test_helix_deepseek_\${SLURM_JOB_ID}_tp${TP}_kvp${KVP}_ep${EP}_dense${DENSE}_ctx_len${SINGLE_CTX_LEN}_batch${BATCH}_quant${QUANT}
  nsys_prefix="nsys profile -e \"NSYS_MPI_STORE_TEAMS_PER_RANK=1\" -o \${nsys_file} -f true -s none --cpuctxsw=none -t cuda -c cudaProfilerApi --cuda-graph-trace node --capture-range-end=stop --gpu-metrics-devices=none"
fi

echo "Starting container..."
if ! srun -l --container-image=\${CONTAINER_IMAGE} \
        --container-name=\${CONTAINER_NAME} \
        --container-mounts=\${CONTAINER_MOUNT} \
        --mpi=pmix \
        echo "Container up." &> \${full_logdir}/container_launch.log; then
    cleanup_on_failure "Failed to start container. Check \${full_logdir}/container_launch.log"
fi

export TLLM_LOG_LEVEL="INFO" # DEBUG is verbose
export TRTLLM_ENABLE_PDL=1
export LD_LIBRARY_PATH=/workspace/TensorRT-LLM/cpp/build/tensorrt_llm/executor/cache_transmission/ucx_utils:\${LD_LIBRARY_PATH}

bench_cmd="python3 \${REPO_DIR}/tests/unittest/_torch/modeling/test_helix_deepseek.py --type v3 --quant ${QUANT} --batch ${BATCH} --tp ${TP} --kvp ${KVP} --ep ${EP} ${dense_arg} ${ctx_len_arg}"

echo "====== Baseline ========"
srun --mpi pmix -N ${NODES} --ntasks ${world_size} --ntasks-per-node ${gpus_per_node} \
  --container-env=MASTER_ADDR,MASTER_PORT \
  --container-name=\${CONTAINER_NAME} \
  --container-mounts=\${CONTAINER_MOUNT} \
  bash -c "if [ "\\\$SLURM_PROCID" = "0" ]; then \${nsys_prefix} \${bench_cmd}; else \${bench_cmd}; fi" \
  &> \${full_logdir}/benchmark.log 2>&1
EOF
