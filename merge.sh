#!/bin/bash

# SLURM configuration - OPTIMIZED for 7B models
partition="dev-g"
time="3:00:00"  # Reduced time - 7B merging is fast
account="project_462000919"
scratch_path="/scratch/project_462000919/yagao"

# Generate run name
if [ $# -eq 1 ]; then
    run_name=$1
else
    run_name="$(date +"%m%d")_$(python -c "import random, string; print(''.join(random.choices(string.ascii_lowercase + string.digits, k=8)))")"
fi

# Confirmation
read -p "Run name: $run_name. Proceed? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

# Create directories
output_dir="${scratch_path}/checkpoints/Colla-LLM/merged_models/${run_name}"
logs_dir="${scratch_path}/logs/mergekit"
mkdir -p "${output_dir}" "${logs_dir}"

echo "Submitting job..."
echo "Output: ${output_dir}"
echo "Logs: ${logs_dir}"

job_id=$(sbatch <<EOL | awk '{print $4}'
#!/bin/bash
#SBATCH --account=${account}
#SBATCH --partition=${partition}
#SBATCH --job-name="merge_${run_name}"
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --time=${time}
#SBATCH --mail-user=ya.gao@aalto.fi
#SBATCH --mail-type=FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH --gpus-per-node=2
#SBATCH --mem=120G
#SBATCH --output=${logs_dir}/merge_${run_name}_%j.out
#SBATCH --error=${logs_dir}/merge_${run_name}_%j.err

echo "=========================================="
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_NODELIST"
echo "Start Time: \$(date)"
echo "Run Name: ${run_name}"
echo "=========================================="

# Load modules
module use /appl/local/csc/modulefiles/
module load pytorch/2.5

cd ${scratch_path}/code_repo/Colla-LLM/mergekit
source .venv/bin/activate

# Environment info
echo "Python: \$(python --version)"
echo "PyTorch: \$(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: \$(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: \$(python -c 'import torch; print(torch.cuda.device_count())')"

# Run merge with optimal settings
echo "Starting merge at \$(date)..."
mergekit-yaml config/linear.yaml "${output_dir}" --gpu-rich --verbose

if [ \$? -eq 0 ]; then
    echo "=========================================="
    echo "Merge completed successfully at \$(date)"
    echo "Output: ${output_dir}"
    ls -la "${output_dir}"
    echo "=========================================="
else
    echo "Merge failed at \$(date)"
    exit 1
fi
EOL
)

echo "Job submitted: ${job_id}"
echo "Log: ${logs_dir}/merge_${run_name}_${job_id}.out"
