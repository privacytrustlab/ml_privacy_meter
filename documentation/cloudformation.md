# AWS Sagemaker Deployment Guide for Privacy Meter

## Authors
| Author | Contribution | Version | Date |
|--------|-------------|----------|------|
| Rob van Eijk [@rvaneijk](https://github.com/rvaneijk) | AWS SageMaker CloudFormation template and documentation | 1.0 | 2025-01-25 |
| Rob van Eijk [@rvaneijk](https://github.com/rvaneijk) | Multi-GPU support and documentation | 1.1 | 2025-02-08 |

## Changelog
### v1.1 (2025-02-08)
- Added multi-GPU support with PyTorch distributed training
- Enhanced CloudFormation templates for larger GPU instances
- Added real-time training progress monitoring
- Split templates into production/development and single/multi-GPU variants

### v1.0 (2025-01-25)
- Initial CloudFormation template for Privacy Meter deployment on AWS SageMaker
- Custom Conda environment with PyTorch GPU support 
- Automated package verification system
- Documentation and deployment guide

# Available Templates

The deployment offers four CloudFormation template variants to match your needs:

1. **cloudformation-template.yml (single GPU)**
   - Production environment, single GPU
   - Default instance: ml.g4dn.xlarge (1x T4 GPU)
   - Suitable for: Production deployments with smaller workloads
   - Stable release repository

2. **cloudformation-template-multi-gpu.yml (4 GPUs)**
   - Production environment, multiple GPUs
   - Default instance: ml.g4dn.12xlarge (4x T4 GPUs)
   - Suitable for: Production deployments with larger workloads
   - Stable release repository

3. **cloudformation-template-dev.yml (single GPU)**
   - Development environment, single GPU
   - Default instance: ml.g4dn.xlarge (1x T4 GPU)
   - Suitable for: Development and testing with smaller workloads
   - Development repository with latest features

4. **cloudformation-template-dev-multi-gpu.yml (4 GPUs)**
   - Development environment, multiple GPUs
   - Default instance: ml.g4dn.12xlarge (4x T4 GPUs)
   - Suitable for: Development and testing with larger workloads
   - Development repository with latest features

# Deployment Guide

This CloudFormation template deploys Privacy Meter as an AWS SageMaker notebook instance. Key features include:

[Previous features list remains the same...]

**Disclaimer:** This template is provided as-is without any warranties. Users are responsible for understanding AWS costs, security implications, and maintaining their deployments. Always review and test the template before deploying in a production environment.

### Quick Start (note: deployment may sometimes take a few minutes or more)

1. Select appropriate template:
   - For production use:
     - Single GPU: `cloudformation-template.yml`
     - Multiple GPUs: `cloudformation-template-multi-gpu.yml`
   - For development use:
     - Single GPU: `cloudformation-template-dev.yml`
     - Multiple GPUs: `cloudformation-template-dev-multi-gpu.yml`

2. Deploy using AWS console:
   - Navigate to AWS CloudFormation console
   - Click "Create stack" (with new resources)
   - Upload your selected template file
   - Follow the prompts to complete deployment

   Alternatively, deploy using AWS CLI:
   ```bash
   aws cloudformation create-stack \
     --stack-name privacy-meter-dev \
     --template-body file://[selected-template].yml \
     --capabilities CAPABILITY_IAM
   ```

3. Access your notebook:
   - Navigate to the AWS SageMaker console
   - Click "Notebook instances"
   - Find your instance (default name: PrivacyMeterNotebook)
   - Click "Open JupyterLab"

# Instance Types and GPU Support

## Available Configurations
The template supports various instance types based on your workload:

1. **Single GPU (Small Workloads, single GPU)**
   - Instance: `ml.g4dn.xlarge`
   - 1x NVIDIA T4 GPU
   - 4 vCPUs
   - 16 GB RAM

2. **Multi-GPU (Medium Workloads, 4 GPUs)**
   - Instance: `ml.g4dn.12xlarge` (default)
   - 4x NVIDIA T4 GPUs
   - 48 vCPUs
   - 192 GB RAM
   - Up to 900 GB NVMe SSD

## Multi-GPU Usage
The `demo_aws_multi_gpu.ipynb` notebook provides multi-GPU training capabilities:

```python
# Configure multi-GPU training
models_list = parallel_prepare_models(
    log_dir, 
    dataset, 
    data_splits,
    memberships,
    configs, 
    logger,
    num_gpus=4  # Adjust based on instance type
)
```

Features include:
- Real-time training progress for each GPU
- Memory utilization tracking
- Cost estimation for spot instances
- Comprehensive logging and metadata collection

# Configurable Parameters
   - Instance name (default: PrivacyMeterNotebook)
   - GPU instance type (based on workload size)
   - Repository URL (privacy_meter or privacy_meter_dev)
   - Python version (default: 3.10)
   - Kernel name (default: conda_privacymeter_p310)
   - PyTorch Installation URL (Default: CUDA 11.8 build of PyTorch https://download.pytorch.org/whl/cu118)
   - Number of GPUs to utilize (default: all available)

# Troubleshooting

1. Monitor setup progress and check the log file:
   - **Installation Complete Flag:** Wait for `/home/ec2-user/SageMaker/setup-complete` file to appear
   -  **Log files:** Check `/home/ec2-user/SageMaker/setup.log` for installation progress and errors. The tail of the file should look like:
   ```bash
   All packages match required versions!
   Installed kernelspec privacymeter_p310 in /home/ec2-user/.local/share/jupyter/kernels/privacymeter_p310
   ```
2. **Verification Notebook:** Verify the conda environment is active in the notebook `verification.ipynb`:
   - Launch a Jupyter notebook with the privacymeter kernel **preferred kernel: conda_privacy_meter_p310**
   - Reload with a different kernel first if the list of installed kernels has not been updated yet
   - Check the last line of the output of Cell 4 in the notebook. It should look like:
   ```bash
   All packages match required versions!
   ```
3. **Conda Env:** Verify the conda environment is active from a AWS SageMaker Terminal window:
   - Make sure to match the extension in 'conda activate privacymeter_**p310**' with the Python version in the Cloudformation template
   ```bash
   export WORKING_DIR=/home/ec2-user/SageMaker/custom-miniconda
   "$WORKING_DIR/miniconda/bin/conda" init bash
   source ~/.bashrc
   conda activate privacymeter_p310
   conda info --envs
   ```
   - The output should look like:
   ```bash
   # conda environments:
   #
   base                     /home/ec2-user/SageMaker/custom-miniconda/miniconda
   privacymeter_p310     *  /home/ec2-user/SageMaker/custom-miniconda/miniconda/envs/privacymeter_p310
                         /home/ec2-user/anaconda3
                            /home/ec2-user/anaconda3/envs/JupyterSystemEnv
                         /home/ec2-user/anaconda3/envs/R
                            /home/ec2-user/anaconda3/envs/python3
                         /home/ec2-user/anaconda3/envs/pytorch_p310
                            /home/ec2-user/anaconda3/envs/tensorflow2_p310
   ```

## Cleanup
To remove all resources go to the AWS Cloudformation console and select the stack you want to delete.
  
Alternatively, remove all resources using AWS CLI:
```bash
aws cloudformation delete-stack --stack-name privacy-meter-dev
```

## Monitoring Costs
   - Stop the notebook when not in use via the AWS Cloudformation console
  
     Alternatively, deploy using AWS CLI:
     ```bash
     aws sagemaker stop-notebook-instance --notebook-instance-name PrivacyMeterNotebook
     ```
   - Consider scheduling automatic starts/stops during work hours
   - Monitor costs through AWS Cost Explorer and set up billing alarms as needed
   - For multi-GPU instances, be especially mindful of usage as costs scale with GPU count