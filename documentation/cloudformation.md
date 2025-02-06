# AWS Sagemaker Deployment Guide for Privacy Meter

## Authors
| Author | Contribution | Version | Date |
|--------|-------------|----------|------|
| Rob van Eijk [@rvaneijk](https://github.com/rvaneijk) | AWS SageMaker CloudFormation template and documentation | 1.0 | 2025-01-25 |

## Changelog
### v1.0 (2025-01-25)
- Initial CloudFormation template for Privacy Meter deployment on AWS SageMaker
- Custom Conda environment with PyTorch GPU support 
- Automated package verification system
- Documentation and deployment guide

# Deployment Guide

This CloudFormation template deploys Privacy Meter as an AWS SageMaker notebook instance. Key features include:

- Custom Conda environment (Python 3.10) with PyTorch GPU support (CUDA 11.8)
- Persistent custom Conda environment that survives instance stops/restarts
- Git integration with Privacy Meter repositories
- Automated verification of pinned versions in requirements.txt and reconciliation system
- 50GB EBS volume with KMS encryption
- Amazon Linux 2 platform with JupyterLab interface
- Built-in troubleshooting with setup logs (/home/ec2-user/SageMaker/setup.log) and verification notebook for environment, GPU, and package validation

**Disclaimer:** This template is provided as-is without any warranties. Users are responsible for understanding AWS costs, security implications, and maintaining their deployments. Always review and test the template before deploying in a production environment.

### Quick Start (note: deployment may sometimes take a few minutes or more)

1. Deploy using AWS console:
   - Navigate to AWS CloudFormation console
   - Click "Create stack" (with new resources)
   - Upload the template file 'cloudformation-template.yml' for the release repository or 'cloudformation-template-dev.yml' if you want to work with the development repository
   - Follow the prompts to complete deployment

   Alternatively, deploy using AWS CLI:
   ```bash
   aws cloudformation create-stack \
     --stack-name privacy-meter-dev \
     --template-body file://cloudformation-template.yml \
     --capabilities CAPABILITY_IAM
   ```
2. Access your notebook:
   - Navigate to the AWS SageMaker console
   - Click "Notebook instances"
   - Find your instance (default name: PrivacyMeterNotebook)
   - Click "Open JupyterLab"

# Configurable parameters:
   - Instance name (default: PrivacyMeterNotebook)
   - GPU instance type (default: ml.g4dn.xlarge)
   - Repository URL (privacy_meter or privacy_meter_dev)
   - Python version (default: 3.10)
   - Kernel name (default: conda_privacymeter_p310)
   - PyTorch Installation URL (Default: CUDA 11.8 build of PyTorch https://download.pytorch.org/whl/cu118)

# Troubleshooting

1. Monitor setup progress and check the log file:
   - Wait for `/home/ec2-user/SageMaker/setup-complete` file to appear
   - Check `/home/ec2-user/SageMaker/setup.log` for installation progress and errors. The tail of the file should look like:
   ```bash
   All packages match required versions!
   Installed kernelspec privacymeter_p310 in /home/ec2-user/.local/share/jupyter/kernels/privacymeter_p310
   ```
2. Verify the conda environment is active in the Python notebook `cloudformation-template-checker.ipynb`:
   - Launch a Jupyter notebook with the privacymeter kernel **preferred kernel: conda_privacy_meter_p310**. 
   - Reload with a different kernel first if the list of installed kernels has not been updated yet.
   - Check the last line of the output of Cell 4 in the notebook. It should look like:
   ```bash
   All packages match required versions!
   ```
3. Verify the conda environment is active from a AWS SageMaker Terminal window. 
   - Make sure to match the extenstion in 'conda activate privacymeter_**p310**' with the Python version in the Cloudformation template.
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
  
Alternatively, remove all resources  using AWS CLI:
```bash
aws cloudformation delete-stack --stack-name privacy-meter-dev
```

## Monitoring Costs
   - Stop the notebook when not in go to the AWS Cloudformation console.
  
     Alternatively, deploy using AWS CLI:
     ```bash
     aws sagemaker stop-notebook-instance --notebook-instance-name PrivacyMeterNotebook
     ```
   - Consider scheduling automatic starts/stops during work hours
   - Monitor costs through AWS Cost Explorer and set up billing alarms as needed.