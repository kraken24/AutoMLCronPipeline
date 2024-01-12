# AutoMLCronPipeline

## Overview
AutoMLCronPipeline is a Python-based solution designed for automating machine learning workflows. It integrates continuous deployment and training strategies without relying on cloud-based services, focusing on cron jobs for scheduling tasks.

## Features
- **Continuous Deployment**: Seamless integration for ML model deployment.
- **Cron-based Scheduling**: Utilize cron jobs for periodic task execution.
- **Machine Learning Automation**: Streamline the process of training and updating models.
- **Cloudless Architecture**: Operate independently of cloud platforms for enhanced privacy and control.

## Requirements
- **UNIX-based System**: Essential for cron job execution.
- **Python Environment**: Ensure Python is installed with necessary libraries for machine learning and automation tasks.
- **Network Access**: For retrieving data and interacting with any APIs if required.

## Usage
Details on how to set up and use the pipeline will be provided, including setup instructions, configuration details, and execution steps.

Create a virtual environment using:
```
conda env create -f environment.yml -n auto_ml
conda activate auto_ml
# or
python3.9 -m venv auto_ml
pip install -r requirements.txt
```

Test execution of training and check if log files are updated:
```
python src/main.py
```

## License
This project is licensed under the MIT License.

---
