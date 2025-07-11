import boto3
import time
from datetime import datetime, timedelta

# === Configuration Section ===

# Replace these values with actual values from your AWS environment
AMI_ID = 'ami-0f7a9d46859727c76'  # AMI with stream.py and Kafka configured
INSTANCE_TYPE = 't2.medium'
#KEY_NAME = 'your-key-pair-name'
SECURITY_GROUP_ID = 'sg-0e4c516f78c3e2e70'
SUBNET_ID = 'subnet-0159083c230628552'

# CPU thresholds and instance scaling limits
CPU_HIGH_THRESHOLD = 60
CPU_LOW_THRESHOLD = 10
INSTANCE_LIMIT = 2
CHECK_INTERVAL = 30  # in seconds

# EC2 instance tag for filtering
PROJECT_TAG = {'Key': 'Project', 'Value': 'covid-stream-worker'}

# AWS clients
ec2 = boto3.client('ec2')
cw = boto3.client('cloudwatch')

def get_worker_instances():
    """
    Get the list of running EC2 instance IDs that are tagged for this project.
    """
    response = ec2.describe_instances(
        Filters=[
            {'Name': 'tag:Project', 'Values': [PROJECT_TAG['Value']]},
            {'Name': 'instance-state-name', 'Values': ['running']}
        ]
    )
    instances = []
    for reservation in response['Reservations']:
        for instance in reservation['Instances']:
            instances.append(instance['InstanceId'])
    return instances

def get_average_cpu(instance_ids):
    """
    Compute the average CPU utilization over the last 5 minutes for the given instances.
    """
    if not instance_ids:
        return 0

    #end = datetime.utcnow() 
    end = datetime.utcnow() - timedelta(minutes=1)
    start = end - timedelta(minutes=5)
    total_cpu = 0
    valid_count = 0

    for instance_id in instance_ids:
        response = cw.get_metric_statistics(
            Namespace='AWS/EC2',
            MetricName='CPUUtilization',
            Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
            StartTime=start,
            EndTime=end,
            Period=300,
            Statistics=['Average']
        )
        print(f"Datapoints for {instance_id}: {response['Datapoints']}")
        print(f"Time Range: {start} to {end}")
        datapoints = response.get('Datapoints', [])
        if datapoints:
            latest = sorted(datapoints, key=lambda x: x['Timestamp'], reverse=True)[0]
            avg = latest['Average']
            total_cpu += avg
            valid_count += 1

    return total_cpu / valid_count if valid_count else 0

def launch_instance():
    """
    Launch a new EC2 instance and run stream.py using UserData.
    """
    print("Launching new EC2 instance...")
    try:
        response = ec2.run_instances(
            ImageId=AMI_ID,
            InstanceType=INSTANCE_TYPE,
            MinCount=1,
            MaxCount=1,
            #KeyName=KEY_NAME,
            SecurityGroupIds=[SECURITY_GROUP_ID],
            SubnetId=SUBNET_ID,
            UserData='''#!/bin/bash
            cd /home/ec2-user/environment/x23418613-c9-env/environment/Stream
            nohup python3 stream.py > stream.log 2>&1 &
            ''',
            TagSpecifications=[
                {
                    'ResourceType': 'instance',
                    'Tags': [PROJECT_TAG]
                }
            ]
        )
        instance_id = response['Instances'][0]['InstanceId']
        print(f"Instance {instance_id} launched and tagged.")
    except Exception as e:
        print("Failed to launch EC2 instance:", e)

def terminate_instance(instance_id):
    
    #Terminate the specified EC2 instance.
    
    try:
        ec2.terminate_instances(InstanceIds=[instance_id])
        print(f"Terminated EC2 instance: {instance_id}")
    except Exception as e:
        print("Failed to terminate instance:", e)

# === Auto-Scaling Controller Loop ===

print("\nAuto-Scaler Started (CPU-based Monitoring)\n")

while True:
    worker_instances = get_worker_instances()
    average_cpu = get_average_cpu(worker_instances)
    print(f"{len(worker_instances)} worker instance(s) running | Average CPU: {average_cpu:.2f}%")

    # Scale out if CPU is high and under instance limit
    if average_cpu > CPU_HIGH_THRESHOLD and len(worker_instances) < INSTANCE_LIMIT:
        print("High CPU usage detected. Launching new instance.")
        launch_instance()

    # Scale in if CPU is low and more than 1 instance running
    elif average_cpu < CPU_LOW_THRESHOLD and len(worker_instances) > 1:
        print("Low CPU usage detected. Terminating one instance.")
        terminate_instance(worker_instances[-1])

    else:
        print("No scaling action needed at this time.")

    time.sleep(CHECK_INTERVAL)
