
# Understanding Cron Jobs
## What is a Cron Job?

A **cron job** is a scheduled task that is automated to run at specified intervals or times. It is a Unix/Linux utility that is typically used for system maintenance or administration. One of the innovative applications of cron job is to setup automated continuous training and deployment pipelines for machine learning models. It is a very powerful alternative to setting up pipelines on various cloud services.

### Benefits of Cron Jobs
- **Automation**: Once set up, cron jobs run automatically at the scheduled time without further intervention.
- **Consistency**: Tasks can be performed consistently at the same time, ensuring regular maintenance or backups.
- **Efficiency**: Offloading tasks to be run during off-peak hours can help with resource management and cost savings.
- **Flexibility**: Cron jobs can be configured to run at nearly any conceivable time or date interval.

## Cron Job Command Structure

A cron job is defined by a line of text, known as a "cron expression", that consists of five or six fields separated by spaces, followed by the command to be executed.

```
* * * * * command-to-execute
│ │ │ │ │
│ │ │ │ │
│ │ │ │ └─── Day of week (0 - 7) (Sunday to Saturday; 7 is also Sunday)
│ │ │ └───────── Month (1 - 12)
│ │ └─────────────── Day of the month (1 - 31)
│ └──────────────────── Hour (0 - 23)
└───────────────────────── Minute (0 - 59)
```

- `*` (asterisk) - Stands for "every" or "all values".

### Examples:

- `0 * * * *` - Run the command at the start of every hour.
- `30 1 * * *` - Run the command daily at 1:30 AM.
- `45 4 * * 2` - Run the command every Tuesday at 4:45 AM.
- `0 0 1 * *` - Run the command on the first day of every month at midnight.

### Basic Commands:
`crontab -l  # list cron jobs for the user`
`crontab -e  # create a new cron job  # for first time users, they might have to select default file editor in the terminal.`
`crontab -r  # remove the crontab files`
`crontab -u username -e  # edit crontab file of a different user "username"`
## Setting Up a Cron Job

To set up a cron job, follow these steps:

1. **Open the Crontab Configuration**:
   Open a terminal session and type `crontab -e`. This will open the crontab file for the current user in the default editor.

2. **Add a Cron Job**:
   At the bottom of the crontab file, add a line that specifies the schedule and command for your cron job.

3. **Save the Crontab File**:
   Save and close the file. The cron daemon will automatically pick up the new job and begin executing it at the specified time.

4. **Verify Your Cron Jobs**:
   List all cron jobs for the current user with `crontab -l` to verify that your new cron job has been added correctly.

### Example Entry:
```crontab
# Retrain a machine learning model every week on sunday at 11 pm
0 23 * * sun /bin/bash -c 'cd /path/to/this/repository && source /path/to/your/conda/bin/activate conda_env_name && python ./src/main.py'
```

### Resources:
- Generate cron job command online:
   - https://crontab.guru
   - https://crontab-generator.org
- Comprehensive Guide: https://www.sitepoint.com/cron-jobs/
