# Repository Maintenance Setup

This guide explains how to set up automatic repository maintenance that checks and fixes issues with your repositories.

## What it does

The repository maintenance system automatically:
- Checks for repositories with generic descriptions
- Fixes missing ARCHITECTURE.md files
- Updates README.md files with proper content
- Adds appropriate topics/tags to repositories
- Cleans up references to deleted repositories
- Ensures all projects are properly configured

## Option 1: Using Cron (Recommended for most systems)

Run the setup script:

```bash
export $(cat .env.local | grep -v '^#' | xargs)
./setup_maintenance_cron.sh
```

This will:
- Set up a cron job that runs every 6 hours
- Log output to `maintenance_cron.log`

To check if it's working:
```bash
crontab -l
```

To remove the cron job:
```bash
crontab -l | grep -v 'scheduled_repository_maintenance.py' | crontab -
```

## Option 2: Using systemd (For systems with systemd)

Run the installer with sudo:

```bash
sudo ./install_maintenance_service.sh
```

This will:
- Install a systemd service and timer
- Run maintenance every 6 hours
- Provide better logging and control

To check status:
```bash
systemctl status repository-maintenance.timer
systemctl status repository-maintenance.service
```

To view logs:
```bash
journalctl -u repository-maintenance.service -f
```

## Option 3: Integration with Continuous Orchestrator

The repository maintenance is already integrated into the continuous orchestrator. When running the orchestrator, it will:
- Check for repositories needing fixes every 30 minutes
- Create maintenance tasks for repositories with issues
- Automatically apply fixes without creating GitHub issues

No additional setup needed if you're already running the orchestrator.

## Manual Execution

You can run maintenance manually at any time:

```bash
export $(cat .env.local | grep -v '^#' | xargs)
python scheduled_repository_maintenance.py
```

Or to check specific repositories:

```bash
export $(cat .env.local | grep -v '^#' | xargs)
python fix_repository_customizations.py --check
python fix_repository_customizations.py --fix --with-ai
```

## Configuration

The maintenance system uses:
- GitHub token from `.env.local`
- AI brain for generating content
- Excludes system repositories (cwmai, .github)

## Monitoring

Check the logs to ensure maintenance is running:
- Cron: `tail -f maintenance_cron.log`
- Systemd: `journalctl -u repository-maintenance.service -f`
- Manual: `tail -f repository_maintenance.log`

## Troubleshooting

1. **Permission errors**: Ensure your GitHub token has repo write access
2. **AI errors**: Check that AI API keys are properly configured
3. **No fixes applied**: Some repositories may be empty or have other issues preventing fixes

## Benefits

- **Consistency**: All repositories maintain proper documentation
- **Automation**: No manual intervention needed
- **Intelligence**: Uses AI to generate project-specific content
- **Reliability**: Multiple scheduling options available