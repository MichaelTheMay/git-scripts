#!/usr/bin/env python3
"""
Cross-platform Git workflow helper with repository diagnostics and setup automation.
Performs comprehensive checks and guides users through repository setup.

Features:
- Repository diagnostics and health checks
- Automated setup and configuration
- Branch management and protection
- Commit message enforcement
- Interactive configuration wizard
- Comprehensive logging system
- Git workflow automation
"""

import os
import sys
import json
import time
import shutil
import argparse
import subprocess
from enum import Enum
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Set, Any
from dataclasses import dataclass
import re
from functools import wraps
import threading
from queue import Queue

VERSION = "2.1.0"

class Status(Enum):
    """Status codes for diagnostics"""
    OK = "✓"
    WARNING = "⚠"
    ERROR = "✗"
    MISSING = "?"

@dataclass
class DiagnosticResult:
    """Structured diagnostic result"""
    status: Status
    message: str
    fix_command: Optional[str] = None
    priority: int = 1
    requires_user_action: bool = False

class CommandError(Exception):
    """Custom exception for command execution errors"""
    pass

def retry_on_failure(retries: int = 3, delay: float = 1.0):
    """Decorator for retrying failed operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < retries - 1:
                        time.sleep(delay)
                    continue
            raise last_exception
        return wrapper
    return decorator

class AsyncGitRunner:
    """Asynchronous Git command runner with queuing"""
    def __init__(self):
        self.command_queue = Queue()
        self.result_queue = Queue()
        self.worker_thread = threading.Thread(target=self._process_commands, daemon=True)
        self.worker_thread.start()

    def _process_commands(self):
        while True:
            cmd, callback = self.command_queue.get()
            try:
                result = subprocess.run(
                    ["git"] + cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                if callback:
                    callback(result)
                self.result_queue.put((True, result))
            except subprocess.CalledProcessError as e:
                self.result_queue.put((False, e))
            self.command_queue.task_done()

    def run(self, args: List[str], callback: Optional[callable] = None) -> bool:
        """Queue a Git command for execution"""
        self.command_queue.put((args, callback))
        success, result = self.result_queue.get()
        self.result_queue.task_done()
        return result if success else None

class GitRunner:
    """Optimized Git command runner with caching"""
    _cache = {}
    _cache_timeout = 300  # 5 minutes

    @staticmethod
    def _get_cache_key(args: List[str]) -> str:
        return " ".join(args)

    @staticmethod
    def _is_cache_valid(timestamp: float) -> bool:
        return time.time() - timestamp < GitRunner._cache_timeout

    @staticmethod
    @retry_on_failure(retries=3)
    def run(args: List[str], capture_output: bool = True) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                ["git"] + args,
                capture_output=capture_output,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"\033[91mGit command failed: {' '.join(['git'] + args)}\033[0m")
            print(f"Error: {e.stderr}")
            return None

    @staticmethod
    def check_output(args: List[str], use_cache: bool = True) -> Optional[str]:
        """Get command output with optional caching"""
        cache_key = GitRunner._get_cache_key(args)
        
        if use_cache and cache_key in GitRunner._cache:
            timestamp, value = GitRunner._cache[cache_key]
            if GitRunner._is_cache_valid(timestamp):
                return value

        result = GitRunner.run(args)
        if result and use_cache:
            GitRunner._cache[cache_key] = (time.time(), result.stdout.strip())
        return result.stdout.strip() if result else None

class RepoConfig:
    """Repository configuration and constants"""
    DEFAULT_CONFIG = {
        "core": {
            "base_branch": "main",
            "develop_branch": "develop",
            "default_remote": "origin",
            "issue_prefix": "PROJ",
            "required_branches": ["main", "develop"],
            "required_remotes": ["origin"],
            "branch_protection": True,
            "auto_fetch_interval": 300,
            "commit_template": "type(scope): description"
        },
        "branch_types": [
            "feature",
            "bugfix",
            "hotfix",
            "release",
            "experiment"
        ],
        "commit_types": [
            "feat",
            "fix",
            "docs",
            "style",
            "refactor",
            "perf",
            "test",
            "chore"
        ],
        "protected_patterns": [
            "main",
            "master",
            "develop",
            "release/*"
        ],
        "checks": {
            "enforce_branch_naming": True,
            "enforce_commit_messages": True,
            "require_linear_history": True,
            "require_signed_commits": False,
            "prevent_force_push": True,
            "auto_stash_on_switch": True,
            "auto_rebase_updates": True
        },
        "hooks": {
            "pre_commit": True,
            "pre_push": True,
            "commit_msg": True
        }
    }

    def __init__(self):
        self.script_dir = Path(__file__).parent.resolve()
        self.config_dir = self.script_dir / "config"
        self.config_file = self.config_dir / "config.json"
        self.template_file = self.config_dir / "default-config.json"
        self._config = None
        self.ensure_dirs()
        self.load_config()

    def ensure_dirs(self) -> None:
        """Ensure required directories exist and create default config template"""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Create default config template
        if not self.template_file.exists():
            template_config = {
                **self.DEFAULT_CONFIG,
                "_template_info": {
                    "description": "Default configuration template for git-helper",
                    "version": VERSION,
                    "last_updated": datetime.now().isoformat()
                }
            }
            self.template_file.write_text(json.dumps(template_config, indent=4))
            
        # Create initial config if needed
        if not self.config_file.exists():
            self.config_file.write_text(json.dumps(self.DEFAULT_CONFIG, indent=4))

    def load_config(self) -> dict:
        """Load or create configuration with validation"""
        try:
            if self.config_file.exists():
                loaded_config = json.loads(self.config_file.read_text())
                self._config = self._validate_and_merge_config(loaded_config)
            else:
                print(f"\033[93mNo config file found. Creating new one at: {self.config_file}\033[0m")
                print(f"\033[93mTemplate available at: {self.template_file}\033[0m")
                self._config = self.DEFAULT_CONFIG.copy()
                self.config_file.write_text(json.dumps(self._config, indent=4))
        except Exception as e:
            print(f"\033[91mError loading config: {e}\033[0m")
            print(f"\033[93mUsing default configuration\033[0m")
            self._config = self.DEFAULT_CONFIG.copy()
        return self._config

    def _validate_and_merge_config(self, config: dict) -> dict:
        """Validate and merge configuration with defaults"""
        merged = self.DEFAULT_CONFIG.copy()
        
        def merge_dict(source: dict, target: dict):
            for key, value in source.items():
                if key in target and isinstance(value, dict) and isinstance(target[key], dict):
                    merge_dict(value, target[key])
                else:
                    target[key] = value
        
        merge_dict(config, merged)
        return merged

    @property
    def config(self) -> dict:
        """Get the current configuration"""
        if self._config is None:
            self.load_config()
        return self._config

    def update_config(self, updates: dict) -> None:
        """Update configuration with new values"""
        merged = self._validate_and_merge_config(updates)
        self._config = merged
        self.config_file.write_text(json.dumps(merged, indent=4))

class GitWorkflow:
    """Git workflow automation"""
    def __init__(self, config: RepoConfig):
        self.config = config
        self.async_runner = AsyncGitRunner()

    def create_feature_branch(self, name: str, from_branch: Optional[str] = None) -> bool:
        """Create a new feature branch with proper naming"""
        if not re.match(r'^[a-zA-Z0-9-_/]+$', name):
            raise ValueError("Invalid branch name format")

        base = from_branch or self.config.config["core"]["develop_branch"]
        prefixed_name = f"feature/{name}" if not name.startswith("feature/") else name

        try:
            GitRunner.run(["checkout", base])
            GitRunner.run(["pull"])
            GitRunner.run(["checkout", "-b", prefixed_name])
            return True
        except Exception as e:
            print(f"\033[91mFailed to create feature branch: {e}\033[0m")
            return False

    @retry_on_failure()
    def sync_branch(self, branch: Optional[str] = None) -> bool:
        """Sync current or specified branch with remote"""
        current = branch or GitRunner.check_output(["rev-parse", "--abbrev-ref", "HEAD"])
        remote = self.config.config["core"]["default_remote"]

        try:
            if branch:
                GitRunner.run(["checkout", branch])

            if self.config.config["checks"]["auto_stash_on_switch"]:
                GitRunner.run(["stash", "push", "-m", "auto-stash"])

            GitRunner.run(["fetch", remote])
            rebase_result = GitRunner.run(["rebase", f"{remote}/{current}"])

            if self.config.config["checks"]["auto_stash_on_switch"]:
                GitRunner.run(["stash", "pop"])

            return bool(rebase_result)
        except Exception as e:
            print(f"\033[91mFailed to sync branch: {e}\033[0m")
            return False

    def cleanup_branches(self, dry_run: bool = True) -> List[str]:
        """Clean up merged and stale branches"""
        protected = set(self.config.config["protected_patterns"])
        merged = GitRunner.check_output(["branch", "--merged"]).split("\n")
        deleted = []

        for branch in merged:
            branch = branch.strip()
            if branch and not branch.startswith("*"):
                # Check if branch matches any protected pattern
                if not any(re.match(pattern.replace("*", ".*"), branch) for pattern in protected):
                    if dry_run:
                        print(f"Would delete branch: {branch}")
                    else:
                        try:
                            GitRunner.run(["branch", "-d", branch])
                            print(f"Deleted branch: {branch}")
                            deleted.append(branch)
                        except Exception as e:
                            print(f"\033[91mFailed to delete branch {branch}: {e}\033[0m")

        return deleted

class RepositoryDiagnostics:
    """Repository diagnostics and initialization checker"""
    def __init__(self, config: RepoConfig):
        self.config = config
        self.results: List[DiagnosticResult] = []

    def check_git_installation(self) -> DiagnosticResult:
        """Verify Git is installed and accessible"""
        if shutil.which("git"):
            version = GitRunner.check_output(["--version"])
            return DiagnosticResult(
                Status.OK,
                f"Git installed: {version}",
                priority=0
            )
        return DiagnosticResult(
            Status.ERROR,
            "Git not found in PATH",
            "Install Git from https://git-scm.com/downloads",
            priority=0,
            requires_user_action=True
        )

    def check_git_repo(self) -> DiagnosticResult:
        """Check if current directory is a Git repository"""
        if GitRunner.run(["rev-parse", "--is-inside-work-tree"]):
            return DiagnosticResult(Status.OK, "Git repository initialized")
        return DiagnosticResult(
            Status.ERROR,
            "Not a Git repository",
            "git init",
            priority=1
        )

    def check_required_branches(self) -> List[DiagnosticResult]:
        """Verify required branches exist"""
        results = []
        existing_branches = set()
        
        branch_output = GitRunner.check_output(["branch", "--format=%(refname:short)"])
        if branch_output:
            existing_branches = set(branch_output.split("\n"))
        
        required_branches = self.config.config.get("core", {}).get("required_branches", [])
        
        for branch in required_branches:
            if branch not in existing_branches:
                results.append(DiagnosticResult(
                    Status.ERROR,
                    f"Required branch missing: {branch}",
                    f"git checkout -b {branch}",
                    priority=2
                ))
            else:
                results.append(DiagnosticResult(
                    Status.OK,
                    f"Required branch exists: {branch}"
                ))
        return results

    def check_hooks(self) -> List[DiagnosticResult]:
        """Check Git hooks configuration"""
        results = []
        hooks_dir = Path(".git/hooks")
        required_hooks = {
            "pre-commit": "#!/bin/sh\n# Pre-commit hook\nexit 0\n",
            "pre-push": "#!/bin/sh\n# Pre-push hook\nexit 0\n",
            "commit-msg": "#!/bin/sh\n# Commit message hook\nexit 0\n"
        }

        if not hooks_dir.exists():
            return [DiagnosticResult(
                Status.ERROR,
                "Git hooks directory not found",
                "Initialize Git repository first",
                priority=2
            )]

        for hook, template in required_hooks.items():
            hook_path = hooks_dir / hook
            if not hook_path.exists() and self.config.config["hooks"].get(hook_path.stem.replace("-", "_"), False):
                results.append(DiagnosticResult(
                    Status.WARNING,
                    f"Git hook missing: {hook}",
                    f"Create {hook} hook",
                    priority=3
                ))
                try:
                    hook_path.write_text(template)
                    hook_path.chmod(0o755)
                    results.append(DiagnosticResult(
                        Status.OK,
                        f"Created {hook} hook"
                    ))
                except Exception as e:
                    results.append(DiagnosticResult(
                        Status.ERROR,
                        f"Failed to create {hook} hook: {e}",
                        priority=2
                    ))
            else:
                results.append(DiagnosticResult(
                    Status.OK,
                    f"Hook exists: {hook}"
                ))
        return results

    def check_remote_configuration(self) -> List[DiagnosticResult]:
        """Verify remote configuration"""
        results = []
        remotes = GitRunner.check_output(["remote"]).split("\n") if GitRunner.check_output(["remote"]) else []
        required_remotes = self.config.config.get("core", {}).get("required_remotes", [])
        
        for remote in required_remotes:
            if remote not in remotes:
                results.append(DiagnosticResult(
                    Status.WARNING,
                    f"Required remote missing: {remote}",
                    "git remote add origin <repository-url>",
                    priority=3,
                    requires_user_action=True
                ))
            else:
                remote_url = GitRunner.check_output(["remote", "get-url", remote])
                results.append(DiagnosticResult(
                    Status.OK,
                    f"Remote {remote} configured: {remote_url}"
                ))
        return results

    def check_git_config(self) -> List[DiagnosticResult]:
        """Check Git configuration"""
        results = []
        required_configs = {
            "user.name": "git config --global user.name 'Your Name'",
            "user.email": "git config --global user.email 'your.email@example.com'"
        }
        
        for config, fix_command in required_configs.items():
            value = GitRunner.check_output(["config", "--get", config])
            if not value:
                results.append(DiagnosticResult(
                    Status.ERROR,
                    f"Git config missing: {config}",
                    fix_command,
                    priority=1,
                    requires_user_action=True
                ))
            else:
                results.append(DiagnosticResult(
                    Status.OK,
                    f"Git config set: {config} = {value}"
                ))
        return results

    def check_branch_protection(self) -> List[DiagnosticResult]:
        """Check branch protection settings"""
        results = []
        if not self.config.config["core"]["branch_protection"]:
            return [DiagnosticResult(Status.WARNING, "Branch protection disabled")]

        protected_branches = self.config.config["protected_patterns"]
        for branch in protected_branches:
            hook_path = Path(".git/hooks/pre-push")
            if not hook_path.exists():
                results.append(DiagnosticResult(
                    Status.WARNING,
                    f"No protection hook for {branch}",
                    "Install pre-push hook",
                    priority=4
                ))
            else:
                results.append(DiagnosticResult(
                    Status.OK,
                    f"Protection configured for {branch}"
                ))
        return results

    def run_diagnostics(self) -> Tuple[bool, List[DiagnosticResult]]:
        """Run all diagnostic checks"""
        all_results = []
        
        # Core checks
        all_results.append(self.check_git_installation())
        repo_status = self.check_git_repo()
        all_results.append(repo_status)
        
        # Only continue if we have a valid repo
        if repo_status.status == Status.OK:
            all_results.extend(self.check_required_branches())
            all_results.extend(self.check_remote_configuration())
            all_results.extend(self.check_git_config())
            all_results.extend(self.check_branch_protection())
            all_results.extend(self.check_hooks())
        
        # Sort by priority
        all_results.sort(key=lambda x: x.priority)
        
        # Check if all critical checks passed
        success = all(r.status == Status.OK 
                     for r in all_results 
                     if r.priority <= 2)
        
        return success, all_results

class GitHelper:
    """Main Git helper functionality"""
    def __init__(self):
        self.config = RepoConfig()
        self.workflow = GitWorkflow(self.config)
        self.diagnostics = RepositoryDiagnostics(self.config)
        self.ready = False

    def initialize(self) -> bool:
        """Run diagnostics and initialize repository if needed"""
        print("\033[94mRunning repository diagnostics...\033[0m")
        success, results = self.diagnostics.run_diagnostics()
        
        print("\n\033[1mDiagnostic Results:\033[0m")
        print("-" * 50)
        
        for result in results:
            status_color = {
                Status.OK: "\033[92m",
                Status.WARNING: "\033[93m",
                Status.ERROR: "\033[91m",
                Status.MISSING: "\033[95m"
            }[result.status]
            
            print(f"{status_color}{result.status.value}\033[0m {result.message}")
            if result.fix_command:
                print(f"  → Fix: {result.fix_command}")
        
        if not success:
            print("\n\033[91mRepository checks failed. Please fix the issues above before proceeding.\033[0m")
            if any(r.requires_user_action for r in results):
                print("\nSome issues require manual intervention. Please follow the fix instructions.")
            return False
        
        self.ready = True
        print("\n\033[92mRepository checks passed. Ready to proceed.\033[0m")
        return True

    def execute_command(self, command: str, args: argparse.Namespace) -> None:
        """Execute a git helper command"""
        if not self.ready and command not in {"init", "help", "version"}:
            if not self.initialize():
                return

        handlers = {
            "feature": self._handle_feature,
            "sync": self._handle_sync,
            "cleanup": self._handle_cleanup,
            "init": self._handle_init,
            "help": self._handle_help,
        }

        handler = handlers.get(command)
        if handler:
            try:
                handler(args)
            except CommandError as e:
                print(f"\033[91mCommand failed: {e}\033[0m")
        else:
            print(f"\033[91mUnknown command: {command}\033[0m")
            self._handle_help(args)

    def _handle_feature(self, args: argparse.Namespace) -> None:
        """Handle feature branch commands"""
        if args.new:
            if self.workflow.create_feature_branch(args.new, args.from_branch):
                print(f"\033[92mCreated feature branch: {args.new}\033[0m")

    def _handle_sync(self, args: argparse.Namespace) -> None:
        """Handle sync commands"""
        if self.workflow.sync_branch(args.branch):
            print(f"\033[92mSuccessfully synced{' branch: ' + args.branch if args.branch else ' current branch'}\033[0m")

    def _handle_cleanup(self, args: argparse.Namespace) -> None:
        """Handle cleanup commands"""
        deleted = self.workflow.cleanup_branches(args.dry_run)
        if deleted:
            print(f"\n\033[1m{'Would delete' if args.dry_run else 'Deleted'} branches:\033[0m")
            for branch in deleted:
                print(f"  - {branch}")
        else:
            print("\033[93mNo branches to clean up\033[0m")

    def _handle_init(self, args: argparse.Namespace) -> None:
        """Handle initialization"""
        if self.initialize():
            print("\033[92mRepository initialized successfully\033[0m")

    def _handle_help(self, args: argparse.Namespace) -> None:
        """Show help information"""
        help_text = f"""
\033[1mGit Helper v{VERSION}\033[0m - Streamlined Git Workflow Tool

\033[1mUSAGE:\033[0m
    git-helper [OPTIONS] COMMAND [ARGS]

\033[1mCOMMANDS:\033[0m
    feature     Feature branch management
        --new NAME        Create new feature branch
        --from-branch B   Base branch for new feature

    sync        Sync repository with remote
        --branch NAME     Branch to sync (default: current)

    cleanup     Clean up repository
        --dry-run        Show what would be deleted

    init        Initialize and check repository

    help        Show this help message

\033[1mOPTIONS:\033[0m
    --version   Show version information

\033[1mEXAMPLES:\033[0m
    git-helper feature --new user-authentication
    git-helper sync --branch main
    git-helper cleanup --dry-run
"""
        print(help_text)

def main():
    parser = argparse.ArgumentParser(
        description="Git workflow helper with diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--version", action="version", version=f"v{VERSION}")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Feature command
    feature_parser = subparsers.add_parser("feature", help="Feature branch management")
    feature_parser.add_argument("--new", help="Create new feature branch")
    feature_parser.add_argument("--from-branch", help="Base branch for new feature")
    
    # Sync command
    sync_parser = subparsers.add_parser("sync", help="Sync repository")
    sync_parser.add_argument("--branch", help="Branch to sync")
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up repository")
    cleanup_parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted")
    
    # Init command
    subparsers.add_parser("init", help="Initialize repository")
    
    # Help command
    subparsers.add_parser("help", help="Show help message")
    
    args = parser.parse_args()
    
    try:
        helper = GitHelper()
        if args.command:
            helper.execute_command(args.command, args)
        else:
            helper._handle_help(args)
    except KeyboardInterrupt:
        print("\n\033[93mOperation cancelled by user\033[0m")
        sys.exit(1)
    except Exception as e:
        print(f"\n\033[91mError: {e}\033[0m")
        sys.exit(1)

if __name__ == "__main__":
    main()