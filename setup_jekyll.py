#!/usr/bin/env python3
import os
import subprocess
import sys

def run_command(cmd, capture_output=False):
    """Helper to run a command and return its output if needed."""
    try:
        if capture_output:
            result = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
            return result.decode().strip()
        else:
            subprocess.check_call(cmd)
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error while running command: {' '.join(cmd)}")
        print(e.output.decode() if e.output else e)
        sys.exit(1)
    except FileNotFoundError:
        return None

def check_ruby():
    ruby_version = run_command(["ruby", "--version"], capture_output=True)
    if ruby_version is None:
        print("Ruby is not installed. Please install Ruby (e.g., via Homebrew or from ruby-lang.org) and try again.")
        sys.exit(1)
    print(f"Ruby is installed: {ruby_version}")

def check_homebrew():
    brew_version = run_command(["brew", "--version"], capture_output=True)
    if brew_version is None:
        print("Homebrew is not installed. (Optional) Consider installing Homebrew from https://brew.sh for easier package management.")
    else:
        print(f"Homebrew is installed: {brew_version}")

def check_and_install_bundler():
    bundler_version = run_command(["bundle", "--version"], capture_output=True)
    if bundler_version is None:
        print("Bundler is not installed. Installing Bundler now...")
        run_command(["gem", "install", "bundler"])
    else:
        print(f"Bundler is installed: {bundler_version}")

def create_gemfile():
    gemfile_path = os.path.join(os.getcwd(), "Gemfile")
    gemfile_content = (
        'source "https://rubygems.org"\n'
        'gem "github-pages", group: :jekyll_plugins\n'
    )
    if not os.path.exists(gemfile_path):
        with open(gemfile_path, "w") as f:
            f.write(gemfile_content)
        print("Gemfile created.")
    else:
        print("Gemfile already exists.")

def bundle_install():
    print("Running bundle install...")
    run_command(["bundle", "install"])

def serve_site():
    print("Starting Jekyll server (press Ctrl+C to stop)...")
    try:
        subprocess.check_call(["bundle", "exec", "jekyll", "serve"])
    except KeyboardInterrupt:
        print("\nJekyll server stopped.")
    except Exception as e:
        print("Failed to start Jekyll server:", e)
        sys.exit(1)

def create_config_file():
    config_path = os.path.join(os.getcwd(), "_config.yml")
    config_content = (
        'exclude:\n'
        '  - data\n'
        '  - utils\n'
        '  - .git\n'
        '  - .gitignore\n'
        '  - README.md\n'
        '  - Gemfile\n'
        '  - Gemfile.lock\n'
        '  - .venv\n'
        '  - __pycache__\n'
        'defaults:\n'
        '  -\n'
        '    scope:\n'
        '      path: ""\n'
        '    values:\n'
        '      layout: "main"\n'
    )
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            f.write(config_content)
        print("_config.yml created.")
    else:
        print("_config.yml already exists.")

def main():
    print("Checking Ruby installation...")
    check_ruby()

    print("Checking Homebrew (optional)...")
    check_homebrew()

    print("Checking and installing Bundler if needed...")
    check_and_install_bundler()

    print("Ensuring Gemfile is present...")
    create_gemfile()

    print("Creating Jekyll configuration...")
    create_config_file()

    bundle_install()
    serve_site()

if __name__ == "__main__":
    main()
