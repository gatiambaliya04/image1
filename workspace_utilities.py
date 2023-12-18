import os
import shutil
import glob
import matplotlib.pyplot as plt
import pandas as pd
import logging
from datetime import datetime

def list_files_in_directory(directory):
    """List all files in a directory."""
    return os.listdir(directory)

def create_directory(directory):
    """Create a new directory."""
    os.makedirs(directory, exist_ok=True)

def delete_directory(directory):
    """Delete an existing directory."""
    shutil.rmtree(directory, ignore_errors=True)

def rename_file(old_name, new_name):
    """Rename a file."""
    os.rename(old_name, new_name)

def change_working_directory(directory):
    """Change the current working directory."""
    os.chdir(directory)

def plot_histogram(data, title='Histogram', xlabel='Values', ylabel='Frequency'):
    """Plot a histogram."""
    plt.hist(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def summary_statistics(data):
    """Calculate summary statistics for a dataset."""
    return pd.DataFrame(data).describe()

def setup_logging(log_file='workspace_log.txt'):
    """Setup logging configuration."""
    logging.basicConfig(filename=log_file, level=logging.INFO)

def log_message(message):
    """Log a message."""
    logging.info(f"{datetime.now()} - {message}")

def current_time():
    """Get the current time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Example usage:
if __name__ == "__main__":
    directory_path = 'example_directory'
    
    create_directory(directory_path)
    list_of_files = list_files_in_directory(directory_path)
    log_message(f"Files in directory: {list_of_files}")
    
    data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]
    plot_histogram(data, title='Example Histogram', xlabel='Values', ylabel='Frequency')
    
    summary_stats = summary_statistics(data)
    log_message(f"Summary Statistics:\n{summary_stats}")
