import subprocess
import os

def bash_command(cmd, **kwargs):
    '''Run command from the bash shell'''
    process = subprocess.Popen(['/bin/bash', '-c', cmd], **kwargs)
    return process.communicate()[0]

def get_cadence():
    """Read cadence value from a file named 'cadence', or default to 1."""
    default_cadence = 1
    cadence_file = "cadence"
    if os.path.exists(cadence_file):
        try:
            with open(cadence_file, "r") as file:
                cadence = int(file.read().strip())
                return cadence
        except (ValueError, IOError) as e:
            print(f"Error reading cadence from file: {e}. Using default cadence: {default_cadence}.")
            return default_cadence
    return default_cadence
