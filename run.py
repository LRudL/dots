import wandb
import sys

from dots.experiment import run_experiment

def remove_quotes(string):
    result = ""
    in_quote = False
    for char in string:
        if char == '"':
            in_quote = not in_quote
        elif not in_quote:
            result += char
    return result


def main():
    # Process command-line arguments
    args = sys.argv[1:]
    
    # Form a dictionary from the command-line arguments
    config = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if "=" in remove_quotes(arg):
            # then we're taking in arguments of the form:
            # "--key=value"
            key, value = arg.split("=") 
            config[key.strip('--')] = value 
            i += 1
        else:
            # then we're taking in arguments of the form:
            # "-key", "value"
            key = args[i].strip('--')
            value = args[i+1]
            config[key] = value
            i += 2

    # Run the experiment with the provided configuration
    run_experiment(config)

if __name__ == '__main__':
    main()
