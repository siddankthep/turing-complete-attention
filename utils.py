# Just a pretty print, no need to read
def print_in_box(text: str):
    lines = text.split("\n")
    max_length = max(len(line) for line in lines)
    border = "+" + "-" * (max_length + 2) + "+"

    print(border)
    for line in lines:
        print(f"| {line.ljust(max_length)} |")
    print(border)
