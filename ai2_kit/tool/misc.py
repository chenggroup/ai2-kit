import sys

def format_cp2k_inp(indent: int = 2):
    """
    cp2k uses `&keyword` to start a block, and `&end` to end a block, for example,
    &block1 xxx
      clause 1
      clause 2
      &block2
       clause 3
      &end block2
    &end block1

    To format the file, this function will:
    1. remove leding and trailing whitespaces
    2. autoindent the file
    3. print the file to stdout
    """

    indent_level = 0
    for line in sys.stdin.readlines():
        assert indent_level >= 0, 'indent level should not be negative'
        line = line.strip()
        line_low = line.lower()
        if line_low.startswith('&end'):
            indent_level -= indent
            print(' ' * indent_level + line)
        else:
            print(' ' * indent_level + line)
            if line_low.startswith('&'):
                indent_level += indent

cmd_entry = {
    'format_cp2k_inp': format_cp2k_inp,
}
